"""Contract tests: training <-> serving consistency for the unified analyzer.

The 2026-10-07 lesson from intent-router-laya: training and serving must
use the question spec byte-identically, or confidence silently degrades.
These tests pin that contract for the unified (one-model) track:

1. Every dataset record's label keys match the spec for its agent.
2. Choice option order in the dataset builder == spec order (the order
   becomes label2id at train time; the checkpoint must agree at serve).
3. Serving (UnifiedAnalyzerClient) validates each agent's labels against
   the spec and raises on mismatch (tested with forged calibrations).
4. The unified model has one head pair per agent with matching dimensions.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analyzer_questions import (  # noqa: E402
    AGENT_IDS,
    ANALYZERS,
    choice_options,
    choice_question,
    noul_questions,
    question_ids,
)
from src.unified_analyzers import UnifiedAnalyzerClient  # noqa: E402
from src.analyzers import AnalyzerError  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def test_question_inventory_complete():
    for agent in AGENT_IDS:
        spec = ANALYZERS[agent]
        cq = choice_question(agent)
        assert spec["questions"][cq]["type"] == "choice"
        assert len(choice_options(agent)) >= 5
        assert len(noul_questions(agent)) >= 2
        for qid, q in spec["questions"].items():
            assert q["instructions"].strip()
            if q["type"] == "choice":
                assert len(q["criteria"]) >= 5


def test_dataset_records_match_spec():
    """Label keys in built datasets == the spec's question ids."""
    sys.path.insert(0, str(REPO / "train"))
    from build_dataset import build_agent_records
    for agent in AGENT_IDS:
        recs = build_agent_records(agent, repeat=2, seed=1)
        expected = set(question_ids(agent)) | {"query"}
        for r in recs:
            assert set(r.keys()) == expected, (agent, set(r.keys()))
            assert r[choice_question(agent)] in choice_options(agent)
            for n in noul_questions(agent):
                assert r[n] in (0.0, 1.0)


def test_choice_order_stable():
    """Option order is insertion order — the same order train and serve use."""
    for agent in AGENT_IDS:
        opts = choice_options(agent)
        assert opts == list(
            ANALYZERS[agent]["questions"][choice_question(agent)]["criteria"].keys())


def _good_calibration() -> dict:
    return {
        "model_type": "unified",
        "agents": {
            agent: {
                "choice_question": choice_question(agent),
                "choice_labels": choice_options(agent),
                "noul_questions": noul_questions(agent),
                "temperature_choice": 1.0,
                "temperatures_noul": [1.0] * len(noul_questions(agent)),
            }
            for agent in AGENT_IDS
        },
    }


def test_unified_rejects_label_mismatch():
    """A unified checkpoint whose labels disagree with the spec must fail."""
    calib = _good_calibration()
    calib["agents"]["billing"]["choice_labels"] = list(
        reversed(choice_options("billing")))  # WRONG order
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d)
        (ckpt / "calibration.json").write_text(json.dumps(calib))
        client = UnifiedAnalyzerClient(str(ckpt))
        with pytest.raises(AnalyzerError, match="choice labels"):
            client._ensure_loaded()


def test_unified_rejects_wrong_model_type():
    """A per-agent checkpoint must not load as unified."""
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d)
        (ckpt / "calibration.json").write_text(json.dumps({
            "agent": "billing",
            "choice_labels": choice_options("billing"),
        }))
        client = UnifiedAnalyzerClient(str(ckpt))
        with pytest.raises(AnalyzerError, match="not a unified checkpoint"):
            client._ensure_loaded()


def test_unified_rejects_empty_agents():
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d)
        (ckpt / "calibration.json").write_text(json.dumps(
            {"model_type": "unified", "agents": {}}))
        client = UnifiedAnalyzerClient(str(ckpt))
        with pytest.raises(AnalyzerError, match="covers no agents"):
            client._ensure_loaded()


def test_unified_model_head_dimensions():
    """One choice+noul head per agent, sized from the spec."""
    torch = pytest.importorskip("torch")
    from src.unified_model import UnifiedAnalyzer

    # Build from a tiny local config to avoid downloads.
    from transformers import BertConfig
    cfg = BertConfig(hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32)
    specs = {a: len(noul_questions(a)) for a in AGENT_IDS}
    model = UnifiedAnalyzer.from_local_config(cfg, specs)
    assert set(model.choice_heads.keys()) == set(AGENT_IDS)
    assert set(model.noul_heads.keys()) == set(AGENT_IDS)
    for agent in AGENT_IDS:
        assert model.choice_heads[agent].out_features == len(
            choice_options(agent))
        assert model.noul_heads[agent].out_features == len(
            noul_questions(agent))
