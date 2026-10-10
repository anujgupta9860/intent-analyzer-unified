"""Serving client for the unified ("giant") intent analyzer.

One checkpoint (models/unified/) serves all agents. The agent name in
each request selects which per-agent heads answer — each agent keeps
its own domain-specific question set (choice taxonomy + noul flags)
from src/analyzer_questions.py.

Response contract is identical to the per-agent AnalyzerClient
(src/analyzers.py), so the router -> analyzer handoff is unchanged:
POST /analyze {"query", "worker_agent"}.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .analyzer_questions import (
    AGENT_IDS,
    choice_options,
    choice_question,
    noul_questions,
)
from .analyzers import AnalyzerDecision, AnalyzerError
from .unified_model import UnifiedAnalyzer

log = logging.getLogger(__name__)


class UnifiedAnalyzerClient:
    """One loaded unified checkpoint answering for any agent."""

    def __init__(self, checkpoint: str,
                 confidence_threshold: float = 0.6) -> None:
        self.checkpoint = checkpoint
        self.confidence_threshold = confidence_threshold
        self._tokenizer = None
        self._model = None
        self._specs: dict[str, dict] = {}

    # ------------------------------------------------------------------ API
    def analyze(self, text: str, agent: str) -> AnalyzerDecision:
        if agent not in AGENT_IDS:
            raise AnalyzerError(f"unknown agent: {agent!r}")
        self._ensure_loaded()
        import torch

        spec = self._specs[agent]
        agent_idx = list(AGENT_IDS).index(agent)
        started = time.monotonic()
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128)
        agent_ids = torch.tensor([agent_idx])
        with torch.no_grad():
            out = self._model(**inputs, agent_ids=agent_ids)
            choice_probs = torch.softmax(
                out["choice_logits"] / spec["temp_choice"], dim=-1)[0]
            noul_probs = torch.sigmoid(
                out["noul_logits"] / torch.tensor(spec["temp_noul"]))[0]
        latency_ms = (time.monotonic() - started) * 1000.0

        choice_labels = spec["choice_labels"]
        noul_qids = spec["noul_qids"]
        top_idx = int(choice_probs.argmax())
        choice = choice_labels[top_idx]
        confidence = round(float(choice_probs[top_idx]), 3)
        probs = {l: round(float(choice_probs[i]), 4)
                 for i, l in enumerate(choice_labels)}
        noul = {qid: round(float(noul_probs[i]), 3)
                for i, qid in enumerate(noul_qids)}

        return AnalyzerDecision(
            agent=agent,
            choice_question=choice_question(agent),
            choice=choice,
            choice_confidence=confidence,
            choice_probabilities=probs,
            noul=noul,
            needs_human=confidence < self.confidence_threshold,
            model=f"unified:{self.checkpoint}",
            latency_ms=round(latency_ms, 1),
        )

    def available(self) -> dict[str, bool]:
        """Which agents the unified checkpoint covers."""
        self._ensure_loaded()
        return {agent: agent in self._specs for agent in AGENT_IDS}

    # -------------------------------------------------------------- internals
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise AnalyzerError(
                "transformers is required; pip install transformers torch"
            ) from exc
        ckpt = Path(self.checkpoint)
        calib_path = ckpt / "calibration.json"
        if not calib_path.exists():
            raise AnalyzerError(
                f"unified checkpoint not found: {ckpt} "
                "(train one: python train/finetune_unified.py "
                "--data train/data --out models/unified)")
        calib = json.loads(calib_path.read_text())
        if calib.get("model_type") != "unified":
            raise AnalyzerError(
                f"{ckpt} is not a unified checkpoint "
                f"(model_type={calib.get('model_type')!r})")

        # Contract check per agent: checkpoint labels must match the
        # question spec byte-for-byte in order (same rule as per-agent).
        specs: dict[str, dict] = {}
        for agent in AGENT_IDS:
            a = calib.get("agents", {}).get(agent)
            if a is None:
                continue
            spec_choice = choice_options(agent)
            spec_noul = noul_questions(agent)
            if list(a.get("choice_labels", [])) != spec_choice:
                raise AnalyzerError(
                    f"unified checkpoint: agent {agent} choice labels "
                    f"{a.get('choice_labels')} != spec {spec_choice}")
            if list(a.get("noul_questions", [])) != spec_noul:
                raise AnalyzerError(
                    f"unified checkpoint: agent {agent} noul questions "
                    f"{a.get('noul_questions')} != spec {spec_noul}")
            if a.get("choice_question") != choice_question(agent):
                raise AnalyzerError(
                    f"unified checkpoint: agent {agent} choice_question "
                    f"mismatch")
            t_noul = [float(t) for t in a.get("temperatures_noul", [])]
            if len(t_noul) != len(spec_noul):
                raise AnalyzerError(
                    f"unified checkpoint: agent {agent} "
                    "temperatures_noul length mismatch")
            specs[agent] = {
                "choice_labels": spec_choice,
                "noul_qids": spec_noul,
                "temp_choice": float(a.get("temperature_choice", 1.0)),
                "temp_noul": t_noul,
            }
        if not specs:
            raise AnalyzerError(
                f"unified checkpoint {ckpt} covers no agents")
        self._specs = specs

        log.info("loading unified analyzer checkpoint %s ...", ckpt)
        self._tokenizer = AutoTokenizer.from_pretrained(ckpt)
        import torch
        from transformers import AutoConfig
        enc_config = AutoConfig.from_pretrained(ckpt)
        agent_specs = {a: len(s["noul_qids"]) for a, s in specs.items()}
        self._model = UnifiedAnalyzer.from_local_config(
            enc_config, agent_specs)
        state = torch.load(ckpt / "pytorch_model.bin", map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()
        log.info("unified analyzer loaded (agents=%s)", sorted(specs))
