"""Tests for the decision-tree policy engine."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy.engine import Decision, PolicyEngine, PolicyError

REPO = Path(__file__).resolve().parent.parent
TREE = REPO / "policy" / "decision-tree.yaml"


def engine() -> PolicyEngine:
    return PolicyEngine(TREE)


def out(agent: str, choice_q: str, choice: str, conf: float,
        **noul) -> dict:
    d = {"agent": agent, choice_q: choice, "choice_confidence": conf,
         "needs_human": conf < 0.6}
    d.update(noul)
    return d


def test_starter_tree_loads_and_validates():
    eng = engine()
    assert len(eng.rules) >= 7
    assert eng.rules[-1]["name"].startswith("Fallback")


def test_auto_refund_rule_fires():
    d = engine().decide(out(
        "billing", "billing_action", "refund_request", 0.95,
        is_dispute=0.1, amount_mentioned=0.9))
    assert d.action == "process_refund"
    assert d.worker == "billing"
    assert d.risk == "low"


def test_dispute_beats_auto_refund():
    # same choice, but is_dispute high -> specialist rule (earlier match
    # would be auto-refund; the noul condition routes correctly)
    d = engine().decide(out(
        "billing", "billing_action", "refund_request", 0.95,
        is_dispute=0.8, amount_mentioned=0.9))
    assert d.action == "open_dispute_case"


def test_first_match_wins_order():
    d = engine().decide(out(
        "orders", "order_action", "track", 0.9, has_order_id=0.9))
    assert d.action == "lookup_order_status"
    d2 = engine().decide(out(
        "orders", "order_action", "track", 0.9, has_order_id=0.1))
    assert d2.action == "ask_clarify"


def test_delete_account_always_human():
    d = engine().decide(out(
        "account", "account_action", "delete_account", 1.0,
        is_irreversible=1.0))
    assert d.action == "escalate_human"
    assert d.risk == "high"


def test_low_confidence_escalates():
    d = engine().decide(out(
        "sales", "sales_intent", "pricing", 0.3, is_ready_to_buy=0.1))
    assert d.action == "escalate_human"


def test_fallback_catches_everything():
    d = engine().decide(out(
        "support", "support_topic", "feedback", 0.99,
        needs_troubleshooting=0.0))
    assert d.action == "ask_clarify"


def test_rejects_unknown_agent():
    eng = engine()
    with pytest.raises(PolicyError, match="unknown agent"):
        eng.validate_tree({"rules": [
            {"name": "x", "when": {"agent": "nope"}, "then": {"action": "a"}}]})


def test_rejects_unknown_choice():
    eng = engine()
    with pytest.raises(PolicyError, match="unknown choice"):
        eng.validate_tree({"rules": [
            {"name": "x",
             "when": {"agent": "billing", "choice": "fly_to_moon"},
             "then": {"action": "a"}}]})


def test_rejects_unknown_noul():
    eng = engine()
    with pytest.raises(PolicyError, match="unknown Noul"):
        eng.validate_tree({"rules": [
            {"name": "x",
             "when": {"agent": "billing",
                      "noul": {"is_moon": {">": 0.5}}},
             "then": {"action": "a"}}]})


def test_rejects_bad_operator():
    eng = engine()
    with pytest.raises(PolicyError, match="unknown operator"):
        eng.validate_tree({"rules": [
            {"name": "x", "when": {"confidence": {"~=": 0.5}},
             "then": {"action": "a"}}]})


def test_rejects_empty_rules():
    eng = engine()
    with pytest.raises(PolicyError, match="non-empty"):
        eng.validate_tree({"rules": []})
