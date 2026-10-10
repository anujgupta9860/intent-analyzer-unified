"""Decision-tree policy engine: layman-configurable rules over analyzer output.

A non-technical person authors the decision logic as YAML
(policy/decision-tree.yaml) — optionally via the Studio UI (/studio).
The engine evaluates rules top-to-bottom, first match wins, and returns
a Decision the A2A agent layer can dispatch.

Rule conditions (all optional; a rule with `when: {}` matches everything):
  agent: billing                  # exact match on analyzer agent
  choice: refund_request           # exact match on the agent's choice
  confidence: {">=": 0.9}          # comparison on choice_confidence
  noul:                            # comparisons on Noul flag values
    is_dispute: {"<": 0.5}
    amount_mentioned: {">": 0.5}

Supported comparison operators: >, >=, <, <=, ==, !=

The `then` block:
  action: process_refund          # what to do (handled by the agent layer)
  worker: billing                 # which worker agent (null = no worker)
  risk: low                       # low | medium | high (gates auto-execution)
  message: "..."                  # human-readable note / user-facing text
  params: {...}                   # extra key/values passed to the action

Validation: every rule is checked against src/analyzer_questions.py at
load time — unknown agents, choices, or Noul questions are rejected
with a plain-English error naming the rule and field, so a layman
can't save a broken tree from the Studio UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.analyzer_questions import (
    AGENT_IDS,
    choice_options,
    choice_question,
    noul_questions,
)

_OPERATORS = (">", ">=", "<", "<=", "==", "!=")
_RISKS = ("low", "medium", "high")


@dataclass
class Decision:
    """The winning rule's decision, ready for the A2A agent layer."""
    rule_name: str
    action: str
    worker: str | None
    risk: str
    message: str = ""
    params: dict = field(default_factory=dict)
    analyzer_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "action": self.action,
            "worker": self.worker,
            "risk": self.risk,
            "message": self.message,
            "params": self.params,
        }


class PolicyError(RuntimeError):
    """Raised when the decision tree YAML is invalid."""


def _check_cmp(actual: float, op: str, expected: float) -> bool:
    if op == ">":
        return actual > expected
    if op == ">=":
        return actual >= expected
    if op == "<":
        return actual < expected
    if op == "<=":
        return actual <= expected
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    raise PolicyError(f"unknown operator {op!r}; use one of {_OPERATORS}")


class PolicyEngine:
    """Loads a decision-tree YAML and evaluates analyzer outputs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rules: list[dict] = []
        self.version: int = 1
        self.reload()

    # ------------------------------------------------------------------ API
    def reload(self) -> None:
        import yaml
        if not self.path.exists():
            raise PolicyError(
                f"decision tree not found: {self.path} "
                "(copy policy/decision-tree.yaml into place)")
        data = yaml.safe_load(self.path.read_text()) or {}
        self.version = data.get("version", 1)
        rules = data.get("rules", [])
        if not isinstance(rules, list) or not rules:
            raise PolicyError("decision tree must contain a non-empty 'rules' list")
        for i, rule in enumerate(rules):
            self._validate_rule(rule, i)
        self.rules = rules

    def decide(self, analyzer_output: dict) -> Decision:
        """Evaluate rules top-to-bottom; first full match wins."""
        agent = analyzer_output.get("agent", "")
        cq = choice_question(agent) if agent in AGENT_IDS else None
        choice = analyzer_output.get(cq) if cq else None
        confidence = float(analyzer_output.get("choice_confidence", 0.0))
        for rule in self.rules:
            if self._matches(rule.get("when", {}) or {}, agent, choice,
                             confidence, analyzer_output):
                then = rule.get("then", {}) or {}
                return Decision(
                    rule_name=rule["name"],
                    action=then.get("action", "noop"),
                    worker=then.get("worker"),
                    risk=then.get("risk", "medium"),
                    message=then.get("message", ""),
                    params=dict(then.get("params", {}) or {}),
                    analyzer_snapshot={
                        "agent": agent, "choice": choice,
                        "confidence": confidence},
                )
        # Unreachable when the tree ends with a catch-all, but safe.
        return Decision(rule_name="(no match)", action="escalate_human",
                        worker=None, risk="high",
                        message="No rule matched; escalated.")

    def validate_tree(self, data: dict) -> None:
        """Validate a candidate tree (from the Studio UI) without saving."""
        rules = data.get("rules", [])
        if not isinstance(rules, list) or not rules:
            raise PolicyError("decision tree must contain a non-empty 'rules' list")
        for i, rule in enumerate(rules):
            self._validate_rule(rule, i)

    # -------------------------------------------------------------- internals
    def _matches(self, when: dict, agent: str, choice: str | None,
                 confidence: float, output: dict) -> bool:
        if "agent" in when and when["agent"] != agent:
            return False
        if "choice" in when and when["choice"] != choice:
            return False
        if "confidence" in when:
            cond = when["confidence"]
            if not isinstance(cond, dict) or len(cond) != 1:
                return False
            op, val = next(iter(cond.items()))
            if op not in _OPERATORS:
                return False
            if not _check_cmp(confidence, op, float(val)):
                return False
        noul = when.get("noul", {}) or {}
        for qid, cond in noul.items():
            if not isinstance(cond, dict) or len(cond) != 1:
                return False
            op, val = next(iter(cond.items()))
            if op not in _OPERATORS:
                return False
            actual = output.get(qid)
            if actual is None:
                return False
            if not _check_cmp(float(actual), op, float(val)):
                return False
        return True

    def _validate_rule(self, rule: dict, i: int) -> None:
        where = f"rule #{i + 1} ({rule.get('name', 'unnamed')!r})"
        if not isinstance(rule, dict):
            raise PolicyError(f"{where}: each rule must be a mapping")
        for key in ("name", "when", "then"):
            if key not in rule:
                raise PolicyError(f"{where}: missing required key {key!r}")
        when = rule["when"] or {}
        then = rule["then"] or {}
        if not isinstance(when, dict) or not isinstance(then, dict):
            raise PolicyError(f"{where}: 'when' and 'then' must be mappings")

        agent = when.get("agent")
        if agent is not None and agent not in AGENT_IDS:
            raise PolicyError(
                f"{where}: unknown agent {agent!r}; "
                f"use one of {list(AGENT_IDS)}")

        choice = when.get("choice")
        if choice is not None:
            if agent is None:
                raise PolicyError(
                    f"{where}: 'choice' needs an 'agent' (choices differ "
                    f"per agent)")
            if choice not in choice_options(agent):
                raise PolicyError(
                    f"{where}: unknown choice {choice!r} for agent "
                    f"{agent!r}; use one of {choice_options(agent)}")

        conf = when.get("confidence")
        if conf is not None:
            self._validate_cmp(f"{where} -> confidence", conf)

        noul = when.get("noul", {}) or {}
        if not isinstance(noul, dict):
            raise PolicyError(f"{where}: 'noul' must be a mapping")
        for qid, cond in noul.items():
            if agent is None:
                raise PolicyError(
                    f"{where}: noul condition {qid!r} needs an 'agent' "
                    f"(Noul questions differ per agent)")
            if qid not in noul_questions(agent):
                raise PolicyError(
                    f"{where}: unknown Noul question {qid!r} for agent "
                    f"{agent!r}; use one of {noul_questions(agent)}")
            self._validate_cmp(f"{where} -> noul {qid!r}", cond)

        if "action" not in then:
            raise PolicyError(f"{where}: 'then' needs an 'action'")
        risk = then.get("risk", "medium")
        if risk not in _RISKS:
            raise PolicyError(
                f"{where}: unknown risk {risk!r}; use one of {_RISKS}")
        worker = then.get("worker")
        if worker is not None and worker not in AGENT_IDS:
            raise PolicyError(
                f"{where}: unknown worker {worker!r}; "
                f"use one of {list(AGENT_IDS)} or null")

    @staticmethod
    def _validate_cmp(where: str, cond) -> None:
        if not isinstance(cond, dict) or len(cond) != 1:
            raise PolicyError(
                f"{where}: use one comparison like {'>=': 0.9}")
        op = next(iter(cond))
        if op not in _OPERATORS:
            raise PolicyError(
                f"{where}: unknown operator {op!r}; use one of {_OPERATORS}")
        try:
            float(next(iter(cond.values())))
        except (TypeError, ValueError):
            raise PolicyError(f"{where}: comparison value must be a number")
