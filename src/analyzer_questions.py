"""Analyzer typed-decision spec — SINGLE SOURCE OF TRUTH.

Every question each agent's intent analyzer asks about a query is defined
here, with the EXACT instructions + criteria strings used both when
building training data (train/build_dataset.py) and when serving
(src/analyzers.py).

Prompt/label mismatch between training and serving silently degrades
confidence. This is the direct lesson from the intent-router-laya
2026-10-07 confidence-gap incident: training and serving must import
these strings verbatim. Do NOT rephrase them anywhere else.

Each agent owns one Choice question (its action taxonomy) plus Noul
flags (needs-x probabilities). Agents:

  billing : billing_action (6)  + 4 noul
  orders  : order_action   (6)  + 3 noul
  support : support_topic   (5)  + 2 noul
  account : account_action  (5)  + 2 noul
  sales   : sales_intent    (5)  + 2 noul
"""
from __future__ import annotations

import json
import re
from pathlib import Path

#: Agent id -> spec. Keys are stable and also used as the directory names
#: for per-agent datasets (train/data/<agent>.jsonl) and checkpoints
#: (models/<agent>/).
#: NOTE: AGENT_IDS is extended at the bottom of this module with any
#: UI-defined custom agents found in agents/*.json.
_BUILTIN_AGENT_IDS = ("billing", "orders", "support", "account", "sales")
AGENT_IDS = _BUILTIN_AGENT_IDS

ANALYZERS: dict[str, dict] = {
    "billing": {
        "choice_question": "billing_action",
        "questions": {
            "billing_action": {
                "type": "choice",
                "instructions": "What billing action does this customer need?",
                "criteria": {
                    "check_balance": "asking for current balance, amount due, invoice total",
                    "make_payment": "wants to pay a bill, pay now, set up payment",
                    "dispute_charge": "disputes a charge, says a charge is wrong or unknown",
                    "refund_request": "asks for money back, a refund, reversal of a charge",
                    "update_payment_method": "change card, update payment details, add a payment method",
                    "billing_info": "general billing questions that fit none of the above",
                },
            },
            "needs_payment_method": {
                "type": "noul",
                "instructions": "Does this request require the customer's payment credentials to proceed?",
            },
            "is_dispute": {
                "type": "noul",
                "instructions": "Is the customer disputing a charge?",
            },
            "amount_mentioned": {
                "type": "noul",
                "instructions": "Does the query mention a specific amount of money?",
            },
            "is_recurring": {
                "type": "noul",
                "instructions": "Is this about a subscription or recurring charge?",
            },
        },
    },
    "orders": {
        "choice_question": "order_action",
        "questions": {
            "order_action": {
                "type": "choice",
                "instructions": "What order action does this customer need?",
                "criteria": {
                    "track": "wants to track a shipment or know delivery status",
                    "cancel": "wants to cancel an order",
                    "modify": "wants to change items, quantity, or details of an order",
                    "return": "wants to return an item or start a return",
                    "reorder": "wants to buy the same items again",
                    "order_info": "general order questions that fit none of the above",
                },
            },
            "has_order_id": {
                "type": "noul",
                "instructions": "Does the query include an order identifier?",
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "Is this time-sensitive (needed urgently, deadline, gift)?",
            },
            "needs_address": {
                "type": "noul",
                "instructions": "Does handling this need the customer's shipping address?",
            },
        },
    },
    "support": {
        "choice_question": "support_topic",
        "questions": {
            "support_topic": {
                "type": "choice",
                "instructions": "What support topic is this customer asking about?",
                "criteria": {
                    "technical_issue": "something is broken, an error, a bug, not working",
                    "how_to": "asks how to do something, instructions, guidance",
                    "account_access": "login problems, password reset, locked out",
                    "complaint": "expresses dissatisfaction, wants to complain",
                    "feedback": "shares an opinion, suggestion, or praise",
                },
            },
            "needs_troubleshooting": {
                "type": "noul",
                "instructions": "Does this need step-by-step troubleshooting?",
            },
            "is_escalation_worthy": {
                "type": "noul",
                "instructions": "Should this be escalated to a senior specialist?",
            },
        },
    },
    "account": {
        "choice_question": "account_action",
        "questions": {
            "account_action": {
                "type": "choice",
                "instructions": "What account action does this customer need?",
                "criteria": {
                    "update_info": "change name, email, address, profile details",
                    "delete_account": "delete, close, or deactivate the account",
                    "change_plan": "switch plan, upgrade, downgrade subscription tier",
                    "security": "security settings, two-factor, suspicious activity",
                    "verify_identity": "prove identity, verification code, ID check",
                },
            },
            "is_irreversible": {
                "type": "noul",
                "instructions": "Is this a destructive or irreversible action?",
            },
            "needs_identity_verification": {
                "type": "noul",
                "instructions": "Must the customer's identity be verified before proceeding?",
            },
        },
    },
    "sales": {
        "choice_question": "sales_intent",
        "questions": {
            "sales_intent": {
                "type": "choice",
                "instructions": "What is this customer's sales intent?",
                "criteria": {
                    "pricing": "asks about prices, costs, fees",
                    "compare_plans": "compares plans, tiers, or options",
                    "demo_request": "wants a demo or walkthrough",
                    "trial": "asks about a free trial",
                    "purchase": "ready to buy, asks how to purchase or sign up",
                },
            },
            "is_ready_to_buy": {
                "type": "noul",
                "instructions": "Does this customer look ready to buy now?",
            },
            "needs_human_sales": {
                "type": "noul",
                "instructions": "Should a human salesperson take over this conversation?",
            },
        },
    },
}


def agent_spec(agent: str) -> dict:
    """Return the analyzer spec for an agent id (KeyError if unknown)."""
    return ANALYZERS[agent]


def choice_question(agent: str) -> str:
    return ANALYZERS[agent]["choice_question"]


def choice_options(agent: str) -> list[str]:
    """Choice option names in stable order (label ids for training)."""
    qid = ANALYZERS[agent]["choice_question"]
    return list(ANALYZERS[agent]["questions"][qid]["criteria"].keys())


def noul_questions(agent: str) -> list[str]:
    """Noul question ids in stable order."""
    return [
        qid for qid, q in ANALYZERS[agent]["questions"].items()
        if q["type"] == "noul"
    ]


def question_ids(agent: str) -> list[str]:
    """All question ids for an agent in stable order (choice first)."""
    spec = ANALYZERS[agent]
    cq = spec["choice_question"]
    return [cq] + [qid for qid in spec["questions"] if qid != cq]


# ---------------------------------------------------------------------------
# Custom agents (defined through the Studio "Training Data" tab).
#
# A custom agent is one JSON file: agents/<agent>.json
# {
#   "agent": "refunds",
#   "choice_question": "refund_action",
#   "choice_instructions": "What refund action does this customer need?",
#   "labels": {"request_refund": "asks for money back", ...},
#   "noul": [
#     {"name": "amount_mentioned", "instructions": "...",
#      "keywords": ["$", "dollar"], "implied_by": ["request_refund"]}
#   ],
#   "seeds": {"request_refund": ["I want a refund", ...], ...}
# }
# Merged into ANALYZERS below so training, serving, and the dataset builder
# all see the same spec. A restart picks up new/edited files.
# ---------------------------------------------------------------------------

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"

#: Custom agent ids loaded from agents/*.json (subset of AGENT_IDS).
CUSTOM_AGENT_IDS: tuple[str, ...] = ()

#: agent -> {"seeds": {label: [query, ...]},
#:           "noul_rules": [{"name", "keywords", "implied_by"}]}
#: Used by train/build_dataset.py for UI-defined agents.
DATASET_SPECS: dict[str, dict] = {}

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


def validate_custom_spec(spec: dict) -> dict:
    """Validate a UI-submitted custom agent spec. Returns it unchanged.

    Raises ValueError with a human-readable message on any problem.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")
    agent = spec.get("agent", "")
    if not isinstance(agent, str) or not _ID_RE.match(agent):
        raise ValueError("agent must look like 'refunds' (lowercase, digits/_)")
    if agent in _BUILTIN_AGENT_IDS:
        raise ValueError(f"'{agent}' is a built-in agent and cannot be redefined")
    cq = spec.get("choice_question", "")
    if not isinstance(cq, str) or not _ID_RE.match(cq):
        raise ValueError("choice_question must look like 'refund_action'")
    ci = spec.get("choice_instructions", "")
    if not isinstance(ci, str) or not ci.strip():
        raise ValueError("choice_instructions is required")
    labels = spec.get("labels")
    if not isinstance(labels, dict) or not (2 <= len(labels) <= 20):
        raise ValueError("labels: provide 2-20 labels")
    for name, crit in labels.items():
        if not isinstance(name, str) or not _ID_RE.match(name):
            raise ValueError(f"bad label name: {name!r}")
        if not isinstance(crit, str) or not crit.strip():
            raise ValueError(f"label '{name}' needs a criteria description")
    noul = spec.get("noul", [])
    if not isinstance(noul, list) or len(noul) > 8:
        raise ValueError("noul: provide 0-8 flags")
    seen_n = set()
    for n in noul:
        if not isinstance(n, dict):
            raise ValueError("each noul entry must be an object")
        name = n.get("name", "")
        if not isinstance(name, str) or not _ID_RE.match(name):
            raise ValueError(f"bad noul name: {name!r}")
        if name in seen_n:
            raise ValueError(f"duplicate noul name: {name}")
        seen_n.add(name)
        if not isinstance(n.get("instructions", ""), str) or not n["instructions"].strip():
            raise ValueError(f"noul '{name}' needs instructions")
        kws = n.get("keywords", [])
        if not isinstance(kws, list) or any(not isinstance(k, str) or not k.strip() for k in kws):
            raise ValueError(f"noul '{name}': keywords must be non-empty strings")
        impl = n.get("implied_by", [])
        if not isinstance(impl, list) or any(i not in labels for i in impl):
            raise ValueError(f"noul '{name}': implied_by must reference labels")
    seeds = spec.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != set(labels):
        raise ValueError("seeds: provide seed queries for every label, no extras")
    for label, qs in seeds.items():
        if not isinstance(qs, list) or not (1 <= len(qs) <= 30):
            raise ValueError(f"seeds for '{label}': provide 1-30 queries")
        if any(not isinstance(q, str) or not q.strip() for q in qs):
            raise ValueError(f"seeds for '{label}' must be non-empty strings")
    return spec


def refresh_custom_agents() -> tuple[str, ...]:
    """Re-scan agents/*.json, picking up Studio-saved specs without restart.

    Returns the current custom agent ids. Serving/training still need a
    restart (they snapshot AGENT_IDS at import); the Studio endpoints call
    this so save -> build works in one session.
    """
    global CUSTOM_AGENT_IDS
    for agent in CUSTOM_AGENT_IDS:
        ANALYZERS.pop(agent, None)
        DATASET_SPECS.pop(agent, None)
    CUSTOM_AGENT_IDS = ()
    _load_custom_agents()
    return CUSTOM_AGENT_IDS


def _load_custom_agents() -> None:
    """Merge agents/*.json into ANALYZERS / AGENT_IDS / DATASET_SPECS."""
    global AGENT_IDS, CUSTOM_AGENT_IDS
    customs: list[str] = []
    if AGENTS_DIR.is_dir():
        for path in sorted(AGENTS_DIR.glob("*.json")):
            try:
                spec = validate_custom_spec(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, json.JSONDecodeError) as e:
                # Broken custom file: keep the last good state, like the
                # decision-tree hot-reload policy.
                print(f"[analyzer_questions] ignoring {path.name}: {e}")
                continue
            agent = spec["agent"]
            if agent in ANALYZERS:
                print(f"[analyzer_questions] ignoring {path.name}: duplicate agent '{agent}'")
                continue
            cq = spec["choice_question"]
            ANALYZERS[agent] = {
                "choice_question": cq,
                "questions": {
                    cq: {
                        "type": "choice",
                        "instructions": spec["choice_instructions"],
                        "criteria": dict(spec["labels"]),
                    },
                    **{
                        n["name"]: {"type": "noul", "instructions": n["instructions"]}
                        for n in spec["noul"]
                    },
                },
                "custom": True,
            }
            DATASET_SPECS[agent] = {
                "seeds": {k: list(v) for k, v in spec["seeds"].items()},
                "noul_rules": [
                    {"name": n["name"],
                     "keywords": list(n.get("keywords", [])),
                     "implied_by": list(n.get("implied_by", []))}
                    for n in spec["noul"]
                ],
            }
            customs.append(agent)
    CUSTOM_AGENT_IDS = tuple(customs)
    AGENT_IDS = _BUILTIN_AGENT_IDS + CUSTOM_AGENT_IDS


_load_custom_agents()
