"""Unified Intent Analyzer service: one giant model for all agents.

A single shared-encoder model (train/finetune_unified.py) serves every
agent's domain-specific question set. The request's worker_agent (agent
name) selects which per-agent heads answer — one checkpoint, one model
in memory, same typed-decision contract per agent.

Endpoints:
  GET  /health    - liveness + which agents the unified checkpoint covers
  POST /analyze   - {"query": ..., "worker_agent": "billing",
                     "router_context": {...}} -> that agent's decisions
  POST /decide    - {"query": ..., "worker_agent": "billing"} ->
                     analyzer output + policy decision (the A2A agent
                     integration point: returns the action to dispatch)
  GET  /studio    - Decision Tree Studio UI (layman rule editor)
  GET  /policy/spec - dropdown values for the Studio (agents, choices, nouls)
  GET  /policy/tree - current decision tree (JSON)
  PUT  /policy/tree - save a new decision tree (validated before saving)
  POST /policy/test - dry-run the tree on a query or a hand-made
                     analyzer output
  GET  /feedback/stats   - low-confidence log counters
  GET  /feedback/pending - records awaiting human review
  POST /feedback/review/{id} - approve | correct | reject
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .analyzer_questions import (
    AGENT_IDS,
    choice_options,
    noul_questions,
)
from .config import Settings
from .feedback import AnalyzerFeedback
from .unified_analyzers import UnifiedAnalyzerClient, AnalyzerError

log = logging.getLogger(__name__)

_state: dict = {}

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "policy" / "decision-tree.yaml"
STUDIO_INDEX = REPO_ROOT / "studio" / "index.html"

# Known actions offered in the Studio UI (free text also allowed).
POLICY_ACTIONS = [
    "process_refund", "open_dispute_case", "lookup_order_status",
    "ask_clarify", "escalate_human", "noop",
]


def _policy_engine():
    from policy.engine import PolicyEngine, PolicyError
    eng = _state.get("policy")
    if eng is None:
        try:
            eng = PolicyEngine(POLICY_PATH)
        except PolicyError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _state["policy"] = eng
    return eng


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    client = UnifiedAnalyzerClient(
        f"{settings.checkpoint_dir}/unified",
        confidence_threshold=settings.confidence_threshold)
    feedback = AnalyzerFeedback(enabled=settings.feedback_enabled)
    _state.update(settings=settings, client=client, feedback=feedback)
    log.info("unified intent-analyzer ready: checkpoint_dir=%s available=%s",
             settings.checkpoint_dir, client.available())
    try:
        eng = _policy_engine()
        log.info("policy engine ready: %d rules from %s",
                 len(eng.rules), POLICY_PATH)
    except HTTPException as exc:
        log.warning("policy engine unavailable: %s", exc.detail)
    yield
    _state.clear()


app = FastAPI(title="Unified Intent Analyzer (one model, all agents)",
              version="0.1.0", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    worker_agent: str = Field(
        description="agent name: one of billing|orders|support|account|sales. "
                    "Selects which per-agent question set the unified model "
                    "answers.")
    router_context: dict = Field(
        default_factory=dict,
        description="optional router output echoed for the feedback log "
                    "(intent, skill_required, confidence, ...)")


def _client() -> UnifiedAnalyzerClient:
    client = _state.get("client")
    if client is None:
        raise HTTPException(status_code=503, detail="analyzer not initialized")
    return client


def _feedback() -> AnalyzerFeedback:
    return _state.get("feedback")


@app.get("/health")
def health():
    client = _state.get("client")
    settings = _state.get("settings")
    return {
        "status": "ok",
        "backend": "unified",
        "agents": list(AGENT_IDS),
        "available": client.available() if client else {},
        "checkpoint_dir": settings.checkpoint_dir if settings else None,
        "confidence_threshold": settings.confidence_threshold if settings else None,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Run the unified analyzer on the query for the named agent.

    The agent name picks the question set: each agent gets its own
    choice taxonomy + noul flags, answered by one shared model.
    When choice confidence is below threshold, needs_human=true and
    the decision is logged for human review (RLCD feedback loop).
    """
    agent = req.worker_agent.strip().lower()
    if agent not in AGENT_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown worker_agent {agent!r}; expected one of "
                   f"{list(AGENT_IDS)}")
    client = _client()
    if not client.available().get(agent):
        raise HTTPException(
            status_code=503,
            detail=f"unified checkpoint does not cover agent {agent!r} "
                   f"(train: python train/finetune_unified.py "
                   f"--data train/data --out models/unified)")
    try:
        decision = client.analyze(req.query.strip(), agent)
    except AnalyzerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    out = decision.to_dict()
    if decision.needs_human:
        fb = _feedback()
        if fb is not None:
            fb.log(agent, req.query.strip(), out,
                   router_context=req.router_context)
    return out


@app.get("/feedback/stats")
def feedback_stats():
    return _feedback().stats()


@app.get("/feedback/pending")
def feedback_pending(agent: str | None = None, limit: int = 50):
    return {"pending": _feedback().list_pending(agent=agent, limit=limit)}


class ReviewDecision(BaseModel):
    decision: str = Field(pattern="^(approved|corrected|rejected)$")
    reviewer: str = "human"
    corrections: dict | None = None


@app.post("/feedback/review/{record_id}")
def feedback_review(record_id: str, req: ReviewDecision):
    ok = _feedback().review(record_id, req.decision,
                            reviewer=req.reviewer,
                            corrections=req.corrections)
    if not ok:
        raise HTTPException(status_code=404, detail="record not found")
    return {"record_id": record_id, "decision": req.decision}


# ------------------------------------------------------- decision framework
# Layman-configurable decision tree over analyzer output, integrated with
# the A2A agent layer. The Studio UI (/studio) edits policy/decision-tree.yaml
# through these endpoints; POST /decide is the agent integration point.


@app.get("/studio")
def studio():
    if not STUDIO_INDEX.exists():
        raise HTTPException(status_code=404, detail="studio UI not installed")
    return FileResponse(STUDIO_INDEX, media_type="text/html")


@app.get("/policy/spec")
def policy_spec():
    """Dropdown values for the Studio: only valid options are offered."""
    from policy.engine import _OPERATORS, _RISKS
    return {
        "agents": list(AGENT_IDS),
        "choices": {a: choice_options(a) for a in AGENT_IDS},
        "nouls": {a: noul_questions(a) for a in AGENT_IDS},
        "operators": list(_OPERATORS),
        "risks": list(_RISKS),
        "actions": POLICY_ACTIONS,
    }


@app.get("/policy/tree")
def policy_tree():
    import yaml
    eng = _policy_engine()
    return {"version": eng.version,
            "rules": yaml.safe_load(POLICY_PATH.read_text())["rules"]}


class PolicyTreePut(BaseModel):
    rules: list = Field(description="full rule list replacing the tree")


@app.put("/policy/tree")
def policy_tree_put(req: PolicyTreePut):
    """Validate then save a new decision tree (from the Studio UI)."""
    import yaml
    from policy.engine import PolicyError
    eng = _policy_engine()
    try:
        eng.validate_tree({"rules": req.rules})
    except PolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    POLICY_PATH.write_text(yaml.safe_dump(
        {"version": eng.version, "rules": req.rules},
        sort_keys=False, allow_unicode=True))
    eng.reload()
    return {"saved": True, "rules": len(eng.rules)}


class PolicyTestRequest(BaseModel):
    query: str | None = Field(
        default=None, description="run the analyzer on this query "
        "(needs a trained checkpoint)")
    worker_agent: str | None = Field(
        default=None, description="agent for the analyzer run")
    analyzer_output: dict | None = Field(
        default=None, description="hand-made analyzer output; "
        "tests the rules without the model")


@app.post("/policy/test")
def policy_test(req: PolicyTestRequest):
    """Dry-run the decision tree: query -> analyzer -> policy -> decision."""
    eng = _policy_engine()
    if req.analyzer_output is not None:
        output = req.analyzer_output
    elif req.query and req.worker_agent:
        agent = req.worker_agent.strip().lower()
        if agent not in AGENT_IDS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown worker_agent {agent!r}")
        client = _client()
        if not client.available().get(agent):
            raise HTTPException(
                status_code=503,
                detail=f"no trained checkpoint for agent {agent!r}")
        try:
            output = client.analyze(req.query.strip(), agent).to_dict()
        except AnalyzerError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="provide analyzer_output, or query + worker_agent")
    decision = eng.decide(output)
    return {"analyzer_output": output, "decision": decision.to_dict()}


class DecideRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    worker_agent: str = Field(
        description="agent name: one of billing|orders|support|account|sales")
    router_context: dict = Field(default_factory=dict)


@app.post("/decide")
def decide(req: DecideRequest):
    """Agent integration point: query -> analyzer -> policy -> action.

    Returns the analyzer's typed decisions AND the policy decision
    (action, worker, risk). The A2A agent layer dispatches on
    decision.action; high-risk decisions should go to a human first.
    """
    agent = req.worker_agent.strip().lower()
    if agent not in AGENT_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown worker_agent {agent!r}; expected one of "
                   f"{list(AGENT_IDS)}")
    client = _client()
    if not client.available().get(agent):
        raise HTTPException(
            status_code=503,
            detail=f"no trained checkpoint for agent {agent!r}")
    try:
        analyzer_out = client.analyze(req.query.strip(), agent).to_dict()
    except AnalyzerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    eng = _policy_engine()
    decision = eng.decide(analyzer_out)
    return {
        "agent": agent,
        "query": req.query.strip(),
        "analyzer": analyzer_out,
        "decision": decision.to_dict(),
    }
