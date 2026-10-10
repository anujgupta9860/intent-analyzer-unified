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
DEFAULT_POLICY_PATH = REPO_ROOT / "policy" / "decision-tree.yaml"
DEFAULT_WORKFLOWS_PATH = REPO_ROOT / "workflows" / "workflows.yaml"
STUDIO_INDEX = REPO_ROOT / "studio" / "index.html"

# Known actions offered in the Studio UI (free text also allowed).
POLICY_ACTIONS = [
    "process_refund", "open_dispute_case", "lookup_order_status",
    "ask_clarify", "escalate_human", "noop",
]


def _policy_path() -> Path:
    settings = _state.get("settings")
    override = (settings.policy_path if settings else "") or ""
    return Path(override) if override else DEFAULT_POLICY_PATH


def _policy_engine():
    from policy.engine import PolicyEngine, PolicyError
    eng = _state.get("policy")
    if eng is None:
        try:
            eng = PolicyEngine(_policy_path())
        except PolicyError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _state["policy"] = eng
    return eng


def _workflows_loader():
    from workflows.loader import WorkflowLoader, WorkflowError
    loader = _state.get("workflows")
    if loader is None:
        try:
            loader = WorkflowLoader(DEFAULT_WORKFLOWS_PATH)
        except WorkflowError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _state["workflows"] = loader
    return loader


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
        log.info("policy engine ready: %d rules from %s (hot-reload on)",
                 len(eng.rules), _policy_path())
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
            "rules": yaml.safe_load(_policy_path().read_text())["rules"]}


@app.get("/policy/status")
def policy_status():
    """Hot-reload status: is the live tree healthy, when did it reload,
    and what was the last error (if a bad edit was rejected)."""
    return _policy_engine().status()


@app.post("/policy/reload")
def policy_reload():
    """Force a reload check now (normally automatic on every decision)."""
    eng = _policy_engine()
    reloaded = eng.maybe_reload()
    st = eng.status()
    st["reloaded_now"] = reloaded
    return st


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
    _policy_path().write_text(yaml.safe_dump(
        {"version": eng.version, "rules": req.rules},
        sort_keys=False, allow_unicode=True))
    eng.reload()
    return {"saved": True, "rules": len(eng.rules)}


# ------------------------------------------------------- workflow definitions
# Hot-reloaded workflow library. The Studio edits these; the router
# fetches them from GET /workflows and runs the state machine.


@app.get("/workflows")
def workflows_list():
    """All workflow definitions (hot-reloaded)."""
    import yaml
    loader = _workflows_loader()
    loader.maybe_reload()
    return {"workflows": yaml.safe_load(
        DEFAULT_WORKFLOWS_PATH.read_text())["workflows"]}


@app.get("/workflows/status")
def workflows_status():
    """Hot-reload status for workflows.yaml."""
    return _workflows_loader().status()


class WorkflowsPut(BaseModel):
    workflows: list = Field(description="full workflow list replacing the file")


@app.put("/workflows")
def workflows_put(req: WorkflowsPut):
    """Validate then save new workflow definitions (from the Studio UI)."""
    import yaml
    from workflows.loader import WorkflowError
    loader = _workflows_loader()
    # validate by loading into a temp loader before touching the file
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                     delete=False) as tmp:
        yaml.safe_dump({"workflows": req.workflows}, tmp,
                       sort_keys=False, allow_unicode=True)
        tmp_path = tmp.name
    try:
        from workflows.loader import WorkflowLoader
        WorkflowLoader(tmp_path)  # raises on invalid
    except WorkflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    DEFAULT_WORKFLOWS_PATH.write_text(yaml.safe_dump(
        {"workflows": req.workflows}, sort_keys=False, allow_unicode=True))
    loader.reload()
    return {"saved": True, "workflows": len(loader.workflows())}


@app.post("/workflows/identify")
def workflows_identify(req: DecideRequest):
    """System 1's workflow decision: which workflow does this query start?

    Returns the matched workflow (or null for a single task).
    """
    loader = _workflows_loader()
    wf = loader.match(req.query, req.worker_agent.strip().lower())
    return {"workflow": wf["name"] if wf else None,
            "definition": wf}


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
    eng.maybe_reload()  # pick up on-disk edits without redeploy
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
    workflow: str | None = Field(
        default=None,
        description="active workflow name — the policy tree and analyzer "
                    "run at workflow level with this set")


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
    eng.maybe_reload()  # pick up on-disk edits without redeploy
    decision = eng.decide(analyzer_out)
    out: dict = {
        "agent": agent,
        "query": req.query.strip(),
        "analyzer": analyzer_out,
        "decision": decision.to_dict(),
    }
    if req.workflow:
        # Workflow level: attach the workflow definition so the router's
        # decision mechanism runs with workflow context.
        wf = _workflows_loader().get(req.workflow)
        out["workflow"] = wf
    return out
