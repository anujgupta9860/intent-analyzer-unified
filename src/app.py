"""Unified Intent Analyzer service: one giant model for all agents.

A single shared-encoder model (train/finetune_unified.py) serves every
agent's domain-specific question set. The request's worker_agent (agent
name) selects which per-agent heads answer — one checkpoint, one model
in memory, same typed-decision contract per agent.

Endpoints:
  GET  /health    - liveness + which agents the unified checkpoint covers
  POST /analyze   - {"query": ..., "worker_agent": "billing",
                     "router_context": {...}} -> that agent's decisions
  GET  /feedback/stats   - low-confidence log counters
  GET  /feedback/pending - records awaiting human review
  POST /feedback/review/{id} - approve | correct | reject
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .analyzer_questions import AGENT_IDS
from .config import Settings
from .feedback import AnalyzerFeedback
from .unified_analyzers import UnifiedAnalyzerClient, AnalyzerError

log = logging.getLogger(__name__)

_state: dict = {}


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
