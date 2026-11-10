"""HTTP service: send a ticket, get a team or a handover decision.

Run:  uv run uvicorn handover.api:app --port 8000
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from handover.embed import MODEL_NAME, embed
from handover.model import ARTIFACT_PATH, Router

MAX_CHARS = 5000
PRECEDENT_CHARS = 600
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at start-up so the first request is not slow.
    state["router"] = Router.load(Path(os.environ.get("HANDOVER_ARTIFACT", ARTIFACT_PATH)))
    embed(["warm-up"])
    yield
    state.clear()


app = FastAPI(
    title="Handover",
    description="Routes customer tickets to a team, or hands uncertain ones to a person.",
    version="1.0.0",
    lifespan=lifespan,
)


class RouteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS, description="Subject and body of the ticket.")
    target_risk: float | None = Field(
        default=None,
        description="Accepted error rate among automatically routed tickets. "
        "Must be one of the values listed by /info. Defaults to the model's setting.",
    )


class Vote(BaseModel):
    team: str
    neighbours: int  # how many of the most similar past tickets went to this team


class Precedent(BaseModel):
    text: str
    team: str
    similarity: float


class RouteResponse(BaseModel):
    decision: str  # "route" or "handover"
    reason: str
    team: str | None  # set only when decision is "route"
    confidence: float
    threshold: float | None  # None when no threshold can keep the promise
    target_risk: float
    votes: list[Vote]
    precedent: Precedent  # the most similar past ticket: the reason for the decision
    latency_ms: float


def _threshold_for(router: Router, target_risk: float | None) -> tuple[float, float]:
    default = router.metadata.get("default_target_risk")
    if target_risk is None:
        return router.threshold, default
    points = router.metadata.get("thresholds", {})
    key = f"{target_risk:.2f}"
    if key not in points:
        raise HTTPException(422, f"target_risk must be one of {sorted(float(k) for k in points)}")
    return points[key], target_risk


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "router" in state}


@app.get("/info")
def info() -> dict:
    """What the model is, how it scored on test data, and the available settings."""
    router: Router = state["router"]
    return {
        "teams": router.classes,
        "embedding_model": MODEL_NAME,
        "past_tickets_in_memory": len(router.index_texts),
        "default_threshold": router.threshold if router.threshold != float("inf") else None,
        **{k: v for k, v in router.metadata.items() if k != "thresholds"},
    }


@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(422, "text is empty")
    router: Router = state["router"]
    threshold, target = _threshold_for(router, request.target_risk)

    start = time.perf_counter()
    result = router.decide(embed([text]), threshold=threshold)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    return RouteResponse(
        decision="handover" if result["handover"] else "route",
        reason=result["reason"],
        team=None if result["handover"] else result["team"],
        confidence=result["confidence"],
        threshold=threshold if threshold != float("inf") else None,
        target_risk=target,
        votes=[Vote(**v) for v in result["votes"]],
        precedent=Precedent(
            text=result["precedent"]["text"][:PRECEDENT_CHARS],
            team=result["precedent"]["team"],
            similarity=result["precedent"]["similarity"],
        ),
        latency_ms=round(latency_ms, 1),
    )
