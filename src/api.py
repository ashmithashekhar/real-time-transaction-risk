"""FastAPI scoring service.

Exposes the hybrid scorer as a small HTTP service:

* ``GET  /health``  -> service status, incl. whether the ML model is loaded
* ``POST /score``   -> score one transaction, returns risk + decision

If the request omits the velocity fields, the service maintains them
itself with an in-memory :class:`VelocityTracker`, keyed by user —
mirroring what the streaming consumer sees. (Stateless multi-replica
deployments would move that state to Redis/a feature store; the
scoring code would not change.)

Run::

    uvicorn src.api:app --reload --port 8000
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/score -H 'Content-Type: application/json' \
      -d '{"user_id":"U-1001","timestamp":"2025-01-06T03:15:00Z","amount":2450.00,
           "merchant_category":"gift_cards","payment_method":"online",
           "card_present":false,"is_new_device":true,"country":"GB",
           "home_country":"US","distance_from_home_km":5600}'

Environment:
    RISK_MODEL_PATH  optional path to a trained model bundle
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .features import VelocityTracker
from .scoring import RiskScorer

app = FastAPI(
    title="Real-Time Transaction Risk Platform",
    description="Hybrid rule + ML scoring for payment transactions. "
                "Trained and demoed exclusively on synthetic data.",
    version="1.0.0",
)

_scorer = RiskScorer(model_path=os.getenv("RISK_MODEL_PATH") or None)
_velocity = VelocityTracker()


class TransactionIn(BaseModel):
    transaction_id: Optional[str] = None
    user_id: str
    timestamp: datetime
    amount: float = Field(gt=0)
    merchant_id: Optional[str] = None
    merchant_category: str = "retail"
    payment_method: str = "chip"
    card_present: bool = True
    device_id: Optional[str] = None
    is_new_device: bool = False
    country: str = "US"
    home_country: str = "US"
    distance_from_home_km: float = 0.0
    velocity_1h_count: Optional[int] = None
    velocity_1h_amount: Optional[float] = None


class SignalOut(BaseModel):
    name: str
    weight: float
    detail: str


class ScoreOut(BaseModel):
    transaction_id: Optional[str]
    rule_score: int
    model_score: Optional[float]
    risk_score: float
    decision: str
    signals: List[SignalOut]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    scoring_mode: str
    model_metadata: Dict[str, Any]


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        status="ok",
        model_loaded=_scorer.model_loaded,
        scoring_mode="hybrid (rules + model)" if _scorer.model_loaded
                     else "rules only",
        model_metadata=_scorer.model_metadata,
    )


@app.post("/score", response_model=ScoreOut)
def score(txn: TransactionIn) -> ScoreOut:
    data = txn.model_dump()
    data["timestamp"] = txn.timestamp.isoformat()

    # Fill velocity from service-side state when the caller didn't
    # supply it (e.g. a thin producer that only sends raw events).
    if data["velocity_1h_count"] is None or data["velocity_1h_amount"] is None:
        count, total = _velocity.update(
            data["user_id"], data["timestamp"], data["amount"]
        )
        data["velocity_1h_count"] = count
        data["velocity_1h_amount"] = round(total, 2)

    result = _scorer.score(data)
    return ScoreOut(**result)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "real-time-transaction-risk",
        "docs": "/docs",
        "health": "/health",
        "score": "POST /score",
    }
