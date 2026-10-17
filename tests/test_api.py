"""Smoke tests for the FastAPI scoring service."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["scoring_mode"] in ("rules only", "hybrid (rules + model)")


def test_score_benign():
    resp = client.post("/score", json={
        "user_id": "U-API-1",
        "timestamp": "2025-01-06T12:00:00Z",
        "amount": 25.0,
        "merchant_category": "groceries",
        "payment_method": "chip",
        "card_present": True,
        "country": "US",
        "home_country": "US",
    })
    assert resp.status_code == 200
    assert resp.json()["decision"] == "approve"


def test_score_fraud_like():
    resp = client.post("/score", json={
        "user_id": "U-API-2",
        "timestamp": "2025-01-06T02:30:00Z",
        "amount": 3200.0,
        "merchant_category": "gift_cards",
        "payment_method": "online",
        "card_present": False,
        "is_new_device": True,
        "country": "GB",
        "home_country": "US",
        "distance_from_home_km": 5600.0,
        "velocity_1h_count": 6,
        "velocity_1h_amount": 9400.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "block"
    assert len(body["signals"]) >= 5


def test_score_rejects_nonpositive_amount():
    resp = client.post("/score", json={
        "user_id": "U-API-3",
        "timestamp": "2025-01-06T12:00:00Z",
        "amount": -5.0,
    })
    assert resp.status_code == 422
