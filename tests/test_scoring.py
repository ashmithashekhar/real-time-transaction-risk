"""Tests for the hybrid scorer (rules-only mode: no model file needed)."""

from datetime import datetime, timezone

from src.scoring import RiskScorer, decide
from tests.test_features import benign_txn, fraud_like_txn

BASE = datetime(2025, 1, 6, 12, 0, 0, tzinfo=timezone.utc)


def test_decision_bands():
    assert decide(0) == "approve"
    assert decide(39.9) == "approve"
    assert decide(40) == "review"
    assert decide(69.9) == "review"
    assert decide(70) == "block"
    assert decide(100) == "block"


def test_rules_only_scorer_without_model(tmp_path):
    scorer = RiskScorer(model_path=tmp_path / "does-not-exist.joblib")
    assert scorer.model_loaded is False

    ok = scorer.score(benign_txn())
    assert ok["decision"] == "approve"
    assert ok["risk_score"] == 0.0
    assert ok["model_score"] is None

    bad = scorer.score(fraud_like_txn())
    assert bad["decision"] == "block"
    assert bad["risk_score"] == 100.0
    assert len(bad["signals"]) >= 5


def test_score_is_monotonic_in_risk():
    scorer = RiskScorer(model_path="models/__missing__.joblib")
    medium = benign_txn(amount=1500.0, merchant_category="electronics",
                        payment_method="online", card_present=False)
    scores = [
        scorer.score(benign_txn())["risk_score"],
        scorer.score(medium)["risk_score"],
        scorer.score(fraud_like_txn())["risk_score"],
    ]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]
