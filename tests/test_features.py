"""Tests for feature engineering, rule signals, and velocity tracking."""

from datetime import datetime, timedelta, timezone

from src.features import (FEATURE_COLUMNS, VelocityTracker, compute_features,
                          feature_vector, risk_signals, rule_score)

BASE = datetime(2025, 1, 6, 12, 0, 0, tzinfo=timezone.utc)


def benign_txn(**overrides):
    txn = {
        "transaction_id": "TXN-T1",
        "user_id": "U-TEST",
        "timestamp": BASE.isoformat(),
        "amount": 42.50,
        "merchant_id": "GROC-001",
        "merchant_category": "groceries",
        "payment_method": "chip",
        "card_present": True,
        "device_id": "DEV-1",
        "is_new_device": False,
        "country": "US",
        "home_country": "US",
        "distance_from_home_km": 3.0,
        "velocity_1h_count": 1,
        "velocity_1h_amount": 42.50,
    }
    txn.update(overrides)
    return txn


def fraud_like_txn(**overrides):
    txn = benign_txn(
        timestamp=(BASE.replace(hour=2)).isoformat(),
        amount=3200.00,
        merchant_category="gift_cards",
        payment_method="online",
        card_present=False,
        is_new_device=True,
        country="GB",
        distance_from_home_km=5600.0,
        velocity_1h_count=6,
        velocity_1h_amount=9400.00,
    )
    txn.update(overrides)
    return txn


def test_feature_vector_shape_and_order():
    feats = compute_features(benign_txn())
    assert list(feats.keys()) == FEATURE_COLUMNS
    vec = feature_vector(benign_txn())
    assert len(vec) == len(FEATURE_COLUMNS)
    assert all(isinstance(v, float) for v in vec)


def test_features_handle_minimal_input():
    feats = compute_features({"amount": 10.0})
    assert feats["amount"] == 10.0
    assert feats["is_international"] == 0.0


def test_benign_transaction_fires_no_signals():
    score, signals = rule_score(benign_txn())
    assert signals == []
    assert score == 0


def test_fraud_like_transaction_fires_many_signals():
    names = {s["name"] for s in risk_signals(fraud_like_txn())}
    expected = {
        "very_high_amount", "international", "extreme_velocity",
        "night_transaction", "new_device", "high_risk_category",
        "far_from_home", "large_card_not_present",
    }
    assert expected <= names
    score, _ = rule_score(fraud_like_txn())
    assert score == 100  # weights far exceed the cap


def test_international_flag_in_features():
    feats = compute_features(benign_txn(country="CA"))
    assert feats["is_international"] == 1.0


def test_velocity_tracker_window():
    tracker = VelocityTracker(window_seconds=3600)
    assert tracker.update("U1", BASE, 10.0) == (1, 10.0)
    assert tracker.update("U1", BASE + timedelta(minutes=30), 20.0) == (2, 30.0)
    # 61 minutes after the first event: it has aged out of the window.
    count, total = tracker.update("U1", BASE + timedelta(minutes=61), 5.0)
    assert count == 2
    assert total == 25.0
    # Users are tracked independently.
    assert tracker.update("U2", BASE, 99.0) == (1, 99.0)


def test_velocity_tracker_peek_does_not_record():
    tracker = VelocityTracker()
    tracker.update("U1", BASE, 10.0)
    assert tracker.peek("U1", BASE) == (1, 10.0)
    assert tracker.peek("U1", BASE) == (1, 10.0)  # still 1: peek is read-only
