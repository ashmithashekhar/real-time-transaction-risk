"""Tests for the synthetic data generator."""

from src.data_generator import generate_transactions, transactions_to_dataframe

REQUIRED_FIELDS = {
    "transaction_id", "user_id", "timestamp", "amount", "merchant_id",
    "merchant_category", "payment_method", "card_present", "device_id",
    "is_new_device", "country", "home_country", "distance_from_home_km",
    "velocity_1h_count", "velocity_1h_amount", "is_fraud",
}


def test_seeded_generation_is_deterministic():
    a = generate_transactions(n=300, seed=7)
    b = generate_transactions(n=300, seed=7)
    assert len(a) == len(b)
    assert a[:25] == b[:25]


def test_different_seed_differs():
    a = generate_transactions(n=300, seed=7)
    b = generate_transactions(n=300, seed=8)
    assert a != b


def test_schema_and_labels():
    txns = generate_transactions(n=500, seed=42)
    for txn in txns:
        assert REQUIRED_FIELDS <= set(txn.keys())
        assert txn["is_fraud"] in (0, 1)
        assert txn["amount"] > 0
    ids = [t["transaction_id"] for t in txns]
    assert len(ids) == len(set(ids))


def test_chronological_with_velocity():
    txns = generate_transactions(n=500, seed=42)
    stamps = [t["timestamp"] for t in txns]
    assert stamps == sorted(stamps)
    assert all(t["velocity_1h_count"] >= 1 for t in txns)


def test_fraud_rate_in_plausible_range():
    txns = generate_transactions(n=5000, seed=42, fraud_rate=0.015)
    rate = sum(t["is_fraud"] for t in txns) / len(txns)
    assert 0.005 <= rate <= 0.04


def test_dataframe_conversion():
    df = transactions_to_dataframe(generate_transactions(n=200, seed=3))
    # Row count is approximate by design: fraud bursts/testing patterns
    # emit a variable number of transactions per injected event.
    assert 200 <= len(df) <= 230
    assert set(REQUIRED_FIELDS) <= set(df.columns)
