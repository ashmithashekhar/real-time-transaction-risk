"""Synthetic transaction data generator.

Generates realistic-looking payment transactions with a labelled
fraud minority class, entirely from random distributions — **no real
customer, bank, or employer data is used or needed**.

Legitimate transactions follow per-user spending profiles (home
country, typical amount, preferred merchant categories, known
devices). Fraud is injected through three recognisable patterns that
mirror textbook card-fraud typologies:

1. **Account takeover** — new device, foreign country, night-time,
   large amount in an easily-liquidated category.
2. **Velocity burst** — many transactions from one user within a few
   minutes.
3. **Card testing** — a run of tiny authorisations followed by one
   large charge.

Output rows follow the canonical schema in :mod:`src.features`,
including trailing-hour velocity fields computed exactly as the
streaming layer computes them (via :class:`VelocityTracker`), so
offline training data and online features agree.

Usage::

    python -m src.data_generator --n 20000 --seed 42 \
        --out data/transactions.csv
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .features import VelocityTracker

COUNTRIES = ["US", "CA", "GB", "IN", "MX", "DE", "AU"]
COUNTRY_WEIGHTS = [0.70, 0.08, 0.07, 0.08, 0.04, 0.02, 0.01]

LEGIT_CATEGORIES = [
    "groceries", "dining", "fuel", "retail", "pharmacy", "utilities",
    "entertainment", "online_retail", "travel", "electronics",
]
LEGIT_CATEGORY_WEIGHTS = [
    0.22, 0.16, 0.10, 0.14, 0.07, 0.06, 0.08, 0.09, 0.04, 0.04,
]

IN_STORE_CATEGORIES = {
    "groceries", "dining", "fuel", "retail", "pharmacy", "electronics",
    "entertainment", "utilities",
}
TAKEOVER_CATEGORIES = ["electronics", "gift_cards", "crypto", "money_transfer"]
BURST_CATEGORIES = ["online_retail", "electronics", "gift_cards"]

DEFAULT_START = datetime(2025, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
DEFAULT_DAYS = 14


@dataclass
class UserProfile:
    user_id: str
    home_country: str
    log_amount_mean: float
    log_amount_sigma: float
    devices: List[str] = field(default_factory=list)


def _make_users(rng: random.Random, n_users: int) -> List[UserProfile]:
    users = []
    for i in range(n_users):
        user_id = f"U-{1001 + i}"
        typical = rng.uniform(15, 120)  # typical spend in USD
        users.append(
            UserProfile(
                user_id=user_id,
                home_country=rng.choices(COUNTRIES, weights=COUNTRY_WEIGHTS)[0],
                log_amount_mean=math.log(typical),
                log_amount_sigma=rng.uniform(0.5, 0.85),
                devices=[f"DEV-{user_id}-{j + 1}" for j in range(rng.randint(1, 2))],
            )
        )
    return users


def _merchant_id(rng: random.Random, category: str) -> str:
    return f"{category[:4].upper()}-{rng.randint(1, 40):03d}"


def _base_txn(rng: random.Random, user: UserProfile, ts: datetime) -> Dict[str, Any]:
    return {
        "user_id": user.user_id,
        "timestamp": ts.isoformat(),
        "home_country": user.home_country,
        "device_id": rng.choice(user.devices),
        "is_new_device": False,
    }


def _legit_txn(rng: random.Random, user: UserProfile, start: datetime,
               span_seconds: float) -> Dict[str, Any]:
    ts = start + timedelta(seconds=rng.random() * span_seconds)
    txn = _base_txn(rng, user, ts)
    category = rng.choices(LEGIT_CATEGORIES, weights=LEGIT_CATEGORY_WEIGHTS)[0]
    amount = rng.lognormvariate(user.log_amount_mean, user.log_amount_sigma)
    amount = round(min(max(amount, 1.0), 2000.0), 2)

    if rng.random() < 0.97:
        country = user.home_country
        distance = round(min(rng.expovariate(1 / 25.0), 300.0), 1)
    else:
        country = rng.choice([c for c in COUNTRIES if c != user.home_country])
        distance = round(rng.uniform(800, 9000), 1)

    card_present = (category in IN_STORE_CATEGORIES) and rng.random() < 0.95
    txn.update({
        "amount": amount,
        "merchant_id": _merchant_id(rng, category),
        "merchant_category": category,
        "payment_method": (
            rng.choice(["chip", "contactless", "wallet", "swipe"])
            if card_present else "online"
        ),
        "card_present": card_present,
        "country": country,
        "distance_from_home_km": distance,
        "is_new_device": rng.random() < 0.02,
        "is_fraud": 0,
    })
    return txn


def _takeover_txn(rng: random.Random, user: UserProfile, start: datetime,
                  span_seconds: float) -> Dict[str, Any]:
    ts = start + timedelta(seconds=rng.random() * span_seconds)
    ts = ts.replace(hour=rng.randint(0, 4))  # night-time
    txn = _base_txn(rng, user, ts)
    category = rng.choice(TAKEOVER_CATEGORIES)
    country = rng.choice([c for c in COUNTRIES if c != user.home_country])
    txn.update({
        "amount": round(min(rng.lognormvariate(math.log(900), 0.7), 5000.0), 2),
        "merchant_id": _merchant_id(rng, category),
        "merchant_category": category,
        "payment_method": "online",
        "card_present": False,
        "device_id": f"DEV-UNKNOWN-{rng.randint(9000, 9999)}",
        "country": country,
        "distance_from_home_km": round(rng.uniform(1500, 9500), 1),
        "is_new_device": True,
        "is_fraud": 1,
    })
    return txn


def _burst_txns(rng: random.Random, user: UserProfile, start: datetime,
                span_seconds: float) -> List[Dict[str, Any]]:
    base = start + timedelta(seconds=rng.random() * span_seconds)
    txns = []
    for k in range(rng.randint(4, 7)):
        ts = base + timedelta(minutes=k * rng.uniform(1, 6))
        txn = _base_txn(rng, user, ts)
        category = rng.choice(BURST_CATEGORIES)
        txn.update({
            "amount": round(rng.uniform(200, 1200), 2),
            "merchant_id": _merchant_id(rng, category),
            "merchant_category": category,
            "payment_method": "online",
            "card_present": False,
            "country": user.home_country,
            "distance_from_home_km": round(rng.uniform(5, 80), 1),
            "is_new_device": rng.random() < 0.3,
            "is_fraud": 1,
        })
        txns.append(txn)
    return txns


def _card_testing_txns(rng: random.Random, user: UserProfile, start: datetime,
                       span_seconds: float) -> List[Dict[str, Any]]:
    base = start + timedelta(seconds=rng.random() * span_seconds)
    txns = []
    probes = rng.randint(3, 5)
    for k in range(probes + 1):
        ts = base + timedelta(minutes=k * rng.uniform(1, 4))
        txn = _base_txn(rng, user, ts)
        is_probe = k < probes
        txn.update({
            "amount": round(rng.uniform(1, 15), 2) if is_probe
                      else round(rng.uniform(800, 2500), 2),
            "merchant_id": _merchant_id(rng, "online_retail"),
            "merchant_category": "online_retail",
            "payment_method": "online",
            "card_present": False,
            "country": user.home_country,
            "distance_from_home_km": round(rng.uniform(5, 80), 1),
            "is_new_device": False,
            "is_fraud": 1,
        })
        txns.append(txn)
    return txns


def generate_transactions(n: int = 10000, seed: int = 42,
                          fraud_rate: float = 0.015,
                          start: datetime = DEFAULT_START,
                          days: int = DEFAULT_DAYS) -> List[Dict[str, Any]]:
    """Generate ``n`` synthetic transactions (approximately).

    The fraud share is approximate because bursts/testing patterns emit
    a variable number of transactions per injected event. Results are
    fully deterministic for a given ``seed``.
    """
    rng = random.Random(seed)
    span_seconds = days * 24 * 3600
    n_users = max(20, n // 200)
    users = _make_users(rng, n_users)

    n_fraud_target = int(n * fraud_rate)
    n_legit = n - n_fraud_target

    txns: List[Dict[str, Any]] = [
        _legit_txn(rng, rng.choice(users), start, span_seconds)
        for _ in range(n_legit)
    ]

    fraud_so_far = 0
    while fraud_so_far < n_fraud_target:
        pattern = rng.random()
        user = rng.choice(users)
        if pattern < 0.40:
            batch = [_takeover_txn(rng, user, start, span_seconds)]
        elif pattern < 0.75:
            batch = _burst_txns(rng, user, start, span_seconds)
        else:
            batch = _card_testing_txns(rng, user, start, span_seconds)
        txns.extend(batch)
        fraud_so_far += len(batch)

    # Chronological order, then sequential velocity — exactly how the
    # streaming layer sees the data.
    txns.sort(key=lambda t: t["timestamp"])
    tracker = VelocityTracker()
    for idx, txn in enumerate(txns, start=1):
        count, total = tracker.update(txn["user_id"], txn["timestamp"], txn["amount"])
        txn["transaction_id"] = f"TXN-{idx:06d}"
        txn["velocity_1h_count"] = count
        txn["velocity_1h_amount"] = round(total, 2)
        txn["currency"] = "USD"
    return txns


def transactions_to_dataframe(txns: List[Dict[str, Any]]):
    """Convert generated transactions to a pandas DataFrame."""
    import pandas as pd

    columns = [
        "transaction_id", "user_id", "timestamp", "amount", "currency",
        "merchant_id", "merchant_category", "payment_method", "card_present",
        "device_id", "is_new_device", "country", "home_country",
        "distance_from_home_km", "velocity_1h_count", "velocity_1h_amount",
        "is_fraud",
    ]
    return pd.DataFrame(txns, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transactions")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraud-rate", type=float, default=0.015)
    parser.add_argument("--out", default="data/transactions.csv")
    args = parser.parse_args()

    txns = generate_transactions(n=args.n, seed=args.seed,
                                 fraud_rate=args.fraud_rate)
    df = transactions_to_dataframe(txns)
    df.to_csv(args.out, index=False)
    fraud = int(df["is_fraud"].sum())
    print(f"Wrote {len(df):,} transactions to {args.out}")
    print(f"Fraud: {fraud:,} ({fraud / len(df):.2%}) | "
          f"Users: {df['user_id'].nunique()} | "
          f"Period: {df['timestamp'].min()} .. {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
