"""Shared feature engineering for the transaction risk platform.

This module is the single source of truth for how a raw transaction
dictionary is turned into model features and rule-based risk signals.
It is used identically by:

* ``train_model.py``   (offline training on synthetic data)
* ``consumer.py``      (streaming scoring)
* ``api.py``           (online scoring service)

Keeping feature logic in exactly one place avoids the classic
training/serving skew problem: the model always sees the same features
in production that it was trained on.

Canonical transaction schema (a plain ``dict``)::

    transaction_id       str    e.g. "TXN-000001"
    user_id              str    e.g. "U-1001"
    timestamp            str (ISO 8601) or datetime
    amount               float  transaction amount in USD
    merchant_id          str
    merchant_category    str    one of MERCHANT_CATEGORIES
    payment_method       str    chip | contactless | swipe | wallet | online
    card_present         bool
    device_id            str
    is_new_device        bool   device never seen for this user before
    country              str    country where the transaction occurred
    home_country         str    user's home country
    distance_from_home_km float
    velocity_1h_count    int    user's txn count in the trailing hour
    velocity_1h_amount   float  user's txn amount sum in the trailing hour
    is_fraud             int    label, only present in generated data

The velocity fields are filled in by :class:`VelocityTracker` in the
streaming/serving layers, and by the data generator (which simulates a
sequential stream) for offline data.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

#: Merchant categories and a coarse, hand-set base-risk weight in [0, 1].
#: These weights encode analyst intuition (stored-value and
#: easily-liquidated categories are abused most often); they are *not*
#: learned from data and should be revisited per portfolio.
MERCHANT_CATEGORY_RISK: Dict[str, float] = {
    "groceries": 0.05,
    "pharmacy": 0.05,
    "fuel": 0.05,
    "utilities": 0.05,
    "dining": 0.08,
    "retail": 0.10,
    "entertainment": 0.15,
    "online_retail": 0.20,
    "travel": 0.30,
    "electronics": 0.35,
    "money_transfer": 0.60,
    "gift_cards": 0.80,
    "gambling": 0.85,
    "crypto": 0.90,
}

MERCHANT_CATEGORIES: List[str] = sorted(MERCHANT_CATEGORY_RISK)

#: Coarse base-risk weight per payment method. Card-not-present
#: ("online") and magnetic-stripe ("swipe") carry more fraud risk than
#: EMV chip or tokenised wallets.
PAYMENT_METHOD_RISK: Dict[str, float] = {
    "chip": 0.05,
    "wallet": 0.10,
    "contactless": 0.10,
    "swipe": 0.25,
    "online": 0.30,
}

DEFAULT_CATEGORY_RISK = 0.20
DEFAULT_PAYMENT_RISK = 0.15

#: The exact, ordered list of numeric features the model consumes.
#: ``compute_features`` returns a dict with exactly these keys, and
#: ``feature_vector`` flattens it in this order.
FEATURE_COLUMNS: List[str] = [
    "amount",
    "log_amount",
    "hour_of_day",
    "is_night",
    "day_of_week",
    "is_weekend",
    "category_risk",
    "payment_risk",
    "is_international",
    "log_distance_km",
    "card_not_present",
    "is_new_device",
    "velocity_1h_count",
    "velocity_1h_amount",
    "log_velocity_1h_amount",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_timestamp(value: Any) -> datetime:
    """Parse an ISO-8601 string (or pass through a datetime).

    Naive timestamps are assumed to be UTC.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _get(txn: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = txn.get(key, default)
    return default if value is None else value


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_features(txn: Dict[str, Any]) -> Dict[str, float]:
    """Compute the model feature dict for one transaction.

    Missing optional fields fall back to low-risk defaults so the
    function never crashes on partial input from an upstream producer.
    """
    ts = parse_timestamp(_get(txn, "timestamp", "1970-01-01T00:00:00Z"))
    amount = float(_get(txn, "amount", 0.0))
    velocity_count = int(_get(txn, "velocity_1h_count", 1))
    velocity_amount = float(_get(txn, "velocity_1h_amount", amount))
    distance = float(_get(txn, "distance_from_home_km", 0.0))

    category = str(_get(txn, "merchant_category", "retail"))
    payment = str(_get(txn, "payment_method", "chip"))
    country = str(_get(txn, "country", "US"))
    home_country = str(_get(txn, "home_country", country))

    hour = ts.hour
    return {
        "amount": amount,
        "log_amount": math.log1p(max(amount, 0.0)),
        "hour_of_day": float(hour),
        "is_night": 1.0 if hour < 5 else 0.0,
        "day_of_week": float(ts.weekday()),
        "is_weekend": 1.0 if ts.weekday() >= 5 else 0.0,
        "category_risk": MERCHANT_CATEGORY_RISK.get(category, DEFAULT_CATEGORY_RISK),
        "payment_risk": PAYMENT_METHOD_RISK.get(payment, DEFAULT_PAYMENT_RISK),
        "is_international": 1.0 if country != home_country else 0.0,
        "log_distance_km": math.log1p(max(distance, 0.0)),
        "card_not_present": 0.0 if bool(_get(txn, "card_present", True)) else 1.0,
        "is_new_device": 1.0 if bool(_get(txn, "is_new_device", False)) else 0.0,
        "velocity_1h_count": float(velocity_count),
        "velocity_1h_amount": velocity_amount,
        "log_velocity_1h_amount": math.log1p(max(velocity_amount, 0.0)),
    }


def feature_vector(txn: Dict[str, Any]) -> List[float]:
    """Return the feature values as a flat list in FEATURE_COLUMNS order."""
    feats = compute_features(txn)
    return [feats[col] for col in FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Rule-based risk signals
# ---------------------------------------------------------------------------

def risk_signals(txn: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate interpretable, analyst-style rules against a transaction.

    Returns a list of fired signals, each ``{"name", "weight", "detail"}``
    where ``weight`` is a contribution toward a 0-100 rule score. Rules
    are deliberately transparent so an analyst can explain any alert —
    a property pure ML scores do not have.
    """
    signals: List[Dict[str, Any]] = []

    def add(name: str, weight: float, detail: str) -> None:
        signals.append({"name": name, "weight": weight, "detail": detail})

    amount = float(_get(txn, "amount", 0.0))
    ts = parse_timestamp(_get(txn, "timestamp", "1970-01-01T00:00:00Z"))
    category = str(_get(txn, "merchant_category", "retail"))
    cat_risk = MERCHANT_CATEGORY_RISK.get(category, DEFAULT_CATEGORY_RISK)
    country = str(_get(txn, "country", "US"))
    home_country = str(_get(txn, "home_country", country))
    distance = float(_get(txn, "distance_from_home_km", 0.0))
    velocity_count = int(_get(txn, "velocity_1h_count", 1))
    card_present = bool(_get(txn, "card_present", True))

    if amount >= 3000:
        add("very_high_amount", 35, f"Amount ${amount:,.2f} >= $3,000")
    elif amount >= 1000:
        add("high_amount", 20, f"Amount ${amount:,.2f} >= $1,000")

    if country != home_country:
        add("international", 20,
            f"Transaction country {country} != home country {home_country}")

    if velocity_count >= 5:
        add("extreme_velocity", 25,
            f"{velocity_count} transactions by this user in the last hour")
    elif velocity_count >= 3:
        add("elevated_velocity", 12,
            f"{velocity_count} transactions by this user in the last hour")

    if ts.hour < 5:
        add("night_transaction", 10,
            f"Transaction at {ts.hour:02d}:{ts.minute:02d} local time")

    if bool(_get(txn, "is_new_device", False)):
        add("new_device", 15, "Device not previously seen for this user")

    if cat_risk >= 0.60:
        add("high_risk_category", 20,
            f"Merchant category '{category}' is high risk")
    elif cat_risk >= 0.35:
        add("medium_risk_category", 10,
            f"Merchant category '{category}' is medium-high risk")

    if distance > 1000:
        add("far_from_home", 12,
            f"{distance:,.0f} km from the user's home area")

    if not card_present and amount > 500:
        add("large_card_not_present", 10,
            f"Card-not-present amount ${amount:,.2f} > $500")

    return signals


def rule_score(txn: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    """Combine fired rule weights into a 0-100 score (capped)."""
    signals = risk_signals(txn)
    score = int(min(100, round(sum(s["weight"] for s in signals))))
    return score, signals


# ---------------------------------------------------------------------------
# Velocity tracking (streaming state)
# ---------------------------------------------------------------------------

class VelocityTracker:
    """In-memory trailing-window velocity state, keyed by user.

    The streaming consumer and the API keep one instance and call
    :meth:`update` for every transaction *before* scoring it, so the
    velocity features describe the user's recent behaviour including
    the current transaction. A production deployment would back this
    with a state store (e.g. Redis or Kafka Streams state); the
    interface here is intentionally identical to what such a store
    would expose.
    """

    def __init__(self, window_seconds: int = 3600) -> None:
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)

    def update(self, user_id: str, timestamp: Any, amount: float) -> Tuple[int, float]:
        """Record a transaction and return (count, amount_sum) in window."""
        ts = parse_timestamp(timestamp).timestamp()
        events = self._events[user_id]
        events.append((ts, float(amount)))
        cutoff = ts - self.window_seconds
        while events and events[0][0] < cutoff:
            events.popleft()
        return len(events), float(sum(a for _, a in events))

    def peek(self, user_id: str, timestamp: Any) -> Tuple[int, float]:
        """Window stats as of ``timestamp`` without recording anything."""
        ts = parse_timestamp(timestamp).timestamp()
        cutoff = ts - self.window_seconds
        count = 0
        total = 0.0
        for event_ts, amt in self._events.get(user_id, ()):  # oldest -> newest
            if event_ts >= cutoff and event_ts <= ts:
                count += 1
                total += amt
        return count, total
