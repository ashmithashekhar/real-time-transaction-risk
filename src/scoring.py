"""Hybrid transaction scoring: interpretable rules + ML model.

The final risk score blends two views of a transaction:

* a **rule score** (0-100) from transparent analyst rules
  (:mod:`src.features`) — always available, always explainable;
* a **model score** (0-100) from a scikit-learn model trained on
  synthetic data — captures interactions rules miss.

Blend (when a model is loaded)::

    final = 0.65 * model_score + 0.35 * rule_score

Without a model file the service degrades gracefully to rules only,
so the platform still runs end-to-end before any training has happened.

Decision bands (configurable constants, deliberately conservative —
in a real portfolio these are tuned to alert capacity and loss rates):

* ``final < 40``          -> "approve"
* ``40 <= final < 70``    -> "review"
* ``final >= 70``         -> "block"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .features import FEATURE_COLUMNS, feature_vector, rule_score

REVIEW_THRESHOLD = 40
BLOCK_THRESHOLD = 70
MODEL_WEIGHT = 0.65
RULE_WEIGHT = 0.35

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"


def decide(final_score: float) -> str:
    if final_score >= BLOCK_THRESHOLD:
        return "block"
    if final_score >= REVIEW_THRESHOLD:
        return "review"
    return "approve"


class RiskScorer:
    """Scores transaction dicts; loads an ML model if one is available."""

    def __init__(self, model_path: Optional[str | os.PathLike] = None) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model = None
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.model_metadata: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            import joblib

            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self.model = bundle["model"]
                self.feature_columns = bundle.get(
                    "feature_columns", list(FEATURE_COLUMNS)
                )
                self.model_metadata = bundle.get("metadata", {})
            else:  # bare estimator saved directly
                self.model = bundle
        except Exception:
            # A corrupt/unreadable model must never take scoring down;
            # the platform falls back to rules-only mode.
            self.model = None

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def model_probability(self, txn: Dict[str, Any]) -> Optional[float]:
        """Fraud probability in [0, 1], or None if no model is loaded."""
        if self.model is None:
            return None
        feats = feature_vector(txn)
        # Reorder defensively in case the saved bundle used a
        # different (older) column order.
        if self.feature_columns != FEATURE_COLUMNS:
            from .features import compute_features

            values = compute_features(txn)
            feats = [values[col] for col in self.feature_columns]
        proba = self.model.predict_proba([feats])[0]
        classes = list(self.model.classes_)
        fraud_idx = classes.index(1) if 1 in classes else int(max(classes))
        return float(proba[fraud_idx])

    def score(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Return the full scoring result for one transaction."""
        rules, signals = rule_score(txn)
        prob = self.model_probability(txn)
        model_points = None if prob is None else round(prob * 100, 1)

        if model_points is None:
            final = float(rules)
        else:
            final = MODEL_WEIGHT * model_points + RULE_WEIGHT * rules

        return {
            "transaction_id": txn.get("transaction_id"),
            "rule_score": rules,
            "model_score": model_points,
            "risk_score": round(final, 1),
            "decision": decide(final),
            "signals": signals,
        }
