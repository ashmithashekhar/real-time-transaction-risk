"""Train the transaction risk model on synthetic data.

Pipeline: generate (or load) synthetic transactions -> build features
with the exact same :mod:`src.features` code used at serving time ->
train a RandomForest classifier -> evaluate on a held-out split ->
save a model bundle plus a JSON metadata file with the *measured*
evaluation numbers.

All metrics printed/saved by this script come from synthetic data and
say nothing about performance on any real portfolio.

Usage::

    python -m src.train_model --n 20000 --seed 42
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .features import FEATURE_COLUMNS, compute_features
from .scoring import DEFAULT_MODEL_PATH

METADATA_PATH = DEFAULT_MODEL_PATH.parent / "model_metadata.json"


def build_dataset(txns):
    """Return (X, y) using the shared serving-time feature code."""
    X, y = [], []
    for txn in txns:
        feats = compute_features(txn)
        X.append([feats[col] for col in FEATURE_COLUMNS])
        y.append(int(txn.get("is_fraud", 0)))
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the risk model")
    parser.add_argument("--n", type=int, default=20000,
                        help="Synthetic transactions to generate if no --data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=None,
                        help="Optional CSV produced by src.data_generator")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (average_precision_score,
                                 classification_report, roc_auc_score)
    from sklearn.model_selection import train_test_split

    if args.data:
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        txns = df.to_dict("records")
    else:
        from .data_generator import generate_transactions

        print(f"Generating {args.n:,} synthetic transactions (seed={args.seed})")
        txns = generate_transactions(n=args.n, seed=args.seed)

    X, y = build_dataset(txns)
    fraud = sum(y)
    print(f"Dataset: {len(y):,} rows | fraud {fraud:,} ({fraud / len(y):.2%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
    }
    print("\nHold-out evaluation (synthetic data only):")
    print(f"  ROC-AUC: {metrics['roc_auc']}   PR-AUC: {metrics['pr_auc']}")
    print(classification_report(y_test, pred, digits=3))

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )
    print("Top feature importances:")
    for name, imp in importances[:5]:
        print(f"  {name:<26} {imp:.3f}")

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "synthetic (src.data_generator)",
        "seed": args.seed,
        "n_total": len(y),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "fraud_rate": round(fraud / len(y), 4),
        "model_type": "RandomForestClassifier(n_estimators=200, "
                      "class_weight='balanced_subsample')",
        "feature_columns": FEATURE_COLUMNS,
        "holdout_metrics_synthetic_data": metrics,
        "note": "Metrics measured on held-out SYNTHETIC data. They are "
                "not evidence of performance on real transactions.",
    }

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS,
         "metadata": metadata},
        out,
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model bundle -> {out}")
    print(f"Saved metadata     -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
