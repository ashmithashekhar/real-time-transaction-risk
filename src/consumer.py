"""Streaming consumer: scores transactions as they arrive.

Consumes the ``transactions`` Kafka topic, scores every message with
the hybrid rule+model scorer (:mod:`src.scoring`), logs a line per
transaction, and prints an alert block whenever the decision is
``review`` or ``block``. Decision counters are reported at the end /
on Ctrl+C.

No broker handy? ``--simulate`` replays the synthetic generator
through the exact same scoring path, which is how the pipeline is
demoed and smoke-tested in CI::

    python -m src.consumer --simulate --count 500
    python -m src.consumer                 # real Kafka mode

Environment:
    KAFKA_BOOTSTRAP_SERVERS  default "localhost:9092"
    KAFKA_TOPIC               default "transactions"
    RISK_MODEL_PATH           optional path to a trained model bundle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from .scoring import RiskScorer


def handle(txn: dict, scorer: RiskScorer, stats: Counter) -> None:
    result = scorer.score(txn)
    stats[result["decision"]] += 1
    stats["total"] += 1
    line = (f"[{txn.get('timestamp', '?')}] {txn.get('transaction_id', '?')} "
            f"user={txn.get('user_id', '?')} "
            f"amount=${float(txn.get('amount', 0)):>8,.2f} "
            f"cat={txn.get('merchant_category', '?'):<14} "
            f"risk={result['risk_score']:>5} -> {result['decision'].upper()}")
    print(line)
    if result["decision"] != "approve":
        print("  ** ALERT **")
        for sig in result["signals"]:
            print(f"     - {sig['name']}: {sig['detail']}")
        if result["model_score"] is not None:
            print(f"     - model_score: {result['model_score']}")


def run_simulation(count: int, seed: int, scorer: RiskScorer) -> None:
    from .data_generator import generate_transactions

    print(f"SIMULATION MODE: replaying {count:,} synthetic transactions "
          f"through the scoring path\n")
    stats: Counter = Counter()
    for txn in generate_transactions(n=count, seed=seed):
        handle(txn, scorer, stats)
    print_summary(stats)


def print_summary(stats: Counter) -> None:
    total = stats.get("total", 0)
    print("\n--- Session summary ---")
    for decision in ("approve", "review", "block"):
        n = stats.get(decision, 0)
        pct = (n / total * 100) if total else 0.0
        print(f"  {decision:<8} {n:>6,}  ({pct:.1f}%)")
    print(f"  {'total':<8} {total:>6,}")


def run_kafka(scorer: RiskScorer) -> None:
    try:
        from kafka import KafkaConsumer
        from kafka.errors import KafkaError
    except ImportError:
        print("kafka-python is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "transactions")
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap,
            group_id="risk-scorers",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        # Force a metadata fetch so a missing broker fails fast & clearly.
        consumer.topics()
    except KafkaError:
        # Covers NoBrokersAvailable (kafka-python 2.x) and
        # KafkaConnectionError/KafkaTimeoutError (kafka-python 3.x).
        print(f"Could not reach a Kafka broker at {bootstrap}.")
        print("Start one with:  docker compose up -d")
        print("Or try the broker-free demo:  python -m src.consumer --simulate")
        sys.exit(1)

    print(f"Consuming topic '{topic}' from {bootstrap} (Ctrl+C to stop)\n")
    stats: Counter = Counter()
    try:
        while True:
            batch = consumer.poll(timeout_ms=1000)
            for records in batch.values():
                for record in records:
                    handle(record.value, scorer, stats)
    except KeyboardInterrupt:
        print("\nStopping consumer.")
    finally:
        consumer.close()
        print_summary(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming risk-scoring consumer")
    parser.add_argument("--simulate", action="store_true",
                        help="Replay synthetic data instead of consuming Kafka")
    parser.add_argument("--count", type=int, default=500,
                        help="Transactions to replay in --simulate mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.getenv("RISK_MODEL_PATH"),
                        help="Path to a trained model bundle (default: "
                             "models/risk_model.joblib if present)")
    args = parser.parse_args()

    scorer = RiskScorer(model_path=args.model) if args.model else RiskScorer()
    mode = "hybrid (rules + model)" if scorer.model_loaded else "rules only"
    print(f"Scorer mode: {mode}\n")

    if args.simulate:
        run_simulation(args.count, args.seed, scorer)
    else:
        run_kafka(scorer)


if __name__ == "__main__":
    main()
