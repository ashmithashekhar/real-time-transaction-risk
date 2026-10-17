"""Kafka producer: publishes synthetic transactions to a topic.

Generates transactions with :mod:`src.data_generator` (which already
computes trailing-hour velocity fields as a sequential stream would)
and publishes them as JSON to the ``transactions`` topic, optionally
throttled to simulate a live feed.

Start a local broker first::

    docker compose up -d

Usage::

    python -m src.producer --count 1000 --interval 0.1

Environment:
    KAFKA_BOOTSTRAP_SERVERS  default "localhost:9092"
    KAFKA_TOPIC               default "transactions"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish synthetic txns to Kafka")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=float, default=0.0,
                        help="Seconds to sleep between messages (stream feel)")
    args = parser.parse_args()

    try:
        from kafka import KafkaProducer
        from kafka.errors import KafkaError
    except ImportError:
        print("kafka-python is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "transactions")

    from .data_generator import generate_transactions

    txns = generate_transactions(n=args.count, seed=args.seed)
    print(f"Publishing {len(txns):,} transactions to topic '{topic}' "
          f"via {bootstrap} ...")

    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=10000,
        )
    except KafkaError:
        # Covers NoBrokersAvailable (kafka-python 2.x) and
        # KafkaConnectionError/KafkaTimeoutError (kafka-python 3.x).
        print(f"\nCould not reach a Kafka broker at {bootstrap}.")
        print("Start the local broker with:  docker compose up -d")
        sys.exit(1)

    sent = 0
    try:
        for txn in txns:
            producer.send(topic, value=txn)
            sent += 1
            if args.interval:
                time.sleep(args.interval)
        producer.flush()
    except KafkaError as exc:
        print(f"Kafka error after {sent} messages: {exc}")
        sys.exit(1)
    finally:
        producer.close()
    print(f"Done. Published {sent:,} messages.")


if __name__ == "__main__":
    main()
