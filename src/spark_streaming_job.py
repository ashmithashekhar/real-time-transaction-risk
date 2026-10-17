"""Spark Structured Streaming variant of the streaming job.

Where ``consumer.py`` is a single-process Python consumer, this job
expresses the same first line of defence — *per-user velocity
aggregation* — as a distributed Spark Structured Streaming query:

1. Read the ``transactions`` topic from Kafka.
2. Parse the JSON payload into typed columns.
3. Compute, per user, a 1-hour event-time window count and amount sum
   (with a 10-minute watermark for late data).
4. Emit windows whose velocity looks abusive (many transactions, or a
   large total) to the console as a high-velocity alert stream.

In a production deployment these aggregated features would feed the
same scorer used by ``consumer.py`` / ``api.py`` (for example via a
feature store or a ``foreachBatch`` sink calling the scoring service).
This job keeps the aggregation layer self-contained and readable.

Requirements (heavier than the rest of the project, hence optional):
a JVM, ``pyspark``, and the Spark/Kafka connector, e.g.::

    pip install pyspark
    spark-submit --packages \
      org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      -m src.spark_streaming_job

or simply ``python -m src.spark_streaming_job`` (the session is
configured with the connector package; the first run downloads it).

If PySpark is not installed, the script explains all of this and
exits cleanly instead of crashing with an import traceback.

Environment:
    KAFKA_BOOTSTRAP_SERVERS  default "localhost:9092"
    KAFKA_TOPIC               default "transactions"
"""

from __future__ import annotations

import os
import sys

KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

SETUP_NOTE = """\
PySpark is not installed in this environment, so the streaming job
cannot start. That is expected — Spark is an optional, heavyweight
dependency of this project.

What this job would do (see the module docstring):
  * read the 'transactions' Kafka topic as a Structured Stream,
  * aggregate per-user 1-hour velocity windows,
  * stream high-velocity windows to the console as alerts.

To run it:
  1. Install a JDK (11 or 17) and PySpark:   pip install pyspark
  2. Start Kafka:                            docker compose up -d
  3. Produce some data:                      python -m src.producer --count 1000
  4. Run the job:                            python -m src.spark_streaming_job

Everything else in this platform (generator, training, consumer,
API) runs without Spark.
"""


def main() -> None:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, from_json, window
        from pyspark.sql.types import (BooleanType, DoubleType, LongType,
                                       StringType, StructField, StructType,
                                       TimestampType)
    except ImportError:
        print(SETUP_NOTE)
        return

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "transactions")

    spark = (
        SparkSession.builder
        .appName("transaction-risk-velocity")
        .config("spark.jars.packages", KAFKA_PACKAGE)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("transaction_id", StringType()),
        StructField("user_id", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("amount", DoubleType()),
        StructField("merchant_id", StringType()),
        StructField("merchant_category", StringType()),
        StructField("payment_method", StringType()),
        StructField("card_present", BooleanType()),
        StructField("device_id", StringType()),
        StructField("is_new_device", BooleanType()),
        StructField("country", StringType()),
        StructField("home_country", StringType()),
        StructField("distance_from_home_km", DoubleType()),
        StructField("velocity_1h_count", LongType()),
        StructField("velocity_1h_amount", DoubleType()),
        StructField("is_fraud", LongType()),
    ])

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .load()
    )

    txns = raw.select(
        from_json(col("value").cast("string"), schema).alias("t")
    ).select("t.*")

    velocity = (
        txns
        .withWatermark("timestamp", "10 minutes")
        .groupBy(window(col("timestamp"), "1 hour"), col("user_id"))
        .agg(
            {"amount": "sum", "transaction_id": "count"}
        )
        .withColumnRenamed("sum(amount)", "window_amount")
        .withColumnRenamed("count(transaction_id)", "window_count")
    )

    alerts = velocity.filter(
        (col("window_count") >= 5) | (col("window_amount") >= 5000)
    )

    query = (
        alerts.writeStream
        .format("console")
        .outputMode("update")
        .option("truncate", "false")
        .start()
    )
    print(f"Streaming per-user velocity alerts from '{topic}' "
          f"via {bootstrap} (Ctrl+C to stop)")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        spark.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
