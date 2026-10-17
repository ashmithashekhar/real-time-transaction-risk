"""Real-Time Transaction Risk Platform.

An original reference implementation of a streaming transaction-risk
pipeline: synthetic data generation -> Kafka -> streaming scoring
(rule signals + an ML model trained on synthetic data) -> FastAPI
scoring service, with a Spark Structured Streaming variant of the
streaming job.

All data in this project is synthetic. No real customer, bank, or
employer data is used anywhere.
"""

__version__ = "1.0.0"
