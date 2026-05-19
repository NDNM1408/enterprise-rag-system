"""Pytest configuration for document-parsing tests.

Ensures ``src/`` is importable so tests can pull in callbacks, S3 helpers,
and schemas without booting the whole worker (no MinerU / ONNX needed).
"""
import os
import sys

# Required env vars referenced at import time by settings.py
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("S3_BUCKET", "document-parsing")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
