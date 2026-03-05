"""
Shared fixtures for data-processing-job tests.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
os.environ.setdefault("PREPROCESS_BUCKET_NAME", "preprocess-bucket")
os.environ.setdefault("UPSERT_BUCKET_NAME", "upsert-bucket")
os.environ.setdefault("DOCUMENT_AI_URL", "http://localhost:11434/v1")
os.environ.setdefault("GOOGLE_GENAI_API_KEY", "test-key")

# GraphRAG / Neo4j test env vars
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")
os.environ.setdefault("GRAPHRAG_LLM_API_BASE", "http://localhost:4000/v1")
os.environ.setdefault("GRAPHRAG_LLM_MODEL", "gpt-4o-mini")
os.environ.setdefault("GRAPHRAG_LLM_API_KEY", "fake")
os.environ.setdefault("GRAPHRAG_WORKING_DIR", "/tmp/graphrag_test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DATABASE", "test")
