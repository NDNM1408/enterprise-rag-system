"""Shared pytest fixtures for data-api tests."""

import pytest
import sys
import os

# Set required env vars BEFORE importing any app module so module-level
# singletons (DatabaseSession, settings) initialize without errors.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
os.environ.setdefault("BUCKET_NAME", "test-bucket")

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

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Patch the DB engine creation so no real asyncpg/DB connection is needed
with patch("sqlalchemy.ext.asyncio.create_async_engine"):
    from app.infrastructure.connectors.postgres.repositories.knowledge_base_repository import KnowledgeBaseRepository
    from app.infrastructure.connectors.postgres.repositories.document_repository import DocumentRepository
    from app.infrastructure.clients.s3_client_service import S3ClientService


# ---------------------------------------------------------------------------
# Mock repository factories
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kb_repo() -> MagicMock:
    repo = MagicMock(spec=KnowledgeBaseRepository)
    repo.get = AsyncMock()
    repo.create = AsyncMock()
    repo.delete = AsyncMock()
    repo.count = AsyncMock(return_value=0)
    repo.paging = AsyncMock(return_value=[])
    repo.update = AsyncMock()
    return repo


@pytest.fixture
def mock_doc_repo() -> MagicMock:
    repo = MagicMock(spec=DocumentRepository)
    repo.get = AsyncMock(return_value=[])
    repo.get_by_ids = AsyncMock(return_value=[])
    repo.find_conflicts = AsyncMock(return_value=[])
    repo.find_etag_conflicts = AsyncMock(return_value=[])
    repo.bulk_create = AsyncMock()
    repo.bulk_delete = AsyncMock()
    return repo


@pytest.fixture
def mock_s3() -> MagicMock:
    s3 = MagicMock(spec=S3ClientService)
    s3.upload_file = AsyncMock()
    s3.delete_file = AsyncMock()
    return s3


# ---------------------------------------------------------------------------
# FastAPI test clients
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    """Synchronous TestClient for simple endpoint tests."""
    # Import here to avoid triggering settings validation without .env
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    os.environ.setdefault("BUCKET_NAME", "test-bucket")

    from main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
async def async_api_client():
    """Async HTTPX client for async endpoint tests."""
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    os.environ.setdefault("BUCKET_NAME", "test-bucket")

    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
