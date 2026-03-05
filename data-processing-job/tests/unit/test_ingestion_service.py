"""Unit tests for IngestionService."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import AsyncMock, MagicMock
from app.application.services.ingestion_service import IngestionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_service(mock_repo=None) -> IngestionService:
    """Create an IngestionService with a mocked repository."""
    svc = object.__new__(IngestionService)
    svc.repository = mock_repo or MagicMock()
    return svc


# ---------------------------------------------------------------------------
# upsert_embedding
# ---------------------------------------------------------------------------

class TestUpsertEmbedding:
    @pytest.mark.asyncio
    async def test_upsert_calls_repository(self):
        repo = MagicMock()
        repo.upsert = AsyncMock()
        svc = make_service(repo)

        await svc.upsert_embedding(
            chunk_id="chunk-001",
            document_id="doc-001",
            kb_id="kb-001",
            embedding=[0.1] * 1024,
            content="Some chunk text",
            metadata={"source": "test.html"},
        )

        repo.upsert.assert_awaited_once_with(
            chunk_id="chunk-001",
            document_id="doc-001",
            kb_id="kb-001",
            embedding=[0.1] * 1024,
            content="Some chunk text",
            metadata={"source": "test.html"},
        )

    @pytest.mark.asyncio
    async def test_upsert_propagates_repository_errors(self):
        repo = MagicMock()
        repo.upsert = AsyncMock(side_effect=RuntimeError("DB error"))
        svc = make_service(repo)

        with pytest.raises(RuntimeError, match="DB error"):
            await svc.upsert_embedding(
                chunk_id="chunk-001",
                document_id="doc-001",
                kb_id="kb-001",
                embedding=[0.0] * 1024,
                content="text",
                metadata={},
            )

    @pytest.mark.asyncio
    async def test_upsert_with_empty_metadata(self):
        repo = MagicMock()
        repo.upsert = AsyncMock()
        svc = make_service(repo)

        await svc.upsert_embedding(
            chunk_id="c1",
            document_id="d1",
            kb_id="k1",
            embedding=[0.5] * 1024,
            content="hello",
            metadata={},
        )
        repo.upsert.assert_awaited_once()


# ---------------------------------------------------------------------------
# delete_document_embeddings
# ---------------------------------------------------------------------------

class TestDeleteDocumentEmbeddings:
    @pytest.mark.asyncio
    async def test_delete_calls_repository(self):
        repo = MagicMock()
        repo.delete_by_document = AsyncMock(return_value=5)
        svc = make_service(repo)

        count = await svc.delete_document_embeddings("doc-001")

        repo.delete_by_document.assert_awaited_once_with("doc-001")
        assert count == 5

    @pytest.mark.asyncio
    async def test_delete_returns_zero_when_nothing_deleted(self):
        repo = MagicMock()
        repo.delete_by_document = AsyncMock(return_value=0)
        svc = make_service(repo)

        count = await svc.delete_document_embeddings("nonexistent-doc")
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_propagates_errors(self):
        repo = MagicMock()
        repo.delete_by_document = AsyncMock(side_effect=RuntimeError("connection lost"))
        svc = make_service(repo)

        with pytest.raises(RuntimeError, match="connection lost"):
            await svc.delete_document_embeddings("doc-xyz")
