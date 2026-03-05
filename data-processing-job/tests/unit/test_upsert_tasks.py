"""
Unit tests for the upsert_chunk Celery task.

Verifies that:
- Idempotency: already-Succeed chunks are skipped.
- Happy path: S3 fetch → embed → vector upsert → chunk status Succeed.
- Error path: chunk is marked Failed; exception is re-raised for retry.
- No new engine is created per task call (uses container.session_factory).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_chunk_repo(status="Processing"):
    repo = MagicMock()
    repo.get_status = AsyncMock(return_value=status)
    repo.set_status = AsyncMock()
    return repo


class TestUpsertChunk:

    @patch("app.celery_app.tasks.upsert_tasks.EmbeddingWriterRepository")
    @patch("app.celery_app.tasks.upsert_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.upsert_tasks.container")
    def test_happy_path(self, mock_container, mock_chunk_repo_cls, mock_emb_repo_cls):
        """Fetches text, embeds it, upserts to vector store, marks Succeed."""
        mock_container.session_factory = MagicMock()
        mock_container.s3.get_txt_file_content = AsyncMock(return_value="chunk text")
        mock_container.embedding_service.get_embedding = AsyncMock(
            return_value=[0.1] * 1024
        )

        # session_factory() context manager
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_container.session_factory.return_value = mock_session

        chunk_repo = _make_chunk_repo(status="Processing")
        mock_chunk_repo_cls.return_value = chunk_repo

        emb_repo = MagicMock()
        emb_repo.upsert = AsyncMock()
        mock_emb_repo_cls.return_value = emb_repo

        from app.celery_app.tasks.upsert_tasks import upsert_chunk

        result = upsert_chunk.run(
            chunk_id="chunk-1",
            s3_path="path/chunk_0.txt",
            knowledge_base_id="kb-1",
            document_id="doc-1",
            metadata={"chunk_order_index": 0, "tokens": 10},
            correlation_id="corr-1",
        )

        assert result == {"chunk_id": "chunk-1", "status": "success"}
        emb_repo.upsert.assert_awaited_once()
        chunk_repo.set_status.assert_awaited_with("chunk-1", "Succeed")

    @patch("app.celery_app.tasks.upsert_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.upsert_tasks.container")
    def test_skips_already_succeeded_chunk(self, mock_container, mock_chunk_repo_cls):
        """Chunk already Succeed → returns skipped, no embedding call."""
        mock_container.session_factory = MagicMock()
        chunk_repo = _make_chunk_repo(status="Succeed")
        mock_chunk_repo_cls.return_value = chunk_repo

        from app.celery_app.tasks.upsert_tasks import upsert_chunk

        result = upsert_chunk.run(
            chunk_id="chunk-1",
            s3_path="path.txt",
            knowledge_base_id="kb-1",
            document_id="doc-1",
            metadata={},
        )

        assert result["status"] == "skipped"
        mock_container.embedding_service.get_embedding.assert_not_called()

    @patch("app.celery_app.tasks.upsert_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.upsert_tasks.container")
    def test_skips_missing_chunk(self, mock_container, mock_chunk_repo_cls):
        """Chunk not in DB → returns skipped."""
        mock_container.session_factory = MagicMock()
        chunk_repo = _make_chunk_repo(status=None)
        mock_chunk_repo_cls.return_value = chunk_repo

        from app.celery_app.tasks.upsert_tasks import upsert_chunk

        result = upsert_chunk.run(
            chunk_id="chunk-missing",
            s3_path="path.txt",
            knowledge_base_id="kb-1",
            document_id="doc-1",
            metadata={},
        )

        assert result["status"] == "skipped"

    @patch("app.celery_app.tasks.upsert_tasks.EmbeddingWriterRepository")
    @patch("app.celery_app.tasks.upsert_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.upsert_tasks.container")
    def test_marks_chunk_failed_on_embedding_error(
        self, mock_container, mock_chunk_repo_cls, mock_emb_repo_cls
    ):
        """Embedding API failure → chunk marked Failed, exception re-raised."""
        mock_container.session_factory = MagicMock()
        mock_container.s3.get_txt_file_content = AsyncMock(return_value="text")
        mock_container.embedding_service.get_embedding = AsyncMock(
            side_effect=RuntimeError("API down")
        )

        chunk_repo = _make_chunk_repo(status="Processing")
        mock_chunk_repo_cls.return_value = chunk_repo

        from app.celery_app.tasks.upsert_tasks import upsert_chunk

        with pytest.raises(Exception):
            upsert_chunk.run(
                chunk_id="chunk-1",
                s3_path="path.txt",
                knowledge_base_id="kb-1",
                document_id="doc-1",
                metadata={},
            )

        chunk_repo.set_status.assert_awaited_with("chunk-1", "Failed")

    @patch("app.celery_app.tasks.upsert_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.upsert_tasks.container")
    def test_no_engine_created_per_task(self, mock_container, mock_chunk_repo_cls):
        """
        Tasks must use container.session_factory, not create a new engine.
        Verified by asserting create_async_engine is never called during task.
        """
        mock_container.session_factory = MagicMock()
        chunk_repo = _make_chunk_repo(status="Succeed")  # skip after idempotency check
        mock_chunk_repo_cls.return_value = chunk_repo

        with patch("app.celery_app.tasks.upsert_tasks.create_async_engine", create=True) as mock_engine:
            from app.celery_app.tasks.upsert_tasks import upsert_chunk
            upsert_chunk.run(
                chunk_id="c1",
                s3_path="p.txt",
                knowledge_base_id="kb-1",
                document_id="d1",
                metadata={},
            )
            mock_engine.assert_not_called()

    @patch("app.celery_app.tasks.upsert_tasks.EmbeddingWriterRepository")
    @patch("app.celery_app.tasks.upsert_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.upsert_tasks.container")
    def test_backward_compat_is_last_chunk_param_accepted(
        self, mock_container, mock_chunk_repo_cls, mock_emb_repo_cls
    ):
        """is_last_chunk=True is accepted without error (backward compat)."""
        mock_container.session_factory = MagicMock()
        chunk_repo = _make_chunk_repo(status="Succeed")
        mock_chunk_repo_cls.return_value = chunk_repo

        from app.celery_app.tasks.upsert_tasks import upsert_chunk

        # Must not raise
        result = upsert_chunk.run(
            chunk_id="c1",
            s3_path="p.txt",
            knowledge_base_id="kb-1",
            document_id="d1",
            metadata={},
            is_last_chunk=True,
        )
        assert result["status"] == "skipped"
