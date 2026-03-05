"""
Integration tests for the preprocess_document and finalize_document tasks.

External dependencies (S3, DB, embedding API) are mocked so no real
infrastructure is required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _format_name_path (moved to document_preprocess_service)
# ---------------------------------------------------------------------------

class TestFormatNamePath:
    def test_replaces_special_chars(self):
        from app.application.services.document_preprocess_service import _format_name_path
        result = _format_name_path("http://example.com/path/to/doc.html")
        assert "://" not in result
        assert "-" not in result
        assert "." not in result
        assert "/" not in result

    def test_spaces_replaced(self):
        from app.application.services.document_preprocess_service import _format_name_path
        result = _format_name_path("my document file.html")
        assert " " not in result


# ---------------------------------------------------------------------------
# preprocess_document — calls DocumentPreprocessService and dispatches chord
# ---------------------------------------------------------------------------

class TestPreprocessDocument:

    @patch("app.celery_app.tasks.preprocess_tasks.chord")
    @patch("app.celery_app.tasks.preprocess_tasks.group")
    @patch("app.celery_app.tasks.preprocess_tasks.DocumentPreprocessService")
    @patch("app.celery_app.tasks.preprocess_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.container")
    def test_happy_path_dispatches_chord(
        self,
        mock_container,
        mock_doc_repo_cls,
        mock_chunk_repo_cls,
        mock_svc_cls,
        mock_group,
        mock_chord,
    ):
        """Happy path: preprocess succeeds → chord with N upsert tasks is dispatched."""
        mock_container.session_factory = MagicMock()
        mock_container.s3 = MagicMock()
        mock_container.parser = MagicMock()
        mock_container.splitter = MagicMock()

        from app.application.services.document_preprocess_service import ChunkRecord

        fake_chunks = [
            ChunkRecord(
                id=f"chunk-{i}",
                content=f"content {i}",
                document_id="doc-1",
                kb_id="kb-1",
                doc_name="test.html",
                s3_path=f"path_{i}.txt",
                chunk_s3_url=f"s3://bucket/kb-1/path_{i}.txt",
                metadata={"chunk_order_index": i, "tokens": 10},
            )
            for i in range(3)
        ]

        mock_svc = MagicMock()
        mock_svc.preprocess = AsyncMock(return_value=fake_chunks)
        mock_svc_cls.return_value = mock_svc

        mock_chord_instance = MagicMock()
        mock_chord.return_value = mock_chord_instance

        from app.celery_app.tasks.preprocess_tasks import preprocess_document

        result = preprocess_document.run(
            document_id="doc-1",
            knowledge_base_id="kb-1",
            name="test.html",
            embedding_model_id="emb-model",
            bucket="my-bucket",
            correlation_id="corr-1",
        )

        assert result["chunk_count"] == 3
        assert result["document_id"] == "doc-1"

        # chord must be dispatched
        mock_chord.assert_called_once()
        mock_chord_instance.apply_async.assert_called_once()

    @patch("app.celery_app.tasks.preprocess_tasks.DocumentPreprocessService")
    @patch("app.celery_app.tasks.preprocess_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.container")
    def test_document_not_found_returns_zero_chunks(
        self, mock_container, mock_doc_repo_cls, mock_chunk_repo_cls, mock_svc_cls
    ):
        """DocumentNotFoundError → task returns chunk_count=0, no chord."""
        from app.application.services.document_preprocess_service import DocumentNotFoundError

        mock_container.session_factory = MagicMock()
        mock_svc = MagicMock()
        mock_svc.preprocess = AsyncMock(side_effect=DocumentNotFoundError("not found"))
        mock_svc_cls.return_value = mock_svc

        from app.celery_app.tasks.preprocess_tasks import preprocess_document

        result = preprocess_document.run(
            document_id="doc-missing",
            knowledge_base_id="kb-1",
            name="test.html",
            embedding_model_id="emb",
            bucket="bucket",
        )
        assert result["chunk_count"] == 0

    @patch("app.celery_app.tasks.preprocess_tasks.DocumentPreprocessService")
    @patch("app.celery_app.tasks.preprocess_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.container")
    def test_already_processed_returns_zero_chunks(
        self, mock_container, mock_doc_repo_cls, mock_chunk_repo_cls, mock_svc_cls
    ):
        """AlreadyProcessedError → task returns chunk_count=0, no chord."""
        from app.application.services.document_preprocess_service import AlreadyProcessedError

        mock_container.session_factory = MagicMock()
        mock_svc = MagicMock()
        mock_svc.preprocess = AsyncMock(side_effect=AlreadyProcessedError("already done"))
        mock_svc_cls.return_value = mock_svc

        from app.celery_app.tasks.preprocess_tasks import preprocess_document

        result = preprocess_document.run(
            document_id="doc-done",
            knowledge_base_id="kb-1",
            name="test.html",
            embedding_model_id="emb",
            bucket="bucket",
        )
        assert result["chunk_count"] == 0


# ---------------------------------------------------------------------------
# finalize_document — reads DB, sets Succeed or Failed
# ---------------------------------------------------------------------------

class TestFinalizeDocument:

    @patch("app.celery_app.tasks.preprocess_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.container")
    def test_marks_succeed_when_all_chunks_done(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls
    ):
        mock_container.session_factory = MagicMock()

        chunk_repo = MagicMock()
        chunk_repo.count_non_succeeded = AsyncMock(return_value=0)
        mock_chunk_repo_cls.return_value = chunk_repo

        doc_repo = MagicMock()
        doc_repo.set_status = AsyncMock()
        mock_doc_repo_cls.return_value = doc_repo

        from app.celery_app.tasks.preprocess_tasks import finalize_document

        finalize_document.run(document_id="doc-1", correlation_id="corr-1")

        doc_repo.set_status.assert_awaited_with("doc-1", "Succeed")

    @patch("app.celery_app.tasks.preprocess_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.preprocess_tasks.container")
    def test_marks_failed_when_chunks_not_all_done(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls
    ):
        mock_container.session_factory = MagicMock()

        chunk_repo = MagicMock()
        chunk_repo.count_non_succeeded = AsyncMock(return_value=2)
        mock_chunk_repo_cls.return_value = chunk_repo

        doc_repo = MagicMock()
        doc_repo.set_status = AsyncMock()
        mock_doc_repo_cls.return_value = doc_repo

        from app.celery_app.tasks.preprocess_tasks import finalize_document

        finalize_document.run(document_id="doc-1")

        doc_repo.set_status.assert_awaited_with("doc-1", "Failed")
