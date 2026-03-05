"""
Unit tests for graph ingestion tasks.

Architecture under test:
  - graph_preprocess_document: fetches/parses/splits/stores chunks, then
    dispatches a sequential Celery chain of graph_ingest_chunk tasks.
  - graph_ingest_chunk: calls GraphIngestor.ingest(), marks Succeed.
    Retried independently on failure.
  - finalize_graph_document: sets document status based on chunk outcomes.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc_repo(doc_status="Created"):
    repo = MagicMock()
    repo.get_status = AsyncMock(return_value=doc_status)
    repo.set_status = AsyncMock()
    return repo


def _make_chunk_repo(existing_chunks=None, chunk_status="Processing", non_succeeded=0):
    repo = MagicMock()
    repo.get_by_document = AsyncMock(return_value=existing_chunks or [])
    repo.get_status = AsyncMock(return_value=chunk_status)
    repo.set_status = AsyncMock()
    repo.batch_insert = AsyncMock()
    repo.count_non_succeeded = AsyncMock(return_value=non_succeeded)
    return repo


# ---------------------------------------------------------------------------
# graph_preprocess_document
# ---------------------------------------------------------------------------

class TestGraphPreprocessDocument:

    @patch("app.celery_app.tasks.graph_tasks.chain")
    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_happy_path_dispatches_chain(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls, mock_chain
    ):
        """Happy path: parses document, inserts chunks, dispatches chain."""
        mock_container.session_factory = MagicMock()
        mock_container.s3.get_file = AsyncMock(return_value=b"<html>content</html>")
        mock_container.parser.parse.return_value = "Hello world text"
        mock_container.splitter.split.return_value = [
            {"content": "chunk one", "tokens": 10, "chunk_order_index": 0},
            {"content": "chunk two", "tokens": 10, "chunk_order_index": 1},
        ]

        mock_doc_repo_cls.return_value = _make_doc_repo(doc_status="Created")
        mock_chunk_repo_cls.return_value = _make_chunk_repo(existing_chunks=[])

        mock_chain_instance = MagicMock()
        mock_chain.return_value = mock_chain_instance

        from app.celery_app.tasks.graph_tasks import graph_preprocess_document

        result = graph_preprocess_document.run(
            document_id="doc-1",
            knowledge_base_id="kb-1",
            name="test.html",
            bucket="my-bucket",
            correlation_id="corr-1",
        )

        assert result["document_id"] == "doc-1"
        assert result["chunk_count"] == 2

        # Chain must be built and dispatched
        mock_chain.assert_called_once()
        mock_chain_instance.apply_async.assert_called_once()

    @patch("app.celery_app.tasks.graph_tasks.chain")
    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_skips_already_succeed_document(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls, mock_chain
    ):
        """Document already Succeed -> no chain dispatched, chunk_count=0."""
        mock_container.session_factory = MagicMock()
        mock_doc_repo_cls.return_value = _make_doc_repo(doc_status="Succeed")
        mock_chunk_repo_cls.return_value = _make_chunk_repo()

        from app.celery_app.tasks.graph_tasks import graph_preprocess_document

        result = graph_preprocess_document.run(
            document_id="doc-done",
            knowledge_base_id="kb-1",
            name="test.html",
            bucket="bucket",
        )

        assert result["chunk_count"] == 0
        mock_chain.assert_not_called()

    @patch("app.celery_app.tasks.graph_tasks.chain")
    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_skips_document_not_found(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls, mock_chain
    ):
        """Document not found -> no chain dispatched, chunk_count=0."""
        mock_container.session_factory = MagicMock()
        mock_doc_repo_cls.return_value = _make_doc_repo(doc_status=None)
        mock_chunk_repo_cls.return_value = _make_chunk_repo()

        from app.celery_app.tasks.graph_tasks import graph_preprocess_document

        result = graph_preprocess_document.run(
            document_id="doc-missing",
            knowledge_base_id="kb-1",
            name="test.html",
            bucket="bucket",
        )

        assert result["chunk_count"] == 0
        mock_chain.assert_not_called()

    @patch("app.celery_app.tasks.graph_tasks.chain")
    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_resumes_from_existing_chunks_dispatches_only_pending(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls, mock_chain
    ):
        """
        On retry with existing chunk records: only non-Succeed chunks are
        included in the chain; Succeed chunks are filtered out.
        """
        mock_container.session_factory = MagicMock()

        existing_chunks = [
            {"id": "chunk-A", "content": "text A", "status": "Succeed"},
            {"id": "chunk-B", "content": "text B", "status": "Processing"},
        ]
        mock_doc_repo_cls.return_value = _make_doc_repo(doc_status="Processing")
        mock_chunk_repo_cls.return_value = _make_chunk_repo(existing_chunks=existing_chunks)

        mock_chain_instance = MagicMock()
        mock_chain.return_value = mock_chain_instance

        from app.celery_app.tasks.graph_tasks import graph_preprocess_document

        result = graph_preprocess_document.run(
            document_id="doc-retry",
            knowledge_base_id="kb-1",
            name="test.html",
            bucket="bucket",
        )

        # Only 1 pending chunk (chunk-B); chunk-A is already Succeed
        assert result["chunk_count"] == 1
        mock_chain.assert_called_once()
        mock_chain_instance.apply_async.assert_called_once()

    @patch("app.celery_app.tasks.graph_tasks.chain")
    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_all_existing_chunks_succeed_finalizes_inline(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls, mock_chain
    ):
        """All existing chunks already Succeed -> finalizes inline, no chain."""
        mock_container.session_factory = MagicMock()

        existing_chunks = [
            {"id": "chunk-A", "content": "text A", "status": "Succeed"},
            {"id": "chunk-B", "content": "text B", "status": "Succeed"},
        ]
        doc_repo = _make_doc_repo(doc_status="Processing")
        mock_doc_repo_cls.return_value = doc_repo
        mock_chunk_repo_cls.return_value = _make_chunk_repo(
            existing_chunks=existing_chunks, non_succeeded=0
        )

        from app.celery_app.tasks.graph_tasks import graph_preprocess_document

        result = graph_preprocess_document.run(
            document_id="doc-all-done",
            knowledge_base_id="kb-1",
            name="test.html",
            bucket="bucket",
        )

        assert result["chunk_count"] == 0
        mock_chain.assert_not_called()
        doc_repo.set_status.assert_awaited_with("doc-all-done", "Succeed")

    @patch("app.celery_app.tasks.graph_tasks.chain")
    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_marks_failed_on_unsupported_file(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls, mock_chain
    ):
        """Unsupported file type -> document marked Failed, no chain."""
        from app.application.core.parser import UnsupportedFileTypeError

        mock_container.session_factory = MagicMock()
        mock_container.s3.get_file = AsyncMock(return_value=b"data")
        mock_container.parser.parse.side_effect = UnsupportedFileTypeError("exe not supported")

        doc_repo = _make_doc_repo(doc_status="Created")
        mock_doc_repo_cls.return_value = doc_repo
        mock_chunk_repo_cls.return_value = _make_chunk_repo(existing_chunks=[])

        from app.celery_app.tasks.graph_tasks import graph_preprocess_document

        result = graph_preprocess_document.run(
            document_id="doc-bad",
            knowledge_base_id="kb-1",
            name="virus.exe",
            bucket="bucket",
        )

        assert result["chunk_count"] == 0
        mock_chain.assert_not_called()
        doc_repo.set_status.assert_awaited_with("doc-bad", "Failed")

    @patch("app.celery_app.tasks.graph_tasks.chain")
    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_marks_failed_on_empty_text(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls, mock_chain
    ):
        """Empty parsed text -> document marked Failed, no chain."""
        mock_container.session_factory = MagicMock()
        mock_container.s3.get_file = AsyncMock(return_value=b"<html></html>")
        mock_container.parser.parse.return_value = ""

        doc_repo = _make_doc_repo(doc_status="Created")
        mock_doc_repo_cls.return_value = doc_repo
        mock_chunk_repo_cls.return_value = _make_chunk_repo(existing_chunks=[])

        from app.celery_app.tasks.graph_tasks import graph_preprocess_document

        result = graph_preprocess_document.run(
            document_id="doc-empty",
            knowledge_base_id="kb-1",
            name="test.html",
            bucket="bucket",
        )

        assert result["chunk_count"] == 0
        mock_chain.assert_not_called()
        doc_repo.set_status.assert_awaited_with("doc-empty", "Failed")


# ---------------------------------------------------------------------------
# graph_ingest_chunk
# ---------------------------------------------------------------------------

class TestGraphIngestChunk:

    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_happy_path(self, mock_container, mock_chunk_repo_cls):
        """Happy path: calls graph_ingestor.ingest, marks chunk Succeed."""
        mock_container.session_factory = MagicMock()
        mock_chunk_repo_cls.return_value = _make_chunk_repo(chunk_status="Processing")

        mock_container.graph_ingestor.ingest = AsyncMock(
            return_value={"entity_count": 3, "relation_count": 2}
        )

        from app.celery_app.tasks.graph_tasks import graph_ingest_chunk

        result = graph_ingest_chunk.run(
            chunk_id="chunk-1",
            content="Hello world text",
            knowledge_base_id="kb-1",
            document_id="doc-1",
            correlation_id="corr-1",
        )

        assert result == {"chunk_id": "chunk-1", "status": "success"}
        mock_container.graph_ingestor.ingest.assert_awaited_once_with(
            content="Hello world text",
            kb_id="kb-1",
            file_path="doc-1",
            chunk_key="chunk-1",
        )
        mock_chunk_repo_cls.return_value.set_status.assert_awaited_with("chunk-1", "Succeed")

    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_skips_already_succeed_chunk(self, mock_container, mock_chunk_repo_cls):
        """Chunk already Succeed -> returns skipped, ingestor never called."""
        mock_container.session_factory = MagicMock()
        mock_chunk_repo_cls.return_value = _make_chunk_repo(chunk_status="Succeed")

        mock_container.graph_ingestor.ingest = AsyncMock()

        from app.celery_app.tasks.graph_tasks import graph_ingest_chunk

        result = graph_ingest_chunk.run(
            chunk_id="chunk-done",
            content="text",
            knowledge_base_id="kb-1",
            document_id="doc-1",
        )

        assert result == {"chunk_id": "chunk-done", "status": "skipped"}
        mock_container.graph_ingestor.ingest.assert_not_awaited()

    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_marks_chunk_failed_and_reraises_on_ingest_error(
        self, mock_container, mock_chunk_repo_cls
    ):
        """ingest failure -> chunk marked Failed, exception re-raised for retry."""
        mock_container.session_factory = MagicMock()
        mock_chunk_repo_cls.return_value = _make_chunk_repo(chunk_status="Processing")

        mock_container.graph_ingestor.ingest = AsyncMock(
            side_effect=RuntimeError("LLM timeout")
        )

        from app.celery_app.tasks.graph_tasks import graph_ingest_chunk

        with pytest.raises(Exception):
            graph_ingest_chunk.run(
                chunk_id="chunk-fail",
                content="text",
                knowledge_base_id="kb-1",
                document_id="doc-1",
            )

        mock_chunk_repo_cls.return_value.set_status.assert_awaited_with("chunk-fail", "Failed")

    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_passes_correct_kb_id_to_ingestor(self, mock_container, mock_chunk_repo_cls):
        """GraphIngestor.ingest is called with the correct kb_id."""
        mock_container.session_factory = MagicMock()
        mock_chunk_repo_cls.return_value = _make_chunk_repo(chunk_status="Processing")
        mock_container.graph_ingestor.ingest = AsyncMock(
            return_value={"entity_count": 0, "relation_count": 0}
        )

        from app.celery_app.tasks.graph_tasks import graph_ingest_chunk

        graph_ingest_chunk.run(
            chunk_id="chunk-1",
            content="text",
            knowledge_base_id="my-kb-id",
            document_id="doc-1",
        )

        mock_container.graph_ingestor.ingest.assert_awaited_once_with(
            content="text",
            kb_id="my-kb-id",
            file_path="doc-1",
            chunk_key="chunk-1",
        )


# ---------------------------------------------------------------------------
# finalize_graph_document
# ---------------------------------------------------------------------------

class TestFinalizeGraphDocument:

    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_marks_succeed_when_all_chunks_done(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls
    ):
        """All chunks Succeed -> document marked Succeed."""
        mock_container.session_factory = MagicMock()

        chunk_repo = _make_chunk_repo(non_succeeded=0)
        mock_chunk_repo_cls.return_value = chunk_repo

        doc_repo = _make_doc_repo()
        mock_doc_repo_cls.return_value = doc_repo

        from app.celery_app.tasks.graph_tasks import finalize_graph_document

        result = finalize_graph_document.run(document_id="doc-1", correlation_id="corr-1")

        assert result["final_status"] == "Succeed"
        assert result["non_succeeded"] == 0
        doc_repo.set_status.assert_awaited_with("doc-1", "Succeed")

    @patch("app.celery_app.tasks.graph_tasks.DocumentRepository")
    @patch("app.celery_app.tasks.graph_tasks.ChunkRepository")
    @patch("app.celery_app.tasks.graph_tasks.container")
    def test_marks_failed_when_chunks_pending(
        self, mock_container, mock_chunk_repo_cls, mock_doc_repo_cls
    ):
        """Some chunks not Succeed -> document marked Failed."""
        mock_container.session_factory = MagicMock()

        chunk_repo = _make_chunk_repo(non_succeeded=2)
        mock_chunk_repo_cls.return_value = chunk_repo

        doc_repo = _make_doc_repo()
        mock_doc_repo_cls.return_value = doc_repo

        from app.celery_app.tasks.graph_tasks import finalize_graph_document

        result = finalize_graph_document.run(document_id="doc-1")

        assert result["final_status"] == "Failed"
        assert result["non_succeeded"] == 2
        doc_repo.set_status.assert_awaited_with("doc-1", "Failed")
