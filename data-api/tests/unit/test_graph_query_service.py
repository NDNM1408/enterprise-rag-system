"""
Unit tests for GraphRAGQueryService.

All Neo4j / Postgres / LLM calls are mocked via GraphQueryEngine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


def _make_kb(rag_mode="graphrag"):
    kb = MagicMock()
    kb.parser_config = {"rag_mode": rag_mode}
    return kb


class TestGraphRAGQueryService:

    @pytest.mark.asyncio
    async def test_graph_search_returns_answer(self):
        """Happy path — returns answer from GraphQueryEngine."""
        from app.application.services.graph_query_service import GraphRAGQueryService

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=_make_kb(rag_mode="graphrag"))

        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value={"answer": "The answer is 42."})

        service = GraphRAGQueryService(mock_repo, mock_engine)
        result = await service.graph_search(
            kb_id="kb-1",
            query_text="What is the meaning of life?",
            mode="hybrid",
            only_context=False,
        )

        assert result["answer"] == "The answer is 42."
        assert result["kb_id"] == "kb-1"
        assert result["mode"] == "hybrid"

        mock_engine.query.assert_called_once_with(
            kb_id="kb-1",
            query_text="What is the meaning of life?",
            mode="hybrid",
            top_k=40,
            only_context=False,
        )

    @pytest.mark.asyncio
    async def test_graph_search_only_context(self):
        """only_context=True must return 'context' key, not 'answer'."""
        from app.application.services.graph_query_service import GraphRAGQueryService

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=_make_kb(rag_mode="graphrag"))

        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(return_value={"context": "context snippet"})

        service = GraphRAGQueryService(mock_repo, mock_engine)
        result = await service.graph_search(
            kb_id="kb-1",
            query_text="Who founded the company?",
            only_context=True,
        )

        assert "context" in result
        assert "answer" not in result

    @pytest.mark.asyncio
    async def test_raises_validation_error_for_classic_kb(self):
        """KB in classic mode must raise ValidationError."""
        from app.application.services.graph_query_service import GraphRAGQueryService
        from app.exceptions import ValidationError

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=_make_kb(rag_mode="classic"))

        mock_engine = MagicMock()

        service = GraphRAGQueryService(mock_repo, mock_engine)

        with pytest.raises(ValidationError, match="not in graphrag mode"):
            await service.graph_search(kb_id="kb-classic", query_text="hello")

    @pytest.mark.asyncio
    async def test_raises_validation_error_for_empty_query(self):
        """Empty query text must raise ValidationError before any DB call."""
        from app.application.services.graph_query_service import GraphRAGQueryService
        from app.exceptions import ValidationError

        mock_repo = MagicMock()
        mock_engine = MagicMock()
        service = GraphRAGQueryService(mock_repo, mock_engine)

        with pytest.raises(ValidationError, match="cannot be empty"):
            await service.graph_search(kb_id="kb-1", query_text="   ")

        mock_repo.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_database_error_on_engine_failure(self):
        """Engine failure must be wrapped in DatabaseError."""
        from app.application.services.graph_query_service import GraphRAGQueryService
        from app.exceptions import DatabaseError

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=_make_kb(rag_mode="graphrag"))

        mock_engine = MagicMock()
        mock_engine.query = AsyncMock(side_effect=RuntimeError("neo4j down"))

        service = GraphRAGQueryService(mock_repo, mock_engine)

        with pytest.raises(DatabaseError):
            await service.graph_search(kb_id="kb-1", query_text="test")


class TestDocumentServiceRagModeRouting:
    """Verify that document_service routes to correct task based on rag_mode."""

    @pytest.mark.asyncio
    async def test_classic_kb_sends_preprocess_task(self):
        """KB with rag_mode=classic (default) must dispatch preprocess_document."""
        from app.application.services.document_service import DocumentsService
        from app.infrastructure.connectors.postgres.schema import KnowledgeBase
        from unittest.mock import patch

        kb = MagicMock(spec=KnowledgeBase)
        kb.embed_id = "emb-model-id"
        kb.parser_config = None  # default → classic

        mock_kb_repo = MagicMock()
        mock_kb_repo.get = AsyncMock(return_value=kb)

        mock_doc_repo = MagicMock()
        mock_doc_repo.find_conflicts = AsyncMock(return_value=[])
        mock_doc_repo.find_etag_conflicts = AsyncMock(return_value=[])
        mock_doc_repo.bulk_create = AsyncMock()

        mock_s3 = MagicMock()
        mock_s3.upload_file = AsyncMock()

        mock_chunk_repo = MagicMock()

        upload_file = MagicMock()
        upload_file.filename = "doc.html"
        upload_file.read = AsyncMock(return_value=b"<html>content</html>")

        with patch(
            "app.application.services.document_service.send_preprocess_task"
        ) as mock_preprocess, patch(
            "app.application.services.document_service.send_graph_preprocess_task"
        ) as mock_graph:
            service = DocumentsService(mock_kb_repo, mock_doc_repo, mock_s3, mock_chunk_repo)
            await service.add_documents(kb_id="kb-1", files=[upload_file], cmetadata=None)

        mock_preprocess.assert_called_once()
        mock_graph.assert_not_called()

    @pytest.mark.asyncio
    async def test_graphrag_kb_sends_graph_ingest_task(self):
        """KB with rag_mode=graphrag must dispatch graph_ingest_document."""
        from app.application.services.document_service import DocumentsService
        from app.infrastructure.connectors.postgres.schema import KnowledgeBase
        from unittest.mock import patch

        kb = MagicMock(spec=KnowledgeBase)
        kb.embed_id = "emb-model-id"
        kb.parser_config = {"rag_mode": "graphrag"}

        mock_kb_repo = MagicMock()
        mock_kb_repo.get = AsyncMock(return_value=kb)

        mock_doc_repo = MagicMock()
        mock_doc_repo.find_conflicts = AsyncMock(return_value=[])
        mock_doc_repo.find_etag_conflicts = AsyncMock(return_value=[])
        mock_doc_repo.bulk_create = AsyncMock()

        mock_s3 = MagicMock()
        mock_s3.upload_file = AsyncMock()

        mock_chunk_repo = MagicMock()

        upload_file = MagicMock()
        upload_file.filename = "report.html"
        upload_file.read = AsyncMock(return_value=b"<html>report</html>")

        with patch(
            "app.application.services.document_service.send_preprocess_task"
        ) as mock_preprocess, patch(
            "app.application.services.document_service.send_graph_preprocess_task"
        ) as mock_graph:
            service = DocumentsService(mock_kb_repo, mock_doc_repo, mock_s3, mock_chunk_repo)
            await service.add_documents(kb_id="kb-lg", files=[upload_file], cmetadata=None)

        mock_graph.assert_called_once()
        mock_preprocess.assert_not_called()
