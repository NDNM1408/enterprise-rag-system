"""Unit tests for KnowledgeBaseService."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import AsyncMock, MagicMock
from app.application.services.knowledge_base_service import KnowledgeBaseService
from app.application.dtos.requests.knowledge_base_request import CreateKnowledgeBaseRequest
from app.exceptions import ValidationError, ResourceNotFoundError
from app.infrastructure.connectors.postgres.schema import KnowledgeBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_service(mock_kb_repo=None, mock_doc_repo=None):
    kb_repo = mock_kb_repo or MagicMock()
    doc_repo = mock_doc_repo or MagicMock()
    return KnowledgeBaseService(kb_repo, doc_repo)


def make_kb(**kwargs) -> KnowledgeBase:
    kb = MagicMock(spec=KnowledgeBase)
    kb.id = kwargs.get("id", "kb-001")
    kb.name = kwargs.get("name", "Test KB")
    kb.embed_id = kwargs.get("embed_id", "embed-model-v1")
    kb.to_dict = MagicMock(return_value={
        "id": kb.id, "name": kb.name, "embed_id": kb.embed_id
    })
    return kb


# ---------------------------------------------------------------------------
# paging
# ---------------------------------------------------------------------------

class TestPaging:
    @pytest.mark.asyncio
    async def test_paging_returns_expected_structure(self):
        kb = make_kb()
        repo = MagicMock()
        repo.paging = AsyncMock(return_value=[kb])
        repo.count = AsyncMock(return_value=1)

        svc = make_service(mock_kb_repo=repo)
        result = await svc.paging(page=1, page_size=10)

        assert result["total"] == 1
        assert result["page"] == 1
        assert result["page_size"] == 10
        assert len(result["items"]) == 1

    @pytest.mark.asyncio
    async def test_paging_calculates_skip_correctly(self):
        repo = MagicMock()
        repo.paging = AsyncMock(return_value=[])
        repo.count = AsyncMock(return_value=0)

        svc = make_service(mock_kb_repo=repo)
        await svc.paging(page=3, page_size=10)

        repo.paging.assert_awaited_once_with(
            skip=20, limit=10, where=None, order_by=None
        )

    @pytest.mark.asyncio
    async def test_paging_passes_filters(self):
        repo = MagicMock()
        repo.paging = AsyncMock(return_value=[])
        repo.count = AsyncMock(return_value=0)

        svc = make_service(mock_kb_repo=repo)
        await svc.paging(page=1, page_size=5, where={"name": "test"}, order_by={"create_time": "desc"})

        repo.paging.assert_awaited_once_with(
            skip=0, limit=5, where={"name": "test"}, order_by={"create_time": "desc"}
        )


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

class TestCreate:
    @pytest.mark.asyncio
    async def test_create_success(self):
        kb = make_kb()
        repo = MagicMock()
        repo.create = AsyncMock(return_value=kb)

        svc = make_service(mock_kb_repo=repo)
        request = MagicMock(spec=CreateKnowledgeBaseRequest)
        request.embed_id = "model-v1"
        request.model_dump = MagicMock(return_value={"name": "My KB", "embed_id": "model-v1"})

        result = await svc.create(request)
        assert result is kb

    @pytest.mark.asyncio
    async def test_create_raises_when_embed_id_missing(self):
        svc = make_service()
        request = MagicMock(spec=CreateKnowledgeBaseRequest)
        request.embed_id = None

        with pytest.raises(ValidationError, match="embed_id is required"):
            await svc.create(request)

    @pytest.mark.asyncio
    async def test_create_raises_when_embed_id_empty_string(self):
        svc = make_service()
        request = MagicMock(spec=CreateKnowledgeBaseRequest)
        request.embed_id = ""

        with pytest.raises(ValidationError):
            await svc.create(request)


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_knowledge_base(self):
        kb = make_kb()
        repo = MagicMock()
        repo.get = AsyncMock(return_value=kb)

        svc = make_service(mock_kb_repo=repo)
        result = await svc.get("kb-001")

        assert result is kb
        repo.get.assert_awaited_once_with(id="kb-001")

    @pytest.mark.asyncio
    async def test_get_propagates_not_found(self):
        repo = MagicMock()
        repo.get = AsyncMock(side_effect=ResourceNotFoundError("KnowledgeBase", "missing-id"))

        svc = make_service(mock_kb_repo=repo)
        with pytest.raises(ResourceNotFoundError):
            await svc.get("missing-id")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_calls_repo(self):
        repo = MagicMock()
        repo.delete = AsyncMock()

        svc = make_service(mock_kb_repo=repo)
        await svc.delete("kb-001")

        repo.delete.assert_awaited_once_with(id="kb-001")

    @pytest.mark.asyncio
    async def test_delete_propagates_not_found(self):
        repo = MagicMock()
        repo.delete = AsyncMock(side_effect=ResourceNotFoundError("KnowledgeBase", "x"))

        svc = make_service(mock_kb_repo=repo)
        with pytest.raises(ResourceNotFoundError):
            await svc.delete("x")
