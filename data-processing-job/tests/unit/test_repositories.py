"""
Unit tests for DocumentRepository and ChunkRepository.

Uses AsyncMock to simulate the async_sessionmaker context-manager protocol
without a real database.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_factory(rows=None, scalar_value=None):
    """
    Build a mock async_sessionmaker that returns a fake AsyncSession.

    rows: list of MagicMock rows with ._mapping attributes (for mappings()).
    scalar_value: int returned by result.scalar().
    """
    mock_result = MagicMock()
    if rows is not None:
        mock_result.one_or_none.return_value = rows[0] if rows else None
        mock_result.mappings.return_value = rows
        mock_result.scalar.return_value = scalar_value
        mock_result.__iter__ = lambda self: iter(rows)
    else:
        mock_result.one_or_none.return_value = None
        mock_result.scalar.return_value = scalar_value or 0

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    # Context manager protocol for the session itself (async with session_factory() as session:)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    # session.begin() must be a SYNC callable returning an async context manager.
    # Using AsyncMock here makes begin() return a coroutine, not a CM.
    mock_begin = MagicMock()
    mock_begin.__aenter__ = AsyncMock(return_value=None)
    mock_begin.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin = MagicMock(return_value=mock_begin)

    mock_factory = MagicMock()
    mock_factory.return_value = mock_session
    return mock_factory, mock_session, mock_result


# ---------------------------------------------------------------------------
# DocumentRepository
# ---------------------------------------------------------------------------

class TestDocumentRepository:

    @pytest.mark.asyncio
    async def test_get_status_returns_status_string(self):
        row = MagicMock()
        row.__getitem__ = lambda self, i: "Processing"
        factory, session, _ = _make_session_factory(rows=[row])

        from app.infrastructure.repositories.document_repository import DocumentRepository
        repo = DocumentRepository(factory)
        result = await repo.get_status("doc-1")

        assert result == "Processing"

    @pytest.mark.asyncio
    async def test_get_status_returns_none_when_not_found(self):
        factory, session, result_mock = _make_session_factory(rows=[])
        result_mock.one_or_none.return_value = None

        from app.infrastructure.repositories.document_repository import DocumentRepository
        repo = DocumentRepository(factory)
        result = await repo.get_status("missing-doc")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_status_executes_update(self):
        factory, session, _ = _make_session_factory()

        from app.infrastructure.repositories.document_repository import DocumentRepository
        repo = DocumentRepository(factory)
        await repo.set_status("doc-1", "Succeed")

        session.execute.assert_awaited_once()
        sql, params = session.execute.call_args.args
        assert "UPDATE document" in str(sql)
        assert params["status"] == "Succeed"
        assert params["id"] == "doc-1"


# ---------------------------------------------------------------------------
# ChunkRepository
# ---------------------------------------------------------------------------

class TestChunkRepository:

    @pytest.mark.asyncio
    async def test_get_status_returns_status(self):
        row = MagicMock()
        row.__getitem__ = lambda self, i: "Succeed"
        factory, session, _ = _make_session_factory(rows=[row])

        from app.infrastructure.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(factory)
        result = await repo.get_status("chunk-1")

        assert result == "Succeed"

    @pytest.mark.asyncio
    async def test_get_status_none_when_not_found(self):
        factory, session, result_mock = _make_session_factory(rows=[])
        result_mock.one_or_none.return_value = None

        from app.infrastructure.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(factory)
        result = await repo.get_status("missing")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_status_executes_update(self):
        factory, session, _ = _make_session_factory()

        from app.infrastructure.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(factory)
        await repo.set_status("chunk-1", "Failed")

        session.execute.assert_awaited_once()
        sql, params = session.execute.call_args.args
        assert "UPDATE chunk" in str(sql)
        assert params["status"] == "Failed"
        assert params["id"] == "chunk-1"

    @pytest.mark.asyncio
    async def test_count_non_succeeded_returns_count(self):
        factory, session, result_mock = _make_session_factory(scalar_value=3)

        from app.infrastructure.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(factory)
        count = await repo.count_non_succeeded("doc-1")

        assert count == 3

    @pytest.mark.asyncio
    async def test_count_non_succeeded_returns_zero_default(self):
        factory, session, result_mock = _make_session_factory(scalar_value=None)
        result_mock.scalar.return_value = None

        from app.infrastructure.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(factory)
        count = await repo.count_non_succeeded("doc-1")

        assert count == 0

    @pytest.mark.asyncio
    async def test_batch_insert_no_op_for_empty_list(self):
        factory, session, _ = _make_session_factory()

        from app.infrastructure.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(factory)
        await repo.batch_insert([])

        session.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_insert_executes_insert(self):
        factory, session, _ = _make_session_factory()

        from app.infrastructure.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(factory)
        await repo.batch_insert(
            [
                {
                    "id": "c1",
                    "content": "hello",
                    "document_id": "d1",
                    "kb_id": "k1",
                    "doc_name": "test.html",
                    "status": "Processing",
                    "chunk_s3_url": None,
                }
            ]
        )

        session.execute.assert_awaited_once()
        sql, params = session.execute.call_args.args
        assert "INSERT INTO chunk" in str(sql)
        assert params["id_0"] == "c1"
        assert params["content_0"] == "hello"

    @pytest.mark.asyncio
    async def test_get_by_document_returns_rows(self):
        row1 = MagicMock()
        row1._mapping = {"id": "c1", "content": "text", "status": "Processing"}
        row2 = MagicMock()
        row2._mapping = {"id": "c2", "content": "more", "status": "Succeed"}

        factory, session, result_mock = _make_session_factory(rows=[row1, row2])
        result_mock.__iter__ = lambda self: iter([row1, row2])

        from app.infrastructure.repositories.chunk_repository import ChunkRepository
        repo = ChunkRepository(factory)
        chunks = await repo.get_by_document("doc-1")

        assert len(chunks) == 2
        assert chunks[0]["id"] == "c1"
        assert chunks[1]["status"] == "Succeed"
