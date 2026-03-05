"""
Unit tests for DocumentPreprocessService.

All external I/O (S3, DB repositories, parser, splitter) is mocked so
these tests run without real infrastructure.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.application.services.document_preprocess_service import (
    DocumentPreprocessService,
    AlreadyProcessedError,
    DocumentNotFoundError,
    ChunkRecord,
)
from app.application.core.parser import UnsupportedFileTypeError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(
    *,
    doc_status="Created",
    s3_content=b"<html><body>hello</body></html>",
    parsed_text="hello world",
    chunks=None,
):
    """Build a DocumentPreprocessService with all deps mocked."""
    if chunks is None:
        chunks = [
            {"content": "chunk one", "tokens": 5, "chunk_order_index": 0},
            {"content": "chunk two", "tokens": 5, "chunk_order_index": 1},
        ]

    s3 = MagicMock()
    s3.get_file = AsyncMock(return_value=s3_content)
    s3.upload_file = AsyncMock()

    doc_repo = MagicMock()
    doc_repo.get_status = AsyncMock(return_value=doc_status)
    doc_repo.set_status = AsyncMock()

    chunk_repo = MagicMock()
    chunk_repo.batch_insert = AsyncMock()

    parser = MagicMock()
    parser.parse.return_value = parsed_text

    splitter = MagicMock()
    splitter.split.return_value = chunks

    svc = DocumentPreprocessService(
        s3=s3,
        doc_repo=doc_repo,
        chunk_repo=chunk_repo,
        parser=parser,
        splitter=splitter,
    )
    return svc, s3, doc_repo, chunk_repo, parser, splitter


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestPreprocessHappyPath:

    @pytest.mark.asyncio
    async def test_returns_chunk_records(self):
        svc, *_ = _make_service()
        records = await svc.preprocess(
            document_id="doc-1",
            kb_id="kb-1",
            name="test.html",
            bucket="my-bucket",
        )
        assert len(records) == 2
        assert all(isinstance(r, ChunkRecord) for r in records)

    @pytest.mark.asyncio
    async def test_chunk_records_have_correct_fields(self):
        svc, *_ = _make_service()
        records = await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")
        assert records[0].document_id == "doc-1"
        assert records[0].kb_id == "kb-1"
        assert records[0].doc_name == "test.html"
        assert records[0].status == "Processing"

    @pytest.mark.asyncio
    async def test_sets_document_status_to_processing(self):
        svc, _, doc_repo, *_ = _make_service()
        await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")
        doc_repo.set_status.assert_awaited_with("doc-1", "Processing")

    @pytest.mark.asyncio
    async def test_batch_inserts_chunks(self):
        svc, _, _, chunk_repo, *_ = _make_service()
        records = await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")
        chunk_repo.batch_insert.assert_awaited_once()
        inserted = chunk_repo.batch_insert.call_args.args[0]
        assert len(inserted) == len(records)

    @pytest.mark.asyncio
    async def test_uploads_chunks_to_s3_when_upload_chunks_true(self):
        svc, s3, *_ = _make_service()
        records = await svc.preprocess(
            "doc-1", "kb-1", "test.html", "src-bucket",
            upload_chunks=True, chunk_bucket="chunk-bucket",
        )
        assert s3.upload_file.await_count == len(records)

    @pytest.mark.asyncio
    async def test_no_s3_upload_when_upload_chunks_false(self):
        svc, s3, *_ = _make_service()
        await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")
        s3.upload_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chunk_s3_url_set_when_uploading(self):
        svc, *_ = _make_service()
        records = await svc.preprocess(
            "doc-1", "kb-1", "test.html", "bucket",
            upload_chunks=True, chunk_bucket="chunks",
        )
        assert all(r.chunk_s3_url is not None for r in records)
        assert all(r.chunk_s3_url.startswith("s3://chunks/") for r in records)

    @pytest.mark.asyncio
    async def test_chunk_s3_url_none_when_not_uploading(self):
        svc, *_ = _make_service()
        records = await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")
        assert all(r.chunk_s3_url is None for r in records)

    @pytest.mark.asyncio
    async def test_metadata_contains_order_and_tokens(self):
        svc, *_ = _make_service()
        records = await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")
        assert records[0].metadata["chunk_order_index"] == 0
        assert records[1].metadata["chunk_order_index"] == 1
        assert "tokens" in records[0].metadata


# ---------------------------------------------------------------------------
# Idempotency / error path tests
# ---------------------------------------------------------------------------

class TestPreprocessIdempotency:

    @pytest.mark.asyncio
    async def test_raises_document_not_found_when_status_none(self):
        svc, *_ = _make_service(doc_status=None)
        with pytest.raises(DocumentNotFoundError):
            await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")

    @pytest.mark.asyncio
    async def test_raises_already_processed_when_processing(self):
        svc, *_ = _make_service(doc_status="Processing")
        with pytest.raises(AlreadyProcessedError):
            await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")

    @pytest.mark.asyncio
    async def test_raises_already_processed_when_succeed(self):
        svc, *_ = _make_service(doc_status="Succeed")
        with pytest.raises(AlreadyProcessedError):
            await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")

    @pytest.mark.asyncio
    async def test_propagates_unsupported_file_type_error(self):
        svc, _, doc_repo, _, parser, _ = _make_service()
        parser.parse.side_effect = UnsupportedFileTypeError("exe not supported")
        with pytest.raises(UnsupportedFileTypeError):
            await svc.preprocess("doc-1", "kb-1", "file.exe", "bucket")

    @pytest.mark.asyncio
    async def test_marks_failed_on_empty_text(self):
        svc, _, doc_repo, *_ = _make_service(parsed_text="")
        records = await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")
        assert records == []
        doc_repo.set_status.assert_awaited_with("doc-1", "Failed")

    @pytest.mark.asyncio
    async def test_marks_failed_when_no_chunks_produced(self):
        svc, _, doc_repo, *_ = _make_service(chunks=[])
        records = await svc.preprocess("doc-1", "kb-1", "test.html", "bucket")
        assert records == []
        doc_repo.set_status.assert_awaited_with("doc-1", "Failed")
