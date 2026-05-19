"""Unit tests for the pre-parsed markdown path in DocumentPreprocessService.

When ``parsed_markdown_s3_url`` is supplied the worker must:
  • Fetch markdown from the full s3:// URL via S3.get_txt_by_url.
  • Skip its local DocumentParser entirely (parser.parse must not be called).
  • Feed the markdown straight to the splitter.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.application.services.document_preprocess_service import (
    DocumentPreprocessService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Row:
    """Lightweight stand-in for whatever the splitter produces.

    The service iterates rows and reads ``content``, ``parent_text``, ``id``,
    ``heading_path``, ``tokens``, ``chunk_order_index``.
    """

    def __init__(self, idx: int, content: str = "x", parent_text: str = "Root section body"):
        self.id = f"chunk-{idx}"
        self.content = content
        self.parent_text = parent_text
        self.heading_path = "Root"
        self.tokens = 5
        self.chunk_order_index = idx


def _make_service(
    *,
    parsed_md: str = "# Heading\n\ncontent",
    chunks=None,
    doc_status: str = "Created",
):
    if chunks is None:
        chunks = [_Row(0), _Row(1)]

    s3 = MagicMock()
    s3.get_file = AsyncMock()                 # Should NOT be called on this path.
    s3.get_txt_by_url = AsyncMock(return_value=parsed_md)
    s3.upload_file = AsyncMock()

    doc_repo = MagicMock()
    doc_repo.get_status = AsyncMock(return_value=doc_status)
    doc_repo.set_status = AsyncMock()

    chunk_repo = MagicMock()
    chunk_repo.batch_insert = AsyncMock()

    parser = MagicMock()
    parser.parse = MagicMock(return_value="should-not-be-used")

    splitter = MagicMock()
    splitter.split = MagicMock(return_value=chunks)

    svc = DocumentPreprocessService(
        s3=s3, doc_repo=doc_repo, chunk_repo=chunk_repo,
        parser=parser, splitter=splitter,
    )
    return svc, s3, doc_repo, chunk_repo, parser, splitter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParsedMarkdownPath:

    @pytest.mark.asyncio
    async def test_fetches_markdown_by_url_when_supplied(self):
        svc, s3, *_ = _make_service()
        await svc.preprocess(
            document_id="doc-1",
            kb_id="kb-1",
            name="report.pdf",
            bucket="ignored",
            parsed_markdown_s3_url="s3://document-parsing/job-x/result.md",
        )
        s3.get_txt_by_url.assert_awaited_once_with(
            "s3://document-parsing/job-x/result.md"
        )

    @pytest.mark.asyncio
    async def test_local_parser_not_invoked(self):
        svc, _, _, _, parser, _ = _make_service()
        await svc.preprocess(
            "doc-1", "kb-1", "report.pdf", "ignored",
            parsed_markdown_s3_url="s3://b/k.md",
        )
        parser.parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_raw_s3_get_file_not_invoked(self):
        svc, s3, *_ = _make_service()
        await svc.preprocess(
            "doc-1", "kb-1", "report.pdf", "ignored",
            parsed_markdown_s3_url="s3://b/k.md",
        )
        s3.get_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_splitter_receives_markdown_content(self):
        svc, _, _, _, _, splitter = _make_service(
            parsed_md="# T\n\nbody",
        )
        await svc.preprocess(
            "doc-1", "kb-1", "report.pdf", "ignored",
            parsed_markdown_s3_url="s3://b/k.md",
        )
        splitter.split.assert_called_once_with("# T\n\nbody")

    @pytest.mark.asyncio
    async def test_default_path_still_uses_local_parser(self):
        """Sanity: omitting parsed_markdown_s3_url keeps the legacy flow."""
        svc, s3, _, _, parser, _ = _make_service()
        s3.get_file = AsyncMock(return_value=b"<html/>")
        await svc.preprocess(
            "doc-1", "kb-1", "page.html", "src-bucket",
        )
        s3.get_file.assert_awaited_once()
        parser.parse.assert_called_once()
        s3.get_txt_by_url.assert_not_awaited()


class TestS3UrlParsing:
    """Sanity checks on the S3ClientService URL helper."""

    def test_parses_valid_url(self):
        from app.infrastructure.clients.s3_client_service import S3ClientService
        bucket, key = S3ClientService.parse_s3_url("s3://my-bucket/path/to/file.md")
        assert bucket == "my-bucket"
        assert key == "path/to/file.md"

    def test_rejects_non_s3_scheme(self):
        from app.infrastructure.clients.s3_client_service import S3ClientService
        with pytest.raises(ValueError):
            S3ClientService.parse_s3_url("https://example.com/file")

    def test_rejects_missing_key(self):
        from app.infrastructure.clients.s3_client_service import S3ClientService
        with pytest.raises(ValueError):
            S3ClientService.parse_s3_url("s3://bucket-only")
