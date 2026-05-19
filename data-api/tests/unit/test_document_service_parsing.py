"""Unit tests for the document-parsing orchestrator paths in DocumentsService.

Covers:
  • Native text formats (.md, .html, .txt) → skip the document-parsing service
    and dispatch preprocess_document directly.
  • Non-native formats (.pdf, .docx) → submit a parse job via HTTP and store
    the returned parsing_job_id on the document row.
  • handle_parse_callback for running / done / failed states.
"""

import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fastapi import UploadFile

from app.application.services.document_service import (
    DocumentsService,
    _extension,
    _is_native_text,
    NATIVE_TEXT_EXTENSIONS,
)
from app.infrastructure.connectors.postgres.schema import (
    Document,
    KnowledgeBase,
    ParsingStatus,
    IngestingStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_upload(filename: str, content: bytes = b"x") -> UploadFile:
    f = MagicMock(spec=UploadFile)
    f.filename = filename
    f.read = AsyncMock(return_value=content)
    return f


def _make_service(*, conflict_names=None, conflict_etags=None):
    kb = MagicMock(spec=KnowledgeBase)
    kb.id = "kb-1"
    kb.embed_id = "embed-1"
    kb.parser_config = {"rag_mode": "classic"}

    kb_repo = MagicMock()
    kb_repo.get = AsyncMock(return_value=kb)

    doc_repo = MagicMock()
    doc_repo.find_conflicts = AsyncMock(return_value=conflict_names or [])
    doc_repo.find_etag_conflicts = AsyncMock(return_value=conflict_etags or [])
    doc_repo.bulk_create = AsyncMock()
    doc_repo.update_fields = AsyncMock()
    doc_repo.get_by_parsing_job_id = AsyncMock()

    s3 = MagicMock()
    s3.upload_file = AsyncMock()

    svc = DocumentsService(
        knowledge_base_repository=kb_repo,
        document_repository=doc_repo,
        s3_client_service=s3,
        chunk_repository=MagicMock(),
    )
    return svc, kb, kb_repo, doc_repo, s3


# ---------------------------------------------------------------------------
# Extension helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("a.md", True),
    ("a.MD", True),
    ("doc.markdown", True),
    ("page.html", True),
    ("page.HTM", True),
    ("notes.txt", True),
    ("report.pdf", False),
    ("file.docx", False),
    ("noext", False),
])
def test_is_native_text(name, expected):
    assert _is_native_text(name) is expected


def test_native_extensions_set_contains_all_documented():
    assert ".md" in NATIVE_TEXT_EXTENSIONS
    assert ".html" in NATIVE_TEXT_EXTENSIONS
    assert ".htm" in NATIVE_TEXT_EXTENSIONS
    assert ".txt" in NATIVE_TEXT_EXTENSIONS


# ---------------------------------------------------------------------------
# add_documents — native (skipped) path
# ---------------------------------------------------------------------------

class TestAddDocumentsNativePath:

    @pytest.mark.asyncio
    async def test_native_html_skips_parse_job_and_dispatches_preprocess(self):
        svc, _, _, doc_repo, _ = _make_service()
        files = [_make_upload("note.html")]

        with patch(
            "app.application.services.document_service.send_preprocess_task"
        ) as mock_preprocess, patch.object(
            svc, "_submit_parse_job", new=AsyncMock()
        ) as mock_submit:
            await svc.add_documents("kb-1", files, None)

        # Native text → preprocess called directly, no parse job.
        mock_preprocess.assert_called_once()
        mock_submit.assert_not_called()
        # parsed_markdown_s3_url omitted: worker will fall back to local parser.
        assert mock_preprocess.call_args.kwargs.get("parsed_markdown_s3_url") is None

    @pytest.mark.asyncio
    async def test_native_document_row_has_skipped_parsing_status(self):
        svc, _, _, doc_repo, _ = _make_service()
        files = [_make_upload("notes.md")]

        with patch("app.application.services.document_service.send_preprocess_task"):
            await svc.add_documents("kb-1", files, None)

        created = doc_repo.bulk_create.call_args.args[0]
        assert len(created) == 1
        d = created[0]
        assert d.parsing_status == ParsingStatus.Skipped.value
        assert d.parsing_progress == 100
        assert d.ingesting_status == IngestingStatus.Pending.value


# ---------------------------------------------------------------------------
# add_documents — non-native (parse) path
# ---------------------------------------------------------------------------

class TestAddDocumentsParsePath:

    @pytest.mark.asyncio
    async def test_pdf_submits_parse_job_and_stores_job_id(self):
        svc, _, _, doc_repo, _ = _make_service()
        files = [_make_upload("report.pdf")]

        with patch(
            "app.application.services.document_service.send_preprocess_task"
        ) as mock_preprocess, patch.object(
            svc, "_submit_parse_job", new=AsyncMock(return_value="job-abc")
        ) as mock_submit:
            await svc.add_documents("kb-1", files, None)

        mock_submit.assert_awaited_once()
        # preprocess is NOT dispatched yet — that happens in handle_parse_callback.
        mock_preprocess.assert_not_called()
        # The returned job id must land on the document row.
        doc_repo.update_fields.assert_awaited_once()
        update_args = doc_repo.update_fields.await_args.args
        assert update_args[1] == {"parsing_job_id": "job-abc"}

    @pytest.mark.asyncio
    async def test_pdf_document_row_has_pending_parsing_status(self):
        svc, _, _, doc_repo, _ = _make_service()
        files = [_make_upload("report.pdf")]

        with patch.object(svc, "_submit_parse_job", new=AsyncMock(return_value="j")):
            await svc.add_documents("kb-1", files, None)

        d = doc_repo.bulk_create.call_args.args[0][0]
        assert d.parsing_status == ParsingStatus.Pending.value
        assert d.parsing_progress == 0

    @pytest.mark.asyncio
    async def test_parse_submission_failure_marks_document_failed(self):
        svc, _, _, doc_repo, _ = _make_service()
        files = [_make_upload("report.pdf")]

        with patch.object(
            svc, "_submit_parse_job",
            new=AsyncMock(side_effect=RuntimeError("doc-parsing unreachable")),
        ):
            await svc.add_documents("kb-1", files, None)

        # update_fields called with the failure payload.
        doc_repo.update_fields.assert_awaited_once()
        update_kwargs = doc_repo.update_fields.await_args.args[1]
        assert update_kwargs["parsing_status"] == ParsingStatus.Failed.value
        assert update_kwargs["status"] == "Failed"
        assert "submit failed" in update_kwargs["parsing_error"]

    @pytest.mark.asyncio
    async def test_mixed_batch_routes_each_file_correctly(self):
        svc, _, _, doc_repo, _ = _make_service()
        files = [_make_upload("a.md", b"a"), _make_upload("b.pdf", b"b")]

        with patch(
            "app.application.services.document_service.send_preprocess_task"
        ) as mock_preprocess, patch.object(
            svc, "_submit_parse_job", new=AsyncMock(return_value="job-1")
        ) as mock_submit:
            await svc.add_documents("kb-1", files, None)

        # The native .md → preprocess; the .pdf → submit_parse_job.
        assert mock_preprocess.call_count == 1
        assert mock_submit.await_count == 1


# ---------------------------------------------------------------------------
# handle_parse_callback
# ---------------------------------------------------------------------------

class TestHandleParseCallback:

    def _doc(self, doc_id="doc-1", kb_id="kb-1", name="r.pdf"):
        d = MagicMock(spec=Document)
        d.id = doc_id
        d.kb_id = kb_id
        d.name = name
        return d

    @pytest.mark.asyncio
    async def test_running_updates_progress_and_phase(self):
        svc, _, _, doc_repo, _ = _make_service()
        doc_repo.get_by_parsing_job_id = AsyncMock(return_value=self._doc())

        await svc.handle_parse_callback({
            "job_id": "job-1",
            "state": "running",
            "pages_done": 3,
            "pages_total": 10,
        })

        doc_repo.update_fields.assert_awaited_once()
        fields = doc_repo.update_fields.await_args.args[1]
        assert fields["parsing_status"] == ParsingStatus.Parsing.value
        assert fields["parsing_progress"] == 30
        assert fields["status"] == "Processing"

    @pytest.mark.asyncio
    async def test_done_dispatches_preprocess_with_markdown_url(self):
        svc, kb, kb_repo, doc_repo, _ = _make_service()
        doc_repo.get_by_parsing_job_id = AsyncMock(return_value=self._doc())

        with patch(
            "app.application.services.document_service.send_preprocess_task"
        ) as mock_preprocess:
            await svc.handle_parse_callback({
                "job_id": "job-1",
                "state": "done",
                "pages_done": 5,
                "pages_total": 5,
                "s3_markdown_url": "s3://document-parsing/job-1/result.md",
            })

        mock_preprocess.assert_called_once()
        kwargs = mock_preprocess.call_args.kwargs
        assert kwargs["parsed_markdown_s3_url"] == "s3://document-parsing/job-1/result.md"
        assert kwargs["document_id"] == "doc-1"

        fields = doc_repo.update_fields.await_args.args[1]
        assert fields["parsing_status"] == ParsingStatus.Parsed.value
        assert fields["parsing_progress"] == 100
        assert fields["parsed_markdown_s3_key"] == "s3://document-parsing/job-1/result.md"

    @pytest.mark.asyncio
    async def test_failed_marks_document_failed(self):
        svc, _, _, doc_repo, _ = _make_service()
        doc_repo.get_by_parsing_job_id = AsyncMock(return_value=self._doc())

        with patch(
            "app.application.services.document_service.send_preprocess_task"
        ) as mock_preprocess:
            await svc.handle_parse_callback({
                "job_id": "job-1",
                "state": "failed",
                "error": "MinerU OOM",
            })

        mock_preprocess.assert_not_called()
        fields = doc_repo.update_fields.await_args.args[1]
        assert fields["parsing_status"] == ParsingStatus.Failed.value
        assert fields["status"] == "Failed"
        assert "MinerU OOM" in fields["parsing_error"]

    @pytest.mark.asyncio
    async def test_callback_with_unknown_job_id_is_silently_ignored(self):
        svc, _, _, doc_repo, _ = _make_service()
        doc_repo.get_by_parsing_job_id = AsyncMock(return_value=None)

        with patch(
            "app.application.services.document_service.send_preprocess_task"
        ) as mock_preprocess:
            await svc.handle_parse_callback({
                "job_id": "ghost",
                "state": "done",
                "s3_markdown_url": "s3://x/y.md",
            })

        mock_preprocess.assert_not_called()
        doc_repo.update_fields.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_job_id_is_silently_ignored(self):
        svc, _, _, doc_repo, _ = _make_service()
        await svc.handle_parse_callback({"state": "running"})
        doc_repo.get_by_parsing_job_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_pages_total_progress_stays_zero(self):
        """pages_total=0 must not divide-by-zero — progress stays 0."""
        svc, _, _, doc_repo, _ = _make_service()
        doc_repo.get_by_parsing_job_id = AsyncMock(return_value=self._doc())

        await svc.handle_parse_callback({
            "job_id": "job-1",
            "state": "running",
            "pages_done": 0,
            "pages_total": 0,
        })

        fields = doc_repo.update_fields.await_args.args[1]
        assert fields["parsing_progress"] == 0
