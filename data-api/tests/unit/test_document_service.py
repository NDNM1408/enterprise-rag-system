"""Unit tests for DocumentsService."""

import hashlib
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile

from app.application.services.document_service import DocumentsService
from app.exceptions import ConflictError, ResourceNotFoundError, ExternalServiceError
from app.infrastructure.connectors.postgres.schema import KnowledgeBase, Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_upload_file(filename: str, content: bytes = b"file content") -> UploadFile:
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.read = AsyncMock(return_value=content)
    return file


def make_kb(kb_id: str = "kb-001", embed_id: str = "model-v1") -> MagicMock:
    kb = MagicMock(spec=KnowledgeBase)
    kb.id = kb_id
    kb.embed_id = embed_id
    return kb


def make_doc_repo(conflict_names=None, conflict_etags=None) -> MagicMock:
    repo = MagicMock()
    repo.find_conflicts = AsyncMock(return_value=conflict_names or [])
    repo.find_etag_conflicts = AsyncMock(return_value=conflict_etags or [])
    repo.bulk_create = AsyncMock()
    return repo


def make_service(
    kb_repo=None, doc_repo=None, s3=None
) -> DocumentsService:
    return DocumentsService(
        knowledge_base_repository=kb_repo or MagicMock(),
        document_repository=doc_repo or make_doc_repo(),
        s3_client_service=s3 or MagicMock(),
        chunk_repository=MagicMock(),
    )


# ---------------------------------------------------------------------------
# add_documents — happy path
# ---------------------------------------------------------------------------

class TestAddDocuments:
    @pytest.mark.asyncio
    async def test_add_documents_success(self):
        kb = make_kb()
        kb_repo = MagicMock()
        kb_repo.get = AsyncMock(return_value=kb)

        doc_repo = make_doc_repo()

        s3 = MagicMock()
        s3.upload_file = AsyncMock()

        svc = make_service(kb_repo, doc_repo, s3)
        files = [make_upload_file("doc1.html"), make_upload_file("doc2.html")]

        with patch("app.application.services.document_service.send_preprocess_task") as mock_task:
            await svc.add_documents(kb_id="kb-001", files=files, cmetadata=None)

        doc_repo.bulk_create.assert_awaited_once()
        assert s3.upload_file.await_count == 2
        assert mock_task.call_count == 2

    @pytest.mark.asyncio
    async def test_add_documents_stores_s3_url_and_etag(self):
        """Document records must be created with s3_url and etag set."""
        kb = make_kb()
        kb_repo = MagicMock()
        kb_repo.get = AsyncMock(return_value=kb)

        doc_repo = make_doc_repo()

        s3 = MagicMock()
        s3.upload_file = AsyncMock()

        svc = make_service(kb_repo, doc_repo, s3)
        content = b"<html>hello</html>"
        files = [make_upload_file("doc.html", content=content)]
        expected_etag = hashlib.md5(content).hexdigest()

        with patch("app.application.services.document_service.send_preprocess_task"):
            await svc.add_documents(kb_id="kb-001", files=files, cmetadata=None)

        created_docs = doc_repo.bulk_create.call_args[0][0]
        assert len(created_docs) == 1
        doc = created_docs[0]
        assert doc.etag == expected_etag
        assert doc.s3_url == f"s3://test-bucket/kb-001/doc.html"

    # ---------------------------------------------------------------------------
    # Conflict detection
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_add_documents_raises_when_kb_not_found(self):
        kb_repo = MagicMock()
        kb_repo.get = AsyncMock(side_effect=ResourceNotFoundError("KnowledgeBase", "kb-missing"))

        svc = make_service(kb_repo=kb_repo)

        with pytest.raises(ResourceNotFoundError):
            await svc.add_documents("kb-missing", [make_upload_file("a.html")], None)

    @pytest.mark.asyncio
    async def test_raises_on_filename_conflict(self):
        kb_repo = MagicMock()
        kb_repo.get = AsyncMock(return_value=make_kb())

        doc_repo = make_doc_repo(conflict_names=["existing.html"])
        svc = make_service(kb_repo=kb_repo, doc_repo=doc_repo)

        with pytest.raises(ConflictError) as exc_info:
            await svc.add_documents("kb-001", [make_upload_file("existing.html")], None)

        assert "conflicting_filenames" in exc_info.value.details
        assert "existing.html" in exc_info.value.details["conflicting_filenames"]

    @pytest.mark.asyncio
    async def test_raises_on_etag_conflict(self):
        """Uploading a file with identical content to an existing document raises ConflictError."""
        kb_repo = MagicMock()
        kb_repo.get = AsyncMock(return_value=make_kb())

        content = b"duplicate file content"
        duplicate_etag = hashlib.md5(content).hexdigest()
        doc_repo = make_doc_repo(conflict_etags=[duplicate_etag])
        svc = make_service(kb_repo=kb_repo, doc_repo=doc_repo)

        with pytest.raises(ConflictError) as exc_info:
            await svc.add_documents("kb-001", [make_upload_file("new_name.html", content=content)], None)

        assert "conflicting_etags" in exc_info.value.details
        assert duplicate_etag in exc_info.value.details["conflicting_etags"]

    @pytest.mark.asyncio
    async def test_raises_on_both_name_and_etag_conflict(self):
        """When both name and etag conflict, both are reported in the error details."""
        kb_repo = MagicMock()
        kb_repo.get = AsyncMock(return_value=make_kb())

        content = b"some content"
        etag = hashlib.md5(content).hexdigest()
        doc_repo = make_doc_repo(conflict_names=["dup.html"], conflict_etags=[etag])
        svc = make_service(kb_repo=kb_repo, doc_repo=doc_repo)

        with pytest.raises(ConflictError) as exc_info:
            await svc.add_documents("kb-001", [make_upload_file("dup.html", content=content)], None)

        details = exc_info.value.details
        assert "conflicting_filenames" in details
        assert "conflicting_etags" in details

    @pytest.mark.asyncio
    async def test_s3_not_called_when_conflict_detected(self):
        """S3 upload must not happen if duplicates are found."""
        kb_repo = MagicMock()
        kb_repo.get = AsyncMock(return_value=make_kb())

        doc_repo = make_doc_repo(conflict_names=["dup.html"])
        s3 = MagicMock()
        s3.upload_file = AsyncMock()
        svc = make_service(kb_repo=kb_repo, doc_repo=doc_repo, s3=s3)

        with pytest.raises(ConflictError):
            await svc.add_documents("kb-001", [make_upload_file("dup.html")], None)

        s3.upload_file.assert_not_awaited()

    # ---------------------------------------------------------------------------
    # S3 failure
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_raises_on_s3_failure(self):
        kb_repo = MagicMock()
        kb_repo.get = AsyncMock(return_value=make_kb())

        doc_repo = make_doc_repo()
        s3 = MagicMock()
        s3.upload_file = AsyncMock(side_effect=RuntimeError("S3 connection refused"))

        svc = make_service(kb_repo=kb_repo, doc_repo=doc_repo, s3=s3)

        with pytest.raises(ExternalServiceError) as exc_info:
            await svc.add_documents("kb-001", [make_upload_file("test.html")], None)

        assert exc_info.value.service_name == "S3"


# ---------------------------------------------------------------------------
# delete_documents
# ---------------------------------------------------------------------------

class TestDeleteDocuments:
    @pytest.mark.asyncio
    async def test_delete_documents_success(self):
        doc = MagicMock(spec=Document)
        doc.kb_id = "kb-001"
        doc.name = "doc1.html"

        doc_repo = MagicMock()
        doc_repo.get_by_ids = AsyncMock(return_value=[doc])
        doc_repo.bulk_delete = AsyncMock()

        s3 = MagicMock()
        s3.delete_file = AsyncMock()

        svc = make_service(doc_repo=doc_repo, s3=s3)
        await svc.delete_documents(["doc-id-1"])

        s3.delete_file.assert_awaited_once()
        doc_repo.bulk_delete.assert_awaited_once_with(["doc-id-1"])

    @pytest.mark.asyncio
    async def test_delete_documents_raises_when_not_found(self):
        doc_repo = MagicMock()
        doc_repo.get_by_ids = AsyncMock(return_value=[])

        svc = make_service(doc_repo=doc_repo)

        with pytest.raises(ResourceNotFoundError):
            await svc.delete_documents(["nonexistent-id"])
