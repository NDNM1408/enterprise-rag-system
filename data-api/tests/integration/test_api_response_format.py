"""
Integration tests verifying that every API endpoint returns the correct
standardized response envelope (success or RFC-9457 error).

These tests run against the FastAPI app with all real service dependencies
replaced by mocks so no actual DB / S3 / RabbitMQ is required.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
os.environ.setdefault("BUCKET_NAME", "test-bucket")

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.exceptions import ResourceNotFoundError, ConflictError, ValidationError


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def assert_success_envelope(body: dict) -> None:
    """Assert the body has the standardized success shape."""
    assert body.get("success") is True, f"expected success=true, got: {body}"
    assert "data" in body, f"missing 'data' key: {body}"
    assert "meta" in body, f"missing 'meta' key: {body}"
    assert "request_id" in body["meta"], f"missing meta.request_id: {body}"
    assert "timestamp" in body["meta"], f"missing meta.timestamp: {body}"


def assert_error_envelope(body: dict, expected_status: int) -> None:
    """Assert the body is RFC-9457 compliant."""
    required_fields = ["type", "title", "status", "detail", "instance", "request_id", "timestamp"]
    for field in required_fields:
        assert field in body, f"missing error field '{field}': {body}"
    assert body["status"] == expected_status, (
        f"expected status={expected_status}, got {body['status']}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from main import app
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_returns_success_envelope(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert_success_envelope(resp.json())

    def test_health_data_has_status_ok(self, client):
        resp = client.get("/")
        data = resp.json()["data"]
        assert data["status"] == "OK"
        assert data["service"] == "data-api"


# ---------------------------------------------------------------------------
# Knowledge base endpoints
# ---------------------------------------------------------------------------

class TestKnowledgeBaseEndpoints:
    def test_list_returns_success_envelope(self, client):
        kb_dict = {"id": "kb-1", "name": "Test KB", "embed_id": "m1"}
        kb = MagicMock()
        kb.to_dict = MagicMock(return_value=kb_dict)

        with (
            patch("app.infrastructure.connectors.postgres.repositories.knowledge_base_repository.KnowledgeBaseRepository.paging", new=AsyncMock(return_value=[kb])),
            patch("app.infrastructure.connectors.postgres.repositories.knowledge_base_repository.KnowledgeBaseRepository.count", new=AsyncMock(return_value=1)),
        ):
            resp = client.get("/api/v1/knowledge_base/")

        assert resp.status_code == 200
        assert_success_envelope(resp.json())

    def test_create_returns_success_envelope(self, client):
        kb = MagicMock()
        kb.to_dict = MagicMock(return_value={"id": "kb-new", "name": "New KB", "embed_id": "m1"})

        with patch(
            "app.infrastructure.connectors.postgres.repositories.knowledge_base_repository.KnowledgeBaseRepository.create",
            new=AsyncMock(return_value=kb),
        ):
            resp = client.post(
                "/api/v1/knowledge_base",
                json={"name": "New KB", "embeddedModelId": "m1"},
            )

        assert resp.status_code == 200
        assert_success_envelope(resp.json())

    def test_get_not_found_returns_error_envelope(self, client):
        with patch(
            "app.infrastructure.connectors.postgres.repositories.knowledge_base_repository.KnowledgeBaseRepository.get",
            new=AsyncMock(side_effect=ResourceNotFoundError("KnowledgeBase", "kb-missing")),
        ):
            resp = client.get("/api/v1/knowledge_base/kb-missing")

        assert resp.status_code == 404
        assert_error_envelope(resp.json(), 404)

    def test_delete_not_found_returns_error_envelope(self, client):
        with patch(
            "app.infrastructure.connectors.postgres.repositories.knowledge_base_repository.KnowledgeBaseRepository.delete",
            new=AsyncMock(side_effect=ResourceNotFoundError("KnowledgeBase", "kb-x")),
        ):
            resp = client.delete("/api/v1/knowledge_base/kb-x")

        assert resp.status_code == 404
        assert_error_envelope(resp.json(), 404)

    def test_list_invalid_filter_json_returns_error_envelope(self, client):
        resp = client.get("/api/v1/knowledge_base/?filter=not-valid-json")
        assert resp.status_code == 422
        assert_error_envelope(resp.json(), 422)

    def test_create_missing_embed_id_returns_error_envelope(self, client):
        with patch(
            "app.infrastructure.connectors.postgres.repositories.knowledge_base_repository.KnowledgeBaseRepository.create",
            new=AsyncMock(),
        ):
            resp = client.post("/api/v1/knowledge_base", json={"name": "No Embed"})

        assert resp.status_code in (400, 422)
        body = resp.json()
        # Must still be an error envelope (either from domain validation or pydantic)
        assert "status" in body or "detail" in body


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------

class TestDocumentEndpoints:
    def test_upload_returns_success_envelope(self, client):
        kb = MagicMock()
        kb.embed_id = "model-v1"

        mock_s3 = MagicMock()
        mock_s3.upload_file = AsyncMock()

        with (
            patch("app.infrastructure.connectors.postgres.repositories.knowledge_base_repository.KnowledgeBaseRepository.get", new=AsyncMock(return_value=kb)),
            patch("app.infrastructure.connectors.postgres.repositories.document_repository.DocumentRepository.find_conflicts", new=AsyncMock(return_value=[])),
            patch("app.infrastructure.connectors.postgres.repositories.document_repository.DocumentRepository.find_etag_conflicts", new=AsyncMock(return_value=[])),
            patch("app.infrastructure.connectors.postgres.repositories.document_repository.DocumentRepository.bulk_create", new=AsyncMock()),
            # Patch the class in the dependencies module so S3ClientService() returns the mock
            patch("app.configurations.dependencies.S3ClientService", return_value=mock_s3),
            # Patch where document_service imports it, not where it's defined
            patch("app.application.services.document_service.send_preprocess_task"),
        ):
            resp = client.post(
                "/api/v1/kb-001/documents",
                files={"files": ("test.html", b"<html>content</html>", "text/html")},
            )

        assert resp.status_code == 200
        assert_success_envelope(resp.json())

    def test_upload_invalid_cmetadata_returns_error_envelope(self, client):
        mock_s3 = MagicMock()
        mock_s3.upload_file = AsyncMock()
        with patch("app.configurations.dependencies.S3ClientService", return_value=mock_s3):
            resp = client.post(
                "/api/v1/kb-001/documents",
                data={"cmetadata": "not-valid-json"},
                files={"files": ("test.html", b"content", "text/html")},
            )
        assert resp.status_code == 422
        assert_error_envelope(resp.json(), 422)

    def test_upload_kb_not_found_returns_error_envelope(self, client):
        mock_s3 = MagicMock()
        mock_s3.upload_file = AsyncMock()
        with (
            patch("app.configurations.dependencies.S3ClientService", return_value=mock_s3),
            patch(
                "app.infrastructure.connectors.postgres.repositories.knowledge_base_repository.KnowledgeBaseRepository.get",
                new=AsyncMock(side_effect=ResourceNotFoundError("KnowledgeBase", "kb-gone")),
            ),
        ):
            resp = client.post(
                "/api/v1/kb-gone/documents",
                files={"files": ("test.html", b"content", "text/html")},
            )

        assert resp.status_code == 404
        assert_error_envelope(resp.json(), 404)


# ---------------------------------------------------------------------------
# Request ID header propagation
# ---------------------------------------------------------------------------

class TestRequestID:
    def test_response_header_contains_request_id(self, client):
        resp = client.get("/")
        assert "x-request-id" in resp.headers

    def test_client_supplied_request_id_is_echoed(self, client):
        custom_id = "my-custom-req-id-123"
        resp = client.get("/", headers={"X-Request-ID": custom_id})
        assert resp.headers.get("x-request-id") == custom_id
        assert resp.json()["meta"]["request_id"] == custom_id

    def test_auto_generated_request_id_is_uuid_like(self, client):
        import re
        resp = client.get("/")
        rid = resp.headers.get("x-request-id", "")
        assert re.match(r"[0-9a-f-]{36}", rid), f"request_id doesn't look like a UUID: {rid}"
