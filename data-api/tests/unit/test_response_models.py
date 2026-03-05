"""Tests for unified response models."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from app.application.dtos.responses.success_response import create_success_response
from app.application.dtos.responses.error_response import create_error_response, ErrorResponse


class TestSuccessResponse:
    def test_create_success_response_structure(self):
        result = create_success_response(data={"id": "abc"}, request_id="req-123")

        assert result["success"] is True
        assert result["data"] == {"id": "abc"}
        assert "meta" in result
        assert result["meta"]["request_id"] == "req-123"
        assert "timestamp" in result["meta"]

    def test_success_response_with_list_data(self):
        result = create_success_response(data=[1, 2, 3], request_id="req-456")

        assert result["success"] is True
        assert result["data"] == [1, 2, 3]

    def test_success_response_with_none_data(self):
        result = create_success_response(data=None, request_id="req-789")

        assert result["success"] is True
        assert result["data"] is None

    def test_meta_contains_required_fields(self):
        result = create_success_response(data={}, request_id="req-000")

        meta = result["meta"]
        assert "request_id" in meta
        assert "timestamp" in meta

    def test_timestamp_is_iso_format(self):
        from datetime import datetime
        result = create_success_response(data={}, request_id="req-ts")
        ts = result["meta"]["timestamp"]
        # Should be parseable as ISO datetime
        datetime.fromisoformat(ts)


class TestErrorResponse:
    def test_create_error_response_structure(self):
        err = create_error_response(
            status=404,
            title="Not Found",
            detail="Resource missing",
            request_id="req-err",
            instance="/api/v1/resource/123",
        )

        assert isinstance(err, ErrorResponse)
        assert err.status == 404
        assert err.title == "Not Found"
        assert err.detail == "Resource missing"
        assert err.request_id == "req-err"
        assert err.instance == "/api/v1/resource/123"
        assert "not-found" in err.type

    def test_error_response_default_type_mapping(self):
        cases = [
            (400, "bad-request"),
            (401, "unauthorized"),
            (403, "forbidden"),
            (404, "not-found"),
            (409, "conflict"),
            (422, "validation-error"),
            (500, "internal-server-error"),
        ]
        for status_code, expected_slug in cases:
            err = create_error_response(
                status=status_code,
                title="Test",
                detail="Test detail",
                request_id="r",
                instance="/",
            )
            assert expected_slug in err.type, (
                f"Expected '{expected_slug}' in type for status {status_code}, got '{err.type}'"
            )

    def test_error_response_with_custom_type(self):
        err = create_error_response(
            status=400,
            title="Custom",
            detail="Custom error",
            request_id="r",
            instance="/",
            error_type="https://example.com/errors/custom",
        )
        assert err.type == "https://example.com/errors/custom"

    def test_error_response_with_extra_errors(self):
        err = create_error_response(
            status=422,
            title="Validation Error",
            detail="Invalid fields",
            request_id="r",
            instance="/",
            errors={"field": "is required"},
        )
        assert err.errors == {"field": "is required"}

    def test_error_response_serializable(self):
        err = create_error_response(
            status=500,
            title="Server Error",
            detail="Something went wrong",
            request_id="req-ser",
            instance="/api/v1/test",
        )
        data = err.model_dump()
        assert data["status"] == 500
        assert data["title"] == "Server Error"
        assert "timestamp" in data
