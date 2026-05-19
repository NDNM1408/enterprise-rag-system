"""Unit tests for the document-parsing service's orchestrator-facing helpers.

These tests cover:
  • ``infrastructure.s3.parse_s3_url`` — splits full s3:// URLs.
  • ``celery_app.tasks.parse_tasks._post_callback`` — best-effort HTTP POST
    that must NEVER raise (callback failures can't sink the parse task).

The MinerU pipeline and Celery worker are not exercised here.
"""
from unittest.mock import patch, MagicMock

import pytest

# Both modules under test transitively pull in service-runtime deps (boto3
# for S3, psycopg2 for the SQLAlchemy engine that ParsingJob lives behind).
# Skip cleanly when running outside the document-parsing image rather than
# crashing with a misleading ImportError.
pytest.importorskip("boto3", reason="document-parsing helpers depend on boto3")
pytest.importorskip("psycopg2", reason="document-parsing engine import requires psycopg2")


# ---------------------------------------------------------------------------
# infrastructure.s3.parse_s3_url
# ---------------------------------------------------------------------------

class TestParseS3Url:

    def test_parses_bucket_and_key(self):
        from infrastructure.s3 import parse_s3_url
        bucket, key = parse_s3_url("s3://my-bucket/some/path/file.md")
        assert bucket == "my-bucket"
        assert key == "some/path/file.md"

    def test_supports_nested_keys(self):
        from infrastructure.s3 import parse_s3_url
        bucket, key = parse_s3_url("s3://b/a/b/c/d/e.bin")
        assert bucket == "b"
        assert key == "a/b/c/d/e.bin"

    def test_rejects_non_s3_scheme(self):
        from infrastructure.s3 import parse_s3_url
        with pytest.raises(ValueError):
            parse_s3_url("file:///tmp/x")

    def test_rejects_missing_key(self):
        from infrastructure.s3 import parse_s3_url
        with pytest.raises(ValueError):
            parse_s3_url("s3://only-bucket")


# ---------------------------------------------------------------------------
# parse_tasks._post_callback
# ---------------------------------------------------------------------------

class TestPostCallback:

    def test_no_url_is_no_op(self):
        from celery_app.tasks.parse_tasks import _post_callback
        # Must not raise, must not attempt any urlopen.
        with patch("celery_app.tasks.parse_tasks.urllib.request.urlopen") as mock_open:
            _post_callback(None, {"state": "done"})
        mock_open.assert_not_called()

    def test_posts_json_body(self):
        from celery_app.tasks.parse_tasks import _post_callback
        captured = {}

        class _FakeResp:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def _fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = req.data
            captured["ctype"] = req.get_header("Content-type")
            return _FakeResp()

        with patch("celery_app.tasks.parse_tasks.urllib.request.urlopen", side_effect=_fake_urlopen):
            _post_callback(
                "http://data-api:8000/api/v1/internal/parse-callback",
                {"job_id": "j", "state": "done"},
            )

        assert captured["url"].endswith("/internal/parse-callback")
        assert b"\"state\": \"done\"" in captured["body"]
        assert captured["ctype"] == "application/json"

    def test_swallows_network_errors(self):
        """Callback failures must not propagate — they're best-effort."""
        from celery_app.tasks.parse_tasks import _post_callback
        import urllib.error

        with patch(
            "celery_app.tasks.parse_tasks.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            # Returns None; raises nothing.
            assert _post_callback("http://nope/", {"x": 1}) is None

    def test_swallows_unexpected_exceptions(self):
        from celery_app.tasks.parse_tasks import _post_callback

        with patch(
            "celery_app.tasks.parse_tasks.urllib.request.urlopen",
            side_effect=RuntimeError("boom"),
        ):
            assert _post_callback("http://x/", {"a": "b"}) is None


# ---------------------------------------------------------------------------
# parse_tasks._s3_url_for
# ---------------------------------------------------------------------------

class TestS3UrlBuilder:

    def test_passes_through_full_url(self):
        from celery_app.tasks.parse_tasks import _s3_url_for
        assert _s3_url_for("s3://b/k") == "s3://b/k"

    def test_resolves_relative_key_against_default_bucket(self):
        from celery_app.tasks.parse_tasks import _s3_url_for
        from settings import settings
        url = _s3_url_for("job-1/result.md")
        assert url == f"s3://{settings.s3_bucket}/job-1/result.md"
