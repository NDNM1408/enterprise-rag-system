"""Parse-document Celery task.

The worker:
  1. Loads the ``parsing.parsing_job`` row.
  2. Downloads ``s3_input_key`` to a temp file.
  3. Runs the registered parser. PDF parsers stream progress back to the
     row via ``JobRepo.update_progress``.
  4. Uploads ``result.md`` and each extracted image to S3 under
     ``<job_id>/result.md`` and ``<job_id>/images/<name>``.
  5. Marks the job done with parser/mode/duration metadata.

Failures: the row is marked ``failed`` and the exception re-raised so Celery
routes the message to ``dlq.parse`` (after the configured retries).
"""
from __future__ import annotations

import base64
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import urllib.request
import urllib.error
import json as _json

from celery_app import celery_app
from core.registry import for_extension
from core.timing import timed
from infrastructure import s3
from infrastructure.repositories.job_repo import JobRepo
from settings import settings

log = logging.getLogger(__name__)


# Throttle: emit at most one running-callback every N seconds even if the
# parser streams page-level progress more frequently. Avoids hammering the
# orchestrator on long documents.
_RUNNING_CALLBACK_MIN_INTERVAL_S = 1.5


def _post_callback(url: Optional[str], body: dict[str, Any]) -> None:
    """Best-effort POST to the orchestrator. Logs but never raises."""
    if not url:
        return
    try:
        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status >= 400:
                log.warning("callback %s returned status %d", url, resp.status)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("callback POST to %s failed: %s", url, exc)
    except Exception:
        log.exception("callback POST to %s raised unexpectedly", url)


def _s3_url_for(key_or_url: str) -> str:
    """Return a full ``s3://bucket/key`` URL for the given input.

    If ``key_or_url`` already starts with ``s3://`` it's returned as-is;
    otherwise it's resolved against the worker's default bucket.
    """
    if key_or_url.startswith("s3://"):
        return key_or_url
    return f"s3://{settings.s3_bucket}/{key_or_url}"

_MIME_BY_EXT = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "html": "text/html",
    "htm": "text/html",
    "md": "text/markdown",
    "txt": "text/plain",
    "json": "application/json",
    "epub": "application/epub+zip",
    "csv": "text/csv",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
}


def _ext_of(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@celery_app.task(
    name="celery_app.tasks.parse_tasks.parse_document_task",
    bind=True,
    autoretry_for=(),       # explicit failure → DLQ; no auto-retry on parser errors
    max_retries=0,
    acks_late=True,
)
def parse_document_task(self, job_id: str) -> dict:
    started = time.perf_counter()
    job_uuid = uuid.UUID(job_id)
    job = JobRepo.get(job_uuid)
    if job is None:
        log.warning("job %s not found; dropping task", job_id)
        return {"state": "missing"}

    callback_url = (job.metadata_json or {}).get("callback_url")

    JobRepo.mark_running(job_uuid)
    _post_callback(callback_url, {
        "job_id": str(job_uuid),
        "state": "running",
        "pages_done": 0,
        "pages_total": job.pages_total or 0,
    })
    log.info("parse_document_task start id=%s file=%s", job_id, job.filename)

    ext = _ext_of(job.filename)
    parser = for_extension(ext)
    if parser is None:
        msg = f"unsupported extension: .{ext}"
        JobRepo.mark_failed(job_uuid, msg)
        _post_callback(callback_url, {
            "job_id": str(job_uuid),
            "state": "failed",
            "error": msg,
        })
        return {"state": "failed", "error": msg}

    stage_times: dict[str, float] = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_input = tmp_path / f"input.{ext}" if ext else tmp_path / "input"

            with timed() as t_dl:
                source = job.s3_input_key or ""
                print(f"[task {job_id[:8]}] downloading {source}", flush=True)
                if source.startswith("s3://"):
                    s3.download_url_to_file(source, str(local_input))
                else:
                    s3.download_to_file(source, str(local_input))
                payload = local_input.read_bytes()
            stage_times["s3_download"] = t_dl.seconds
            print(f"[task {job_id[:8]}] [timing] s3_download = {t_dl.seconds:.3f}s ({len(payload)} bytes)", flush=True)

            last_callback_ts = 0.0

            def progress_cb(done: int, total: int) -> None:
                nonlocal last_callback_ts
                JobRepo.update_progress(job_uuid, done, total)
                now = time.monotonic()
                if (
                    callback_url
                    and now - last_callback_ts >= _RUNNING_CALLBACK_MIN_INTERVAL_S
                ):
                    last_callback_ts = now
                    _post_callback(callback_url, {
                        "job_id": str(job_uuid),
                        "state": "running",
                        "pages_done": done,
                        "pages_total": total,
                    })

            with timed() as t_parse:
                print(f"[task {job_id[:8]}] starting parser={parser.name}", flush=True)
                result = parser.parse(payload, job.filename, progress_cb=progress_cb)
            stage_times["parse"] = t_parse.seconds
            print(f"[task {job_id[:8]}] [timing] parse = {t_parse.seconds:.3f}s "
                  f"(pages={result.page_count} images={len(result.images)})", flush=True)

        # Upload result.md and images to S3.
        prefix = f"{job_id}"
        markdown_key = f"{prefix}/result.md"
        with timed() as t_md:
            s3.put_bytes(markdown_key, result.markdown.encode("utf-8"), "text/markdown; charset=utf-8")
        stage_times["s3_upload_md"] = t_md.seconds

        image_prefix = f"{prefix}/images" if result.images else None
        if result.images:
            with timed() as t_imgs:
                for img in result.images:
                    rel = img.name
                    if rel.startswith("images/"):
                        rel = rel[len("images/"):]
                    s3.put_bytes(
                        f"{image_prefix}/{rel}",
                        base64.b64decode(img.bytes_b64),
                        img.mime,
                    )
            stage_times["s3_upload_images"] = t_imgs.seconds
        print(f"[task {job_id[:8]}] [timing] s3_upload "
              f"md={stage_times.get('s3_upload_md', 0):.3f}s "
              f"images={stage_times.get('s3_upload_images', 0):.3f}s "
              f"({len(result.images)} files)", flush=True)

        duration_ms = int((time.perf_counter() - started) * 1000)
        JobRepo.mark_done(
            job_uuid,
            s3_markdown_key=markdown_key,
            s3_image_prefix=image_prefix,
            image_count=len(result.images),
            pages_total=result.page_count,
            duration_ms=duration_ms,
            parser=result.parser,
            mode=str(result.metadata.get("mode")) if result.metadata else None,
            metadata={**(result.metadata or {}), **(job.metadata_json or {})},
        )
        _post_callback(callback_url, {
            "job_id": str(job_uuid),
            "state": "done",
            "pages_done": result.page_count,
            "pages_total": result.page_count,
            "s3_markdown_url": _s3_url_for(markdown_key),
            "s3_markdown_key": markdown_key,
            "image_count": len(result.images),
            "duration_ms": duration_ms,
        })
        timing_summary = " ".join(f"{k}={v:.2f}s" for k, v in stage_times.items())
        print(f"[task {job_id[:8]}] DONE pages={result.page_count} "
              f"images={len(result.images)} duration={duration_ms}ms", flush=True)
        print(f"[task {job_id[:8]}] [timing summary] {timing_summary} "
              f"total={duration_ms / 1000:.2f}s", flush=True)
        return {
            "state": "done",
            "pages": result.page_count,
            "images": len(result.images),
            "duration_ms": duration_ms,
            "timing": stage_times,
        }

    except Exception as e:
        log.exception("parse_document_task failed id=%s", job_id)
        JobRepo.mark_failed(job_uuid, repr(e))
        _post_callback(callback_url, {
            "job_id": str(job_uuid),
            "state": "failed",
            "error": repr(e),
        })
        raise
