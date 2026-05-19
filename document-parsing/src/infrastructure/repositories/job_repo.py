"""Synchronous SQLAlchemy repository for ParsingJob.

Used by both the FastAPI handlers (insert/list/read/delete) and the Celery
worker (state transitions + progress updates).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from infrastructure.db import JobState, ParsingJob, SessionLocal


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobRepo:
    """Each method opens its own session to keep the API simple."""

    @staticmethod
    def create(
        *,
        filename: str,
        s3_input_key: str,
        parser: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ParsingJob:
        with SessionLocal() as s:
            job = ParsingJob(
                filename=filename,
                s3_input_key=s3_input_key,
                parser=parser,
                state=JobState.PENDING.value,
                metadata_json=metadata or {},
            )
            s.add(job)
            s.commit()
            s.refresh(job)
            return job

    @staticmethod
    def update_input_key(job_id: uuid.UUID, s3_input_key: str) -> None:
        with SessionLocal() as s:
            job = s.get(ParsingJob, job_id)
            if not job:
                return
            job.s3_input_key = s3_input_key
            s.commit()

    @staticmethod
    def get(job_id: uuid.UUID) -> Optional[ParsingJob]:
        with SessionLocal() as s:
            return s.get(ParsingJob, job_id)

    @staticmethod
    def list_recent(limit: int = 50) -> list[ParsingJob]:
        with SessionLocal() as s:
            stmt = (
                select(ParsingJob)
                .order_by(ParsingJob.submitted_at.desc())
                .limit(limit)
            )
            return list(s.execute(stmt).scalars())

    @staticmethod
    def mark_running(job_id: uuid.UUID) -> None:
        with SessionLocal() as s:
            job = s.get(ParsingJob, job_id)
            if not job:
                return
            job.state = JobState.RUNNING.value
            job.started_at = _utcnow()
            s.commit()

    @staticmethod
    def update_progress(
        job_id: uuid.UUID,
        pages_done: int,
        pages_total: Optional[int] = None,
    ) -> None:
        with SessionLocal() as s:
            job = s.get(ParsingJob, job_id)
            if not job:
                return
            job.pages_done = pages_done
            if pages_total is not None:
                job.pages_total = pages_total
            s.commit()

    @staticmethod
    def mark_done(
        job_id: uuid.UUID,
        *,
        s3_markdown_key: str,
        s3_image_prefix: Optional[str],
        image_count: int,
        pages_total: Optional[int],
        duration_ms: int,
        parser: Optional[str],
        mode: Optional[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        with SessionLocal() as s:
            job = s.get(ParsingJob, job_id)
            if not job:
                return
            job.state = JobState.DONE.value
            job.s3_markdown_key = s3_markdown_key
            job.s3_image_prefix = s3_image_prefix
            job.image_count = image_count
            job.pages_total = pages_total
            job.pages_done = pages_total or job.pages_done
            job.duration_ms = duration_ms
            if parser:
                job.parser = parser
            job.mode = mode
            job.metadata_json = metadata or {}
            job.finished_at = _utcnow()
            s.commit()

    @staticmethod
    def mark_failed(job_id: uuid.UUID, error: str) -> None:
        with SessionLocal() as s:
            job = s.get(ParsingJob, job_id)
            if not job:
                return
            job.state = JobState.FAILED.value
            job.error = error[:8000]  # cap to avoid pathological payloads
            job.finished_at = _utcnow()
            s.commit()

    @staticmethod
    def delete(job_id: uuid.UUID) -> bool:
        with SessionLocal() as s:
            job = s.get(ParsingJob, job_id)
            if not job:
                return False
            s.delete(job)
            s.commit()
            return True
