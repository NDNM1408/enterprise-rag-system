"""SQLAlchemy engine, session, and ORM models.

Synchronous engine (FastAPI handlers run in threadpool, Celery tasks are sync).
Schema ``parsing`` keeps these tables isolated from data-api's ``public``.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from settings import settings


class Base(DeclarativeBase):
    pass


class JobState(str, enum.Enum):
    PENDING = "pending"     # row inserted, awaiting worker
    RUNNING = "running"     # worker picked up
    DONE = "done"           # markdown + images uploaded to S3
    FAILED = "failed"       # exception, see ``error``


class ParsingJob(Base):
    __tablename__ = "parsing_job"
    __table_args__ = {"schema": "parsing"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default=JobState.PENDING.value, index=True)
    parser: Mapped[Optional[str]] = mapped_column(String(64))
    mode: Mapped[Optional[str]] = mapped_column(String(64))

    pages_total: Mapped[Optional[int]] = mapped_column(Integer)
    pages_done: Mapped[int] = mapped_column(Integer, default=0)

    s3_input_key: Mapped[Optional[str]] = mapped_column(Text)
    s3_markdown_key: Mapped[Optional[str]] = mapped_column(Text)
    s3_image_prefix: Mapped[Optional[str]] = mapped_column(Text)
    image_count: Mapped[int] = mapped_column(Integer, default=0)

    duration_ms: Mapped[Optional[int]] = mapped_column(BigInteger)
    metadata_json: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text)

    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


# ---------------------------------------------------------------------------
#  Engine + session
# ---------------------------------------------------------------------------

def _sync_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


_engine = create_engine(
    _sync_url(settings.database_url),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)


def get_engine():
    return _engine
