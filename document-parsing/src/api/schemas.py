from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Sync /parse (legacy small-file path)
# ---------------------------------------------------------------------------

class ParseImage(BaseModel):
    name: str
    mime: str = "image/png"
    data_base64: str


class ParseResponse(BaseModel):
    markdown: str
    parser: str
    page_count: int = 0
    duration_ms: int
    filename: str
    metadata: dict = Field(default_factory=dict)
    images: list[ParseImage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  Async /jobs
# ---------------------------------------------------------------------------

class JobProgress(BaseModel):
    pages_done: int = 0
    pages_total: int | None = None
    pct: float | None = None


class JobResultLinks(BaseModel):
    markdown_url: str | None = None      # presigned S3 GET
    markdown_key: str | None = None
    image_count: int = 0
    image_prefix: str | None = None
    images_url: str | None = None        # presigned URL for the prefix? no — list endpoint


class JobResponse(BaseModel):
    id: uuid.UUID
    filename: str
    state: str
    parser: str | None = None
    mode: str | None = None
    progress: JobProgress
    result: JobResultLinks | None = None
    error: str | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None


class JobSubmitResponse(BaseModel):
    id: uuid.UUID
    state: str
    filename: str


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


# ---------------------------------------------------------------------------
#  Internal: submit by S3 reference (orchestrator path)
# ---------------------------------------------------------------------------

class JobByReferenceRequest(BaseModel):
    """Submit a parse job for a file already living in S3.

    The orchestrator (data-api) uploads the file once into its own bucket,
    then asks document-parsing to read it back from there — no second S3 copy.
    """

    filename: str = Field(..., description="Original filename (used to pick a parser)")
    source_url: str = Field(
        ...,
        description="Full S3 URL of the source file, e.g. 's3://bucket-name/path/to/file.pdf'",
    )
    callback_url: str | None = Field(
        default=None,
        description="HTTP URL the worker will POST progress / done / failed updates to",
    )
    external_document_id: str | None = Field(
        default=None,
        description="Opaque id echoed back in the callback for the caller's bookkeeping",
    )


# ---------------------------------------------------------------------------
#  Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    parsers: dict[str, str]
