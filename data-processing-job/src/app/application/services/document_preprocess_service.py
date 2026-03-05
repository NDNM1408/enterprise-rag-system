"""
Shared document preprocessing service: fetch → parse → split → store.

Both the vector-embedding pipeline (preprocess_document task) and the
LightRAG graph pipeline (graph_preprocess_document task) call this service
for the common preprocessing step, passing different options for S3 chunk
upload.

Keeping the pipeline in one place means bug fixes and new features (e.g. a
new parser, a different chunking strategy) are made in a single file.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.application.core.parser import UnsupportedFileTypeError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    id: str
    content: str
    document_id: str
    kb_id: str
    doc_name: str
    status: str = "Processing"
    s3_path: Optional[str] = None
    chunk_s3_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DocumentNotFoundError(Exception):
    """Document row is missing from the database."""


class AlreadyProcessedError(Exception):
    """Document is already Processing or Succeed; skip silently."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_name_path(name: str) -> str:
    """Convert a filename to a filesystem-safe path component."""
    return (
        name.replace("://", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class DocumentPreprocessService:
    """
    Orchestrates the fetch → parse → split → store pipeline.

    Args:
        s3:         S3ClientService instance (shared from container).
        doc_repo:   DocumentRepository for document status reads/writes.
        chunk_repo: ChunkRepository for chunk inserts.
        parser:     DocumentParser instance (shared from container).
        splitter:   DocumentSplitter instance (shared from container).
    """

    def __init__(self, s3, doc_repo, chunk_repo, parser, splitter) -> None:
        self.s3 = s3
        self.doc_repo = doc_repo
        self.chunk_repo = chunk_repo
        self.parser = parser
        self.splitter = splitter

    async def preprocess(
        self,
        document_id: str,
        kb_id: str,
        name: str,
        bucket: str,
        *,
        upload_chunks: bool = False,
        chunk_bucket: Optional[str] = None,
    ) -> List[ChunkRecord]:
        """
        Run the full preprocessing pipeline for a document.

        Args:
            document_id:   UUID of the document record.
            kb_id:         UUID of the knowledge base.
            name:          Original filename (determines which parser to use).
            bucket:        S3 bucket containing the raw document.
            upload_chunks: If True, upload each chunk as a text file to S3.
                           Used by the vector-embedding path so upsert_chunk
                           can later fetch the text without reading the DB.
            chunk_bucket:  Destination bucket for chunk uploads.  Required
                           when upload_chunks=True.

        Returns:
            List of ChunkRecord objects ready for downstream task dispatch.

        Raises:
            DocumentNotFoundError:   document row is missing from the DB.
            AlreadyProcessedError:   document is already Processing or Succeed.
            UnsupportedFileTypeError: file extension not handled by the parser.
        """
        # ------------------------------------------------------------------
        # Idempotency / existence check
        # ------------------------------------------------------------------
        status = await self.doc_repo.get_status(document_id)
        if status is None:
            raise DocumentNotFoundError(
                f"Document {document_id} not found in database"
            )
        if status in ("Processing", "Succeed"):
            raise AlreadyProcessedError(
                f"Document {document_id} is already {status}, skipping"
            )

        # ------------------------------------------------------------------
        # Fetch from S3
        # ------------------------------------------------------------------
        logger.info("doc=%s: fetching %s/%s/%s", document_id, bucket, kb_id, name)
        content = await self.s3.get_file(bucket, f"{kb_id}/{name}")

        # ------------------------------------------------------------------
        # Parse to plain text
        # ------------------------------------------------------------------
        logger.info("doc=%s: parsing '%s'", document_id, name)
        # Propagates UnsupportedFileTypeError to the caller (task sets Failed)
        text_content = self.parser.parse(content, name)

        if not text_content or not text_content.strip():
            logger.warning("doc=%s: empty text after parsing '%s'", document_id, name)
            await self.doc_repo.set_status(document_id, "Failed")
            return []

        # ------------------------------------------------------------------
        # Split into chunks
        # ------------------------------------------------------------------
        raw_chunks = self.splitter.split(text_content)
        if not raw_chunks:
            logger.warning("doc=%s: no chunks produced from '%s'", document_id, name)
            await self.doc_repo.set_status(document_id, "Failed")
            return []

        logger.info("doc=%s: %d chunks", document_id, len(raw_chunks))

        # ------------------------------------------------------------------
        # Build records (+ optional S3 upload for vector path)
        # ------------------------------------------------------------------
        name_path = _format_name_path(name)
        records: List[ChunkRecord] = []

        for i, raw in enumerate(raw_chunks):
            chunk_id = str(uuid.uuid4())
            s3_path: Optional[str] = None
            chunk_s3_url: Optional[str] = None

            if upload_chunks and chunk_bucket:
                s3_path = f"{name_path}/{name}_{i}.txt"
                chunk_s3_url = f"s3://{chunk_bucket}/{kb_id}/{s3_path}"
                await self.s3.upload_file(
                    raw["content"].encode(),
                    chunk_bucket,
                    kb_id,
                    s3_path,
                )

            records.append(
                ChunkRecord(
                    id=chunk_id,
                    content=raw["content"],
                    document_id=document_id,
                    kb_id=kb_id,
                    doc_name=name,
                    s3_path=s3_path,
                    chunk_s3_url=chunk_s3_url,
                    metadata={
                        "chunk_order_index": raw["chunk_order_index"],
                        "tokens": raw["tokens"],
                    },
                )
            )

        # ------------------------------------------------------------------
        # Persist chunks + mark document Processing
        # ------------------------------------------------------------------
        await self.chunk_repo.batch_insert(
            [
                {
                    "id": r.id,
                    "content": r.content,
                    "document_id": r.document_id,
                    "kb_id": r.kb_id,
                    "doc_name": r.doc_name,
                    "status": r.status,
                    "chunk_s3_url": r.chunk_s3_url,
                }
                for r in records
            ]
        )
        await self.doc_repo.set_status(document_id, "Processing")

        return records
