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
    content: str                          # verbatim chunk (generation reads this)
    parent_id: Optional[str]              # parent window id, or table_id for segments
    parent_text: str                      # parent window text or LLM enumeration
    document_id: str
    kb_id: str
    doc_name: str
    status: str = "Processing"
    heading_path: Optional[str] = None
    token_count: Optional[int] = None
    s3_path: Optional[str] = None
    chunk_s3_url: Optional[str] = None
    # hier_v2
    chunk_type: str = "text_child"
    embed_text: str = ""
    table_id: Optional[str] = None
    table_dataframe: Optional[str] = None
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
        splitter:   MarkdownSplitter instance (shared from container).
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
        parsed_markdown_s3_url: Optional[str] = None,
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
            parsed_markdown_s3_url: Optional full ``s3://bucket/key`` URL of
                           markdown produced by the document-parsing service.
                           When supplied, the local parser is skipped — the
                           markdown is fed straight to the splitter.

        Returns:
            List of ChunkRecord objects ready for downstream task dispatch.

        Raises:
            DocumentNotFoundError:   document row is missing from the DB.
            AlreadyProcessedError:   document is already Processing or Succeed.
            UnsupportedFileTypeError: file extension not handled by the parser.
        """
        # ------------------------------------------------------------------
        # Idempotency / existence check.
        # The legacy ``status`` column is now a rollup the orchestrator
        # (data-api) writes to as soon as the parse phase starts, so it
        # can already be 'Processing' before this task runs. The
        # phase-specific ``ingesting_status`` is what tells us whether the
        # embed pipeline itself has begun or finished.
        # ------------------------------------------------------------------
        status = await self.doc_repo.get_status(document_id)
        if status is None:
            raise DocumentNotFoundError(
                f"Document {document_id} not found in database"
            )
        ingesting = await self.doc_repo.get_ingesting_status(document_id)
        if ingesting == "Succeed":
            raise AlreadyProcessedError(
                f"Document {document_id} ingestion already Succeeded, skipping"
            )

        # ------------------------------------------------------------------
        # Source the text content: pre-parsed markdown (preferred) or fall
        # back to the local parser on the raw S3 object.
        # ------------------------------------------------------------------
        if parsed_markdown_s3_url:
            logger.info(
                "doc=%s: fetching pre-parsed markdown %s",
                document_id, parsed_markdown_s3_url,
            )
            text_content = await self.s3.get_txt_by_url(parsed_markdown_s3_url)
        else:
            logger.info("doc=%s: fetching %s/%s/%s", document_id, bucket, kb_id, name)
            content = await self.s3.get_file(bucket, f"{kb_id}/{name}")
            logger.info("doc=%s: parsing '%s'", document_id, name)
            # Propagates UnsupportedFileTypeError to the caller (task sets Failed)
            text_content = self.parser.parse(content, name)

        if not text_content or not text_content.strip():
            logger.warning("doc=%s: empty text after parsing '%s'", document_id, name)
            await self.doc_repo.set_status(document_id, "Failed")
            return []

        # ------------------------------------------------------------------
        # Split into chunks (hier_v2: block-bounded parent/child + LLM tables)
        # ------------------------------------------------------------------
        # Splitter is async because the per-table LLM call lives inside it.
        # We're already inside an async context (preprocess() is awaited from
        # the Celery task's asyncio.run wrapper), so call the async API.
        if hasattr(self.splitter, "asplit"):
            rows = await self.splitter.asplit(text_content)
        else:
            rows = self.splitter.split(text_content)
        if not rows:
            logger.warning("doc=%s: no chunks produced from '%s'", document_id, name)
            await self.doc_repo.set_status(document_id, "Failed")
            return []

        logger.info("doc=%s: %d chunks", document_id, len(rows))

        # ------------------------------------------------------------------
        # Build records. Every chunk is the same shape now — embed text +
        # inlined ``parent_text`` for LLM context. All start 'Processing'
        # and progress to 'Succeed' once the embedding lands.
        # ------------------------------------------------------------------
        name_path = _format_name_path(name)
        records: List[ChunkRecord] = []

        for row in rows:
            s3_path: Optional[str] = None
            chunk_s3_url: Optional[str] = None

            if upload_chunks and chunk_bucket:
                s3_path = f"{name_path}/{name}_{row.chunk_order_index}.txt"
                chunk_s3_url = f"s3://{chunk_bucket}/{kb_id}/{s3_path}"
                await self.s3.upload_file(
                    row.content.encode(),
                    chunk_bucket,
                    kb_id,
                    s3_path,
                )

            records.append(
                ChunkRecord(
                    id=row.id,
                    content=row.content,
                    parent_id=row.parent_id,
                    parent_text=row.parent_text,
                    document_id=document_id,
                    kb_id=kb_id,
                    doc_name=name,
                    status="Processing",
                    heading_path=row.heading_path,
                    token_count=row.tokens,
                    s3_path=s3_path,
                    chunk_s3_url=chunk_s3_url,
                    chunk_type=row.chunk_type,
                    embed_text=row.embed_text,
                    table_id=row.table_id,
                    table_dataframe=row.table_dataframe,
                    metadata={
                        "chunk_order_index": row.chunk_order_index,
                        "tokens": row.tokens,
                        "heading_path": row.heading_path,
                        "chunk_type": row.chunk_type,
                        "table_id": row.table_id,
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
                    "parent_id": r.parent_id,
                    "parent_text": r.parent_text,
                    "document_id": r.document_id,
                    "kb_id": r.kb_id,
                    "doc_name": r.doc_name,
                    "status": r.status,
                    "heading_path": r.heading_path,
                    "token_count": r.token_count,
                    "chunk_s3_url": r.chunk_s3_url,
                    "chunk_type": r.chunk_type,
                    "embed_text": r.embed_text,
                    "table_id": r.table_id,
                    "table_dataframe": r.table_dataframe,
                }
                for r in records
            ]
        )
        await self.doc_repo.set_status(document_id, "Processing")

        return records
