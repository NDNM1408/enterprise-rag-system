"""
Graph ingestion tasks.

Architecture (per-chunk tasks for independent retryability):

  graph_preprocess_document
    - Idempotency: skip if already Succeed; resume if chunks exist.
    - Fresh run: fetch -> parse -> split -> batch-INSERT chunk records.
    - Resume run: load existing chunk records from DB.
    - Dispatch a sequential Celery chain:
        chain(graph_ingest_chunk x N, finalize_graph_document)
    - Returns immediately without waiting for ingestion.

  graph_ingest_chunk
    - Processes a single chunk via custom GraphIngestor -> mark Succeed.
    - Idempotent: skips if chunk is already Succeed.
    - Retried independently (max_retries=2, acks_late=True).
    - Marks chunk Failed on error before re-raising for Celery retry.

  finalize_graph_document
    - Counts non-succeeded chunks -> sets document Succeed or Failed.

Sequential chain (not parallel) preserves write ordering for graph
storage consistency.  If a chunk fails, only that chunk is retried --
the chain pauses at the failing task and resumes once it succeeds.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from celery import chain

# ---------------------------------------------------------------------------
# Persistent event loop per worker process.
#
# Neo4j driver pools are event-loop-bound.  Reusing one loop avoids
# "bound to a different event loop" errors across task invocations.
# ---------------------------------------------------------------------------
_worker_loop: asyncio.AbstractEventLoop | None = None


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop

from app.celery_app.config import celery_app
from app.container import container
from app.application.core.parser import UnsupportedFileTypeError
from app.infrastructure.repositories.document_repository import DocumentRepository
from app.infrastructure.repositories.chunk_repository import ChunkRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# graph_preprocess_document -- prepare chunks then dispatch per-chunk chain
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="graph_preprocess_document",
    acks_late=True,
    max_retries=2,
    soft_time_limit=300,
    time_limit=600,
)
def graph_preprocess_document(
    self,
    document_id: str,
    knowledge_base_id: str,
    name: str,
    bucket: str,
    correlation_id: str | None = None,
):
    """
    Fetch, parse, split, store chunks, then dispatch a sequential chain of
    graph_ingest_chunk tasks followed by finalize_graph_document.

    Returns immediately after dispatching; chunk processing is async.

    Args:
        document_id:        UUID of the document record.
        knowledge_base_id:  UUID of the knowledge base (graph workspace).
        name:               Original filename (drives parser selection).
        bucket:             S3 bucket containing the raw document.
        correlation_id:     Optional tracing ID.
    """
    log_prefix = f"[{correlation_id}]" if correlation_id else ""
    logger.info(
        "%s graph_preprocess_document starting: doc=%s kb=%s",
        log_prefix, document_id, knowledge_base_id,
    )

    async def _prepare() -> list | None:
        """
        Return list of pending chunks to dispatch, or None if nothing to do.
        Sets document/chunk status as a side effect.
        """
        doc_repo = DocumentRepository(container.session_factory)
        chunk_repo = ChunkRepository(container.session_factory)

        # ------------------------------------------------------------------
        # Gate: skip if already finished
        # ------------------------------------------------------------------
        status = await doc_repo.get_status(document_id)
        if status is None:
            logger.warning("%s doc=%s not found, skipping", log_prefix, document_id)
            return None
        if status == "Succeed":
            logger.info("%s doc=%s already Succeed, skipping", log_prefix, document_id)
            return None

        # ------------------------------------------------------------------
        # Idempotency: reuse chunk records from a previous (partial) run
        # ------------------------------------------------------------------
        existing_chunks = await chunk_repo.get_by_document(document_id)

        if existing_chunks:
            logger.info(
                "%s doc=%s resuming: %d existing chunks found",
                log_prefix, document_id, len(existing_chunks),
            )
            all_chunks = existing_chunks
        else:
            # ------------------------------------------------------------------
            # First run: fetch -> parse -> split -> insert chunk records
            # ------------------------------------------------------------------
            logger.info(
                "%s doc=%s first run: fetching %s/%s/%s",
                log_prefix, document_id, bucket, knowledge_base_id, name,
            )
            content = await container.s3.get_file(bucket, f"{knowledge_base_id}/{name}")

            try:
                text = container.parser.parse(content, name)
            except UnsupportedFileTypeError as e:
                logger.warning(
                    "%s doc=%s unsupported file '%s': %s", log_prefix, document_id, name, e
                )
                await doc_repo.set_status(document_id, "Failed")
                return None

            if not text or not text.strip():
                logger.warning("%s doc=%s empty text from '%s'", log_prefix, document_id, name)
                await doc_repo.set_status(document_id, "Failed")
                return None

            rows = container.splitter.split(text)
            if not rows:
                logger.warning(
                    "%s doc=%s no chunks from '%s'", log_prefix, document_id, name
                )
                await doc_repo.set_status(document_id, "Failed")
                return None

            logger.info(
                "%s doc=%s split into %d chunks",
                log_prefix, document_id, len(rows),
            )

            new_records = [
                {
                    "id": r.id,
                    "content": r.content,
                    "parent_text": r.parent_text,
                    "document_id": document_id,
                    "kb_id": knowledge_base_id,
                    "doc_name": name,
                    "status": "Processing",
                    "heading_path": r.heading_path,
                    "token_count": r.tokens,
                    "chunk_s3_url": None,
                }
                for r in rows
            ]
            await chunk_repo.batch_insert(new_records)
            await doc_repo.set_status(document_id, "Processing")
            all_chunks = new_records

        # ------------------------------------------------------------------
        # Filter to chunks still pending graph ingestion.
        # ------------------------------------------------------------------
        pending = [c for c in all_chunks if c.get("status") != "Succeed"]

        if not pending:
            # All chunks already succeeded -- finalize inline
            non_succeeded = await chunk_repo.count_non_succeeded(document_id)
            final_status = "Succeed" if non_succeeded == 0 else "Failed"
            await doc_repo.set_status(document_id, final_status)
            logger.info(
                "%s doc=%s all chunks already done, marked %s inline",
                log_prefix, document_id, final_status,
            )
            return None

        return pending

    try:
        pending = _get_worker_loop().run_until_complete(_prepare())
    except Exception as exc:
        logger.error(
            "%s graph_preprocess_document failed: doc=%s error=%s",
            log_prefix, document_id, exc, exc_info=True,
        )
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries * 60, 600))

    if pending is None:
        return {"document_id": document_id, "chunk_count": 0}

    # -----------------------------------------------------------------------
    # Dispatch sequential chain: ingest each chunk, then finalize
    # -----------------------------------------------------------------------
    chain(
        *[
            graph_ingest_chunk.si(
                chunk_id=c["id"],
                content=c["content"],
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                correlation_id=correlation_id,
            ).set(queue="graph_ingest_queue", routing_key="graph.ingest")
            for c in pending
        ],
        finalize_graph_document.si(
            document_id=document_id,
            correlation_id=correlation_id,
        ).set(queue="graph_ingest_queue", routing_key="graph.finalize"),
    ).apply_async()

    logger.info(
        "%s doc=%s dispatched chain of %d chunks",
        log_prefix, document_id, len(pending),
    )
    return {"document_id": document_id, "chunk_count": len(pending)}


# ---------------------------------------------------------------------------
# graph_ingest_chunk -- process a single chunk independently
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="graph_ingest_chunk",
    acks_late=True,
    max_retries=2,
    soft_time_limit=1800,
    time_limit=3600,
)
def graph_ingest_chunk(
    self,
    chunk_id: str,
    content: str,
    knowledge_base_id: str,
    document_id: str,
    correlation_id: str | None = None,
):
    """
    Ingest one chunk into the knowledge graph.

    Uses GraphIngestor (custom extraction + Neo4j + PGVector) instead of
    the full LightRAG library.

    Idempotent: if the chunk is already Succeed, returns immediately.
    On failure: marks chunk Failed and re-raises so Celery retries the task.

    Args:
        chunk_id:           UUID of the chunk record.
        content:            Text content of the chunk.
        knowledge_base_id:  UUID of the knowledge base (graph workspace).
        document_id:        UUID of the parent document.
        correlation_id:     Optional tracing ID.
    """
    log_prefix = f"[{correlation_id}]" if correlation_id else ""
    logger.info(
        "%s graph_ingest_chunk: chunk=%s doc=%s kb=%s",
        log_prefix, chunk_id, document_id, knowledge_base_id,
    )

    async def _run():
        chunk_repo = ChunkRepository(container.session_factory)

        # Idempotency check
        status = await chunk_repo.get_status(chunk_id)
        if status == "Succeed":
            logger.info("%s chunk=%s already Succeed, skipping", log_prefix, chunk_id)
            return {"chunk_id": chunk_id, "status": "skipped"}

        try:
            await container.graph_ingestor.ingest(
                content=content,
                kb_id=knowledge_base_id,
                file_path=document_id,
                chunk_key=chunk_id,
            )
            await chunk_repo.set_status(chunk_id, "Succeed")
            logger.info("%s chunk=%s ingested successfully", log_prefix, chunk_id)
        except Exception:
            await chunk_repo.set_status(chunk_id, "Failed")
            raise

        return {"chunk_id": chunk_id, "status": "success"}

    try:
        return _get_worker_loop().run_until_complete(_run())
    except Exception as exc:
        logger.error(
            "%s graph_ingest_chunk failed: chunk=%s error=%s",
            log_prefix, chunk_id, exc, exc_info=True,
        )
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries * 60, 600))


# ---------------------------------------------------------------------------
# finalize_graph_document -- set document status based on chunk outcomes
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="finalize_graph_document",
    acks_late=True,
    max_retries=3,
)
def finalize_graph_document(
    self,
    document_id: str,
    correlation_id: str | None = None,
):
    """
    Finalize a document after all graph_ingest_chunk tasks complete.

    Queries non-succeeded chunk count and sets document status to
    'Succeed' or 'Failed' accordingly.
    """
    log_prefix = f"[{correlation_id}]" if correlation_id else ""
    logger.info("%s finalize_graph_document: doc=%s", log_prefix, document_id)

    async def _run():
        chunk_repo = ChunkRepository(container.session_factory)
        doc_repo = DocumentRepository(container.session_factory)
        non_succeeded = await chunk_repo.count_non_succeeded(document_id)
        final_status = "Succeed" if non_succeeded == 0 else "Failed"
        await doc_repo.set_status(document_id, final_status)
        logger.info(
            "%s doc=%s marked %s (%d non-Succeed chunks)",
            log_prefix, document_id, final_status, non_succeeded,
        )
        return {
            "document_id": document_id,
            "final_status": final_status,
            "non_succeeded": non_succeeded,
        }

    try:
        return _get_worker_loop().run_until_complete(_run())
    except Exception as exc:
        logger.error(
            "%s finalize_graph_document failed: doc=%s error=%s",
            log_prefix, document_id, exc, exc_info=True,
        )
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries * 60, 600))
