"""
Chunk embedding upsert task.

Fetches chunk text from S3, generates an embedding, writes it to pgvector,
and updates the chunk status.  Finalization is handled by the chord callback
(finalize_document) dispatched in preprocess_document — this task no longer
needs to know whether it is the "last" chunk.
"""
import asyncio
import logging
from typing import Any, Dict

from app.celery_app.config import celery_app
from app.configurations.configurations import settings
from app.container import container
from app.infrastructure.repositories.chunk_repository import ChunkRepository
from app.infrastructure.repositories.document_repository import DocumentRepository
from app.infrastructure.repositories.embedding_writer_repository import EmbeddingWriterRepository

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="upsert_chunk", acks_late=True, max_retries=3)
def upsert_chunk(
    self,
    chunk_id: str,
    s3_path: str,
    knowledge_base_id: str,
    document_id: str,
    metadata: Dict[str, Any],
    # is_last_chunk is accepted but ignored — finalization is handled by the
    # chord callback (finalize_document) dispatched by preprocess_document.
    is_last_chunk: bool = False,
    correlation_id: str = None,
):
    """
    Generate and store the embedding for one chunk.

    Args:
        chunk_id:          UUID of the chunk record.
        s3_path:           S3 key (relative to kb_id prefix) for the chunk text.
        knowledge_base_id: UUID of the knowledge base.
        document_id:       UUID of the parent document.
        metadata:          JSONB metadata stored alongside the embedding.
        is_last_chunk:     Deprecated; kept for backward-compat with in-flight
                           messages produced before the chord refactor.
        correlation_id:    Optional tracing ID.
    """
    log_prefix = f"[{correlation_id}]" if correlation_id else ""
    logger.info("%s upsert_chunk starting: chunk=%s", log_prefix, chunk_id)

    async def _run():
        chunk_repo = ChunkRepository(container.session_factory)

        # ------------------------------------------------------------------
        # Idempotency: skip if already succeeded
        # ------------------------------------------------------------------
        status = await chunk_repo.get_status(chunk_id)
        if status is None:
            logger.warning("%s chunk=%s not found in DB, skipping", log_prefix, chunk_id)
            return {"chunk_id": chunk_id, "status": "skipped"}
        if status == "Succeed":
            logger.info("%s chunk=%s already Succeed, skipping", log_prefix, chunk_id)
            return {"chunk_id": chunk_id, "status": "skipped"}

        try:
            # ------------------------------------------------------------------
            # Pick the text the embedder should read.
            # hier_v2: chunk.embed_text (section_path + retrieval text).
            # Legacy / null: fall back to S3 content (verbatim chunk body).
            # ------------------------------------------------------------------
            embed_input = await chunk_repo.get_text_for_embedding(chunk_id)
            if not embed_input:
                logger.info(
                    "%s chunk=%s no embed_text in DB, falling back to S3 %s/%s/%s",
                    log_prefix, chunk_id, settings.UPSERT_BUCKET_NAME,
                    knowledge_base_id, s3_path,
                )
                embed_input = await container.s3.get_txt_file_content(
                    settings.UPSERT_BUCKET_NAME,
                    f"{knowledge_base_id}/{s3_path}",
                )
            if not embed_input:
                raise ValueError(f"No text available to embed for chunk {chunk_id}")

            text_content = embed_input  # kept for downstream call signature

            # ------------------------------------------------------------------
            # Generate embedding
            # ------------------------------------------------------------------
            logger.info("%s chunk=%s generating embedding", log_prefix, chunk_id)
            embedding = await container.embedding_service.get_embedding(embed_input)

            # ------------------------------------------------------------------
            # Upsert into pgvector
            # ------------------------------------------------------------------
            logger.info("%s chunk=%s upserting embedding", log_prefix, chunk_id)
            async with container.session_factory() as session:
                repo = EmbeddingWriterRepository(session)
                await repo.upsert(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    kb_id=knowledge_base_id,
                    embedding=embedding,
                    content=text_content,
                    metadata=metadata,
                )

            # ------------------------------------------------------------------
            # Mark chunk succeeded and refresh the document's progress %.
            # The recompute is one atomic SQL — concurrent upsert_chunk tasks
            # for the same document can't desync the counter.
            # ------------------------------------------------------------------
            await chunk_repo.set_status(chunk_id, "Succeed")
            try:
                doc_repo = DocumentRepository(container.session_factory)
                await doc_repo.recompute_ingesting_progress(document_id)
            except Exception as prog_err:
                # Progress is cosmetic — never let it sink the upsert.
                logger.warning(
                    "%s chunk=%s progress refresh failed: %s",
                    log_prefix, chunk_id, prog_err,
                )
            logger.info("%s chunk=%s done", log_prefix, chunk_id)
            return {"chunk_id": chunk_id, "status": "success"}

        except Exception as e:
            logger.error(
                "%s chunk=%s error: %s", log_prefix, chunk_id, e, exc_info=True
            )
            try:
                await chunk_repo.set_status(chunk_id, "Failed")
            except Exception as db_err:
                logger.error("%s failed to mark chunk Failed: %s", log_prefix, db_err)
            raise

    try:
        return asyncio.run(_run())
    except Exception as e:
        logger.error("%s chunk=%s task failed: %s", log_prefix, chunk_id, e)
        raise self.retry(exc=e, countdown=min(2 ** self.request.retries * 60, 600))
