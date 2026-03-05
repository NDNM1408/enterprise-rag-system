"""
Vector-embedding preprocessing pipeline tasks.

Task graph:
    preprocess_document
        → chord(
            group(upsert_chunk × N),
            finalize_document          ← runs when ALL upsert_chunk tasks finish
          )

Using a Celery chord instead of the old 'is_last_chunk' flag means
finalize_document is guaranteed to run even if the last chunk in the group
fails permanently after all retries.
"""
import asyncio
import logging
import time

from celery import chord, group

from app.celery_app.config import celery_app
from app.configurations.configurations import settings
from app.container import container
from app.application.services.document_preprocess_service import (
    DocumentPreprocessService,
    AlreadyProcessedError,
    DocumentNotFoundError,
)
from app.application.core.parser import UnsupportedFileTypeError
from app.infrastructure.repositories.document_repository import DocumentRepository
from app.infrastructure.repositories.chunk_repository import ChunkRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# preprocess_document
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="preprocess_document", acks_late=True, max_retries=3)
def preprocess_document(
    self,
    document_id: str,
    knowledge_base_id: str,
    name: str,
    embedding_model_id: str,
    bucket: str,
    correlation_id: str = None,
):
    """
    Fetch a document from S3, parse it, split it into chunks, store the
    chunks in the database and S3, then dispatch a chord of upsert_chunk
    tasks followed by finalize_document.

    Args:
        document_id:       UUID of the document record.
        knowledge_base_id: UUID of the knowledge base.
        name:              Document filename (determines parser).
        embedding_model_id: ID of the embedding model (passed through for audit).
        bucket:            S3 bucket containing the raw document.
        correlation_id:    Optional tracing / correlation ID.
    """
    log_prefix = f"[{correlation_id}]" if correlation_id else ""
    logger.info("%s preprocess_document starting: doc=%s", log_prefix, document_id)

    async def _run():
        start = time.time()
        doc_repo = DocumentRepository(container.session_factory)
        chunk_repo = ChunkRepository(container.session_factory)
        svc = DocumentPreprocessService(
            s3=container.s3,
            doc_repo=doc_repo,
            chunk_repo=chunk_repo,
            parser=container.parser,
            splitter=container.splitter,
        )

        try:
            chunk_records = await svc.preprocess(
                document_id=document_id,
                kb_id=knowledge_base_id,
                name=name,
                bucket=bucket,
                upload_chunks=True,
                chunk_bucket=settings.UPSERT_BUCKET_NAME,
            )
        except DocumentNotFoundError:
            logger.warning("%s doc=%s not found, skipping", log_prefix, document_id)
            return []
        except AlreadyProcessedError as e:
            logger.info("%s %s", log_prefix, e)
            return []
        except UnsupportedFileTypeError as e:
            logger.warning("%s doc=%s unsupported file '%s': %s", log_prefix, document_id, name, e)
            await doc_repo.set_status(document_id, "Failed")
            return []
        except Exception as e:
            logger.error(
                "%s doc=%s preprocess error: %s", log_prefix, document_id, e, exc_info=True
            )
            try:
                await doc_repo.set_status(document_id, "Failed")
            except Exception:
                pass
            raise

        logger.info(
            "%s doc=%s preprocessing done in %.2fs, %d chunks",
            log_prefix, document_id, time.time() - start, len(chunk_records),
        )
        return chunk_records

    try:
        chunk_records = asyncio.run(_run())

        if not chunk_records:
            return {"chunk_count": 0, "document_id": document_id}

        from app.celery_app.tasks.upsert_tasks import upsert_chunk

        chord(
            group(
                upsert_chunk.si(
                    chunk_id=r.id,
                    s3_path=r.s3_path,
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                    metadata=r.metadata,
                    correlation_id=correlation_id,
                ).set(queue="upsert_queue", routing_key="upsert.chunk")
                for r in chunk_records
            ),
            finalize_document.si(
                document_id=document_id,
                correlation_id=correlation_id,
            ).set(queue="preprocess_queue", routing_key="preprocess.finalize"),
        ).apply_async()

        logger.info(
            "%s doc=%s dispatched chord: %d upsert tasks → finalize",
            log_prefix, document_id, len(chunk_records),
        )
        return {"chunk_count": len(chunk_records), "document_id": document_id}

    except Exception as e:
        logger.error("%s doc=%s task failed: %s", log_prefix, document_id, e)
        raise self.retry(exc=e, countdown=min(2 ** self.request.retries * 60, 600))


# ---------------------------------------------------------------------------
# finalize_document
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="finalize_document", acks_late=True, max_retries=3)
def finalize_document(self, document_id: str, correlation_id: str = None):
    """
    Terminal step for the vector pipeline.

    Counts non-Succeed chunks and sets document status to Succeed or Failed.
    Triggered as the chord callback after all upsert_chunk tasks complete.

    Args:
        document_id:    UUID of the document.
        correlation_id: Optional tracing ID.
    """
    log_prefix = f"[{correlation_id}]" if correlation_id else ""
    logger.info("%s finalize_document: doc=%s", log_prefix, document_id)

    async def _run():
        chunk_repo = ChunkRepository(container.session_factory)
        doc_repo = DocumentRepository(container.session_factory)

        not_succeed = await chunk_repo.count_non_succeeded(document_id)
        final_status = "Succeed" if not_succeed == 0 else "Failed"
        await doc_repo.set_status(document_id, final_status)

        if not_succeed == 0:
            logger.info("%s doc=%s marked Succeed", log_prefix, document_id)
        else:
            logger.warning(
                "%s doc=%s has %d non-Succeed chunks, marked Failed",
                log_prefix, document_id, not_succeed,
            )

    try:
        asyncio.run(_run())
    except Exception as exc:
        logger.error("%s finalize_document failed: %s", log_prefix, exc, exc_info=True)
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries * 60, 600))
