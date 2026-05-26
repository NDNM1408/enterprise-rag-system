"""Celery task: llm-wiki ingestion.

Fetches the document's markdown from S3 (either the user upload or the
pre-parsed output from the document-parsing service), splits it with the
Vietnamese-legal chunker, embeds each chunk via the same embedding service
the classic path uses, and indexes everything into a per-KB Elasticsearch
index.

Pipeline is single-task by design: ES bulk-index is fast enough that
splitting embedding + indexing across separate Celery tasks (the way the
classic path does for upsert_chunk) adds queue overhead without buying
parallelism — embeddings are already batched against the LiteLLM proxy.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from app.celery_app.config import celery_app
from app.configurations.configurations import settings
from app.container import container
from app.application.core.legal_chunker import chunk_legal_markdown
from app.infrastructure.search.es_indexer import ElasticsearchIndexer
from app.infrastructure.repositories.document_repository import DocumentRepository
from app.infrastructure.repositories.chunk_repository import ChunkRepository

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="llm_wiki_preprocess_document",
    acks_late=True,
    max_retries=3,
)
def llm_wiki_preprocess_document(
    self,
    document_id: str,
    knowledge_base_id: str,
    name: str,
    bucket: str,
    correlation_id: Optional[str] = None,
    parsed_markdown_s3_url: Optional[str] = None,
):
    """Ingest a document into the KB's Elasticsearch index."""
    log_prefix = f"[{correlation_id}]" if correlation_id else ""
    logger.info(
        "%s llm_wiki_preprocess starting: doc=%s parsed_md=%s",
        log_prefix, document_id, bool(parsed_markdown_s3_url),
    )

    async def _run():
        start = time.time()
        doc_repo = DocumentRepository(container.session_factory)
        chunk_repo = ChunkRepository(container.session_factory)

        await doc_repo.set_ingesting(document_id, status="Processing", progress=5)

        # 1. Fetch markdown source.
        if parsed_markdown_s3_url:
            md_bucket, md_key = _parse_s3_url(parsed_markdown_s3_url)
        else:
            md_bucket = bucket
            md_key = f"{knowledge_base_id}/{name}"

        try:
            markdown = await container.s3.get_txt_file_content(md_bucket, md_key)
        except Exception as exc:
            logger.error("%s failed to fetch markdown s3://%s/%s: %s",
                         log_prefix, md_bucket, md_key, exc)
            await doc_repo.set_status(document_id, "Failed")
            await doc_repo.finalize_ingesting(document_id, success=False)
            raise

        if not markdown or not markdown.strip():
            logger.warning("%s doc=%s empty markdown — nothing to index", log_prefix, document_id)
            await doc_repo.finalize_ingesting(document_id, success=True)
            return {"chunk_count": 0, "document_id": document_id}

        # 2. Chunk.
        legal_chunks = chunk_legal_markdown(markdown)
        if not legal_chunks:
            logger.warning("%s doc=%s chunker returned 0 chunks", log_prefix, document_id)
            await doc_repo.finalize_ingesting(document_id, success=True)
            return {"chunk_count": 0, "document_id": document_id}

        logger.info("%s doc=%s chunked into %d pieces", log_prefix, document_id, len(legal_chunks))
        await doc_repo.set_ingesting(document_id, progress=20)

        # 3. Embed (batched).
        texts = [c.content for c in legal_chunks]
        try:
            vectors = await container.embedding_service.get_embeddings_batch_chunked(texts)
        except Exception as exc:
            logger.error("%s doc=%s embedding failed: %s", log_prefix, document_id, exc)
            await doc_repo.set_status(document_id, "Failed")
            await doc_repo.finalize_ingesting(document_id, success=False)
            raise

        await doc_repo.set_ingesting(document_id, progress=70)

        # 4. Persist a lightweight Postgres ``chunk`` row per ES chunk so the
        #    Documents UI keeps showing chunk counts / progress consistently
        #    with classic-mode KBs. ``parent_text`` is set to the chunk body
        #    (legal articles are already self-contained).
        chunk_records: List[Dict[str, Any]] = []
        es_payload: List[Dict[str, Any]] = []
        for ch, vec in zip(legal_chunks, vectors):
            chunk_id = str(uuid.uuid4())
            heading_path = " > ".join(ch.heading_path) if ch.heading_path else None
            chunk_records.append({
                "id": chunk_id,
                "content": ch.content,
                "parent_text": ch.content,
                "document_id": document_id,
                "kb_id": knowledge_base_id,
                "doc_name": name,
                "status": "Succeed",
                "heading_path": heading_path,
                "token_count": None,
                "chunk_s3_url": None,
            })
            es_payload.append({
                "chunk_id": chunk_id,
                "content": ch.content,
                "section_label": ch.section_label,
                "heading_path": heading_path or "",
                "ordinal": ch.ordinal,
                "start_line": ch.start_line or None,
                "end_line": ch.end_line or None,
                "embedding": vec,
            })

        await chunk_repo.batch_insert(chunk_records)

        # 5. Index into Elasticsearch (idempotent — wipe previous chunks first).
        indexer = ElasticsearchIndexer()
        try:
            indexer.delete_document_chunks(knowledge_base_id, document_id)
            indexed = indexer.bulk_index(
                kb_id=knowledge_base_id,
                document_id=document_id,
                doc_name=name,
                chunks=es_payload,
            )
        finally:
            indexer.close()

        await doc_repo.set_ingesting(document_id, progress=100)
        await doc_repo.finalize_ingesting(document_id, success=True)

        logger.info(
            "%s doc=%s llm-wiki done in %.2fs — %d chunks indexed",
            log_prefix, document_id, time.time() - start, indexed,
        )
        return {"chunk_count": indexed, "document_id": document_id}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error("%s doc=%s task failed: %s", log_prefix, document_id, exc, exc_info=True)
        raise self.retry(exc=exc, countdown=min(2 ** self.request.retries * 60, 600))


def _parse_s3_url(url: str) -> tuple[str, str]:
    """``s3://bucket/key`` → ``(bucket, key)``."""
    without_scheme = url[len("s3://"):]
    bucket, key = without_scheme.split("/", 1)
    return bucket, key
