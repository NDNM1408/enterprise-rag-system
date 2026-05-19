"""Service for document management operations."""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx
from fastapi import UploadFile
from app.exceptions import ConflictError, ResourceNotFoundError, ExternalServiceError
from app.infrastructure.connectors.postgres.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.infrastructure.connectors.postgres.repositories.document_repository import DocumentRepository
from app.infrastructure.connectors.postgres.repositories.chunk_repository import ChunkRepository
from app.infrastructure.connectors.postgres.schema import (
    Document,
    IngestingStatus,
    ParsingStatus,
)
from app.infrastructure.clients.s3_client_service import S3ClientService
from app.celery_client import send_preprocess_task, send_graph_preprocess_task
from app.configurations.configurations import settings

import asyncio

logger = logging.getLogger(__name__)


# File extensions that already produce markdown-compatible text — the
# data-processing-job worker can handle these natively via its local
# DocumentParser, so we skip the document-parsing service round-trip.
NATIVE_TEXT_EXTENSIONS = {".md", ".markdown", ".html", ".htm", ".txt"}


def _extension(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def _is_native_text(filename: str) -> bool:
    return _extension(filename) in NATIVE_TEXT_EXTENSIONS


async def _graphrag_delete_document_graph(
    kb_id: str, document_id: str, chunk_ids: list[str]
) -> None:
    """
    Delete or trim GraphRAG graph data when a document is removed.

    LightRAG-inspired algorithm:
    - Entities/relations whose chunk_ids become empty after removing this
      document's chunks are fully deleted from GRAPHRAG_VDB_ENTITY/RELATION
      and Neo4j (DETACH DELETE removes incident edges automatically).
    - Entities/relations that still reference other documents' chunks have
      their chunk_ids trimmed in PGVector (description kept; no LLM rebuild).
    - Any GRAPHRAG_VDB_RELATION rows pointing to a deleted entity are also
      removed (orphan cleanup).
    """
    if not chunk_ids:
        return

    from app.infrastructure.graph.neo4j_store import Neo4jStore
    from app.infrastructure.connectors.postgres.database import db_session
    from sqlalchemy import text as sql_text

    chunk_ids_set = set(chunk_ids)
    session_factory = db_session.get_session()
    neo4j = Neo4jStore(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )

    try:
        async with session_factory() as pg:
            # --- Classify affected entities ---
            entity_rows = (
                await pg.execute(
                    sql_text("""
                        SELECT id, entity_name, chunk_ids
                        FROM "GRAPHRAG_VDB_ENTITY"
                        WHERE workspace = :kb_id
                          AND chunk_ids && CAST(:chunk_ids AS varchar[])
                    """),
                    {"kb_id": kb_id, "chunk_ids": list(chunk_ids_set)},
                )
            ).fetchall()

            entities_to_delete_ids: list[str] = []
            entities_to_delete_names: list[str] = []
            entities_to_update: list[tuple[str, list[str]]] = []

            for row in entity_rows:
                remaining = [c for c in (row.chunk_ids or []) if c not in chunk_ids_set]
                if not remaining:
                    entities_to_delete_ids.append(row.id)
                    entities_to_delete_names.append(row.entity_name)
                else:
                    entities_to_update.append((row.id, remaining))

            # --- Classify affected relations ---
            rel_rows = (
                await pg.execute(
                    sql_text("""
                        SELECT id, source_id, target_id, chunk_ids
                        FROM "GRAPHRAG_VDB_RELATION"
                        WHERE workspace = :kb_id
                          AND chunk_ids && CAST(:chunk_ids AS varchar[])
                    """),
                    {"kb_id": kb_id, "chunk_ids": list(chunk_ids_set)},
                )
            ).fetchall()

            relations_to_delete_ids: list[str] = []
            relations_to_delete_pairs: list[tuple[str, str]] = []
            relations_to_update: list[tuple[str, list[str]]] = []

            for row in rel_rows:
                remaining = [c for c in (row.chunk_ids or []) if c not in chunk_ids_set]
                if not remaining:
                    relations_to_delete_ids.append(row.id)
                    relations_to_delete_pairs.append((row.source_id, row.target_id))
                else:
                    relations_to_update.append((row.id, remaining))

            # --- Apply PGVector changes ---
            if entities_to_delete_ids:
                await pg.execute(
                    sql_text("""
                        DELETE FROM "GRAPHRAG_VDB_ENTITY"
                        WHERE workspace = :kb_id AND id = ANY(:ids)
                    """),
                    {"kb_id": kb_id, "ids": entities_to_delete_ids},
                )

            for vdb_id, remaining in entities_to_update:
                await pg.execute(
                    sql_text("""
                        UPDATE "GRAPHRAG_VDB_ENTITY"
                        SET chunk_ids = :chunk_ids
                        WHERE workspace = :kb_id AND id = :id
                    """),
                    {"kb_id": kb_id, "id": vdb_id, "chunk_ids": remaining},
                )

            if relations_to_delete_ids:
                await pg.execute(
                    sql_text("""
                        DELETE FROM "GRAPHRAG_VDB_RELATION"
                        WHERE workspace = :kb_id AND id = ANY(:ids)
                    """),
                    {"kb_id": kb_id, "ids": relations_to_delete_ids},
                )

            for vdb_id, remaining in relations_to_update:
                await pg.execute(
                    sql_text("""
                        UPDATE "GRAPHRAG_VDB_RELATION"
                        SET chunk_ids = :chunk_ids
                        WHERE workspace = :kb_id AND id = :id
                    """),
                    {"kb_id": kb_id, "id": vdb_id, "chunk_ids": remaining},
                )

            # --- Orphan cleanup: relations pointing to deleted entities ---
            if entities_to_delete_names:
                await pg.execute(
                    sql_text("""
                        DELETE FROM "GRAPHRAG_VDB_RELATION"
                        WHERE workspace = :kb_id
                          AND (source_id = ANY(:names) OR target_id = ANY(:names))
                    """),
                    {"kb_id": kb_id, "names": entities_to_delete_names},
                )

            await pg.commit()

        # --- Apply Neo4j changes ---
        # DETACH DELETE removes the node and all its incident edges.
        if entities_to_delete_names:
            await neo4j.delete_nodes_batch(kb_id, entities_to_delete_names)

        # Delete relation edges where both endpoints are kept but chunk_ids emptied.
        if relations_to_delete_pairs:
            await neo4j.delete_edges_batch(kb_id, relations_to_delete_pairs)

        logger.info(
            "GraphRAG delete: doc=%s kb=%s | "
            "entities deleted=%d updated=%d | relations deleted=%d updated=%d",
            document_id, kb_id,
            len(entities_to_delete_ids), len(entities_to_update),
            len(relations_to_delete_ids), len(relations_to_update),
        )

    except Exception as exc:
        logger.error(
            "GraphRAG graph delete failed for doc=%s kb=%s: %s",
            document_id, kb_id, exc, exc_info=True,
        )
    finally:
        await neo4j.close()


def _parse_s3_url(url: str) -> Tuple[str, str]:
    """Parse 's3://bucket/key' into (bucket, key)."""
    without_scheme = url[5:]  # strip "s3://"
    bucket, key = without_scheme.split("/", 1)
    return bucket, key


class DocumentsService:
    def __init__(
        self,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
        s3_client_service: S3ClientService,
        chunk_repository: ChunkRepository,
    ):
        self.knowledge_base_repository = knowledge_base_repository
        self.document_repository = document_repository
        self.s3_client_service = s3_client_service
        self.chunk_repository = chunk_repository

    async def add_documents(
        self,
        kb_id: str,
        files: List[UploadFile],
        cmetadata: Optional[Dict[str, Any]],
    ) -> None:
        """
        Upload documents to S3 and enqueue them for processing.

        Args:
            kb_id: Knowledge base ID
            files: Files to upload
            cmetadata: Optional custom metadata

        Raises:
            ResourceNotFoundError: If knowledge base does not exist
            ConflictError: If any filename already exists in this knowledge base
            ExternalServiceError: If S3 upload fails
        """
        # 1. Verify knowledge base exists (raises ResourceNotFoundError if not)
        knowledge_base = await self.knowledge_base_repository.get(id=kb_id)

        # 2. Read file bytes early so we can compute etags for duplicate detection
        file_datas = await asyncio.gather(*[f.read() for f in files])

        # 3. Compute MD5 etag for each file (matches S3 ETag for single-part uploads)
        file_etags = [hashlib.md5(data).hexdigest() for data in file_datas]

        # 4. Check for duplicate filenames and duplicate content (etag) in parallel
        conflict_names, conflict_etags = await asyncio.gather(
            self.document_repository.find_conflicts(kb_id, [f.filename for f in files]),
            self.document_repository.find_etag_conflicts(kb_id, file_etags),
        )

        if conflict_names or conflict_etags:
            details = {}
            if conflict_names:
                details["conflicting_filenames"] = list(conflict_names)
            if conflict_etags:
                details["conflicting_etags"] = list(conflict_etags)
            logger.warning(f"Duplicate documents detected in kb {kb_id}: {details}")
            raise ConflictError(
                message="One or more documents already exist in this knowledge base",
                details=details,
            )

        # 5. Upload to S3
        try:
            await asyncio.gather(*[
                self.s3_client_service.upload_file(
                    data_buffer=file_data,
                    bucket=settings.BUCKET_NAME,
                    kb_id=kb_id,
                    file_name=file.filename,
                )
                for file_data, file in zip(file_datas, files)
            ])
        except Exception as exc:
            raise ExternalServiceError("S3", f"Failed to upload files: {exc}") from exc

        # 6. Build document records — each starts in its appropriate parsing phase.
        documents: List[Document] = []
        for file, etag in zip(files, file_etags):
            native = _is_native_text(file.filename)
            documents.append(Document(
                id=str(uuid4()),
                kb_id=kb_id,
                name=file.filename,
                cmetadata=cmetadata,
                status="Created",
                s3_url=f"s3://{settings.BUCKET_NAME}/{kb_id}/{file.filename}",
                etag=etag,
                parsing_status=(ParsingStatus.Skipped if native else ParsingStatus.Pending).value,
                parsing_progress=100 if native else 0,
                ingesting_status=IngestingStatus.Pending.value,
                ingesting_progress=0,
            ))
        await self.document_repository.bulk_create(documents)

        # 7. Route each document: GraphRAG, direct preprocess (native), or parse+preprocess.
        rag_mode = (knowledge_base.parser_config or {}).get("rag_mode", "classic")

        for doc in documents:
            correlation_id = str(uuid4())

            if rag_mode == "graphrag":
                logger.info(
                    f"Enqueueing graph_ingest task for document {doc.id} "
                    f"[rag_mode=graphrag correlation_id={correlation_id}]"
                )
                send_graph_preprocess_task(
                    document_id=doc.id,
                    knowledge_base_id=kb_id,
                    name=doc.name,
                    bucket=settings.BUCKET_NAME,
                    correlation_id=correlation_id,
                )
                continue

            if _is_native_text(doc.name):
                logger.info(
                    f"Native text format — skipping parse, dispatching preprocess directly "
                    f"for document {doc.id} [correlation_id={correlation_id}]"
                )
                send_preprocess_task(
                    document_id=doc.id,
                    knowledge_base_id=kb_id,
                    name=doc.name,
                    embedding_model_id=knowledge_base.embed_id,
                    bucket=settings.BUCKET_NAME,
                    correlation_id=correlation_id,
                )
            else:
                logger.info(
                    f"Submitting parse job for document {doc.id} "
                    f"[ext={_extension(doc.name)} correlation_id={correlation_id}]"
                )
                try:
                    job_id = await self._submit_parse_job(
                        document_id=doc.id,
                        filename=doc.name,
                        source_url=doc.s3_url,
                    )
                    await self.document_repository.update_fields(
                        doc.id,
                        {"parsing_job_id": job_id},
                    )
                except Exception as exc:
                    logger.error(
                        f"Failed to submit parse job for document {doc.id}: {exc}",
                        exc_info=True,
                    )
                    await self.document_repository.update_fields(
                        doc.id,
                        {
                            "parsing_status": ParsingStatus.Failed.value,
                            "parsing_error": f"submit failed: {exc}",
                            "status": "Failed",
                        },
                    )

    async def _submit_parse_job(
        self,
        document_id: str,
        filename: str,
        source_url: str,
    ) -> str:
        """
        Ask the document-parsing service to parse a file already in S3.

        Returns the ParsingJob id (UUID string). Raises on transport / non-2xx.
        """
        callback_url = (
            f"{settings.DOCUMENT_PARSING_CALLBACK_BASE.rstrip('/')}"
            f"/api/v1/internal/parse-callback"
        )
        payload = {
            "filename": filename,
            "source_url": source_url,
            "callback_url": callback_url,
            "external_document_id": document_id,
        }
        url = f"{settings.DOCUMENT_PARSING_URL.rstrip('/')}/api/v1/jobs/by-reference"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        job_id = data.get("id") or data.get("job_id")
        if not job_id:
            raise ExternalServiceError(
                "document-parsing",
                f"submit returned no job id: {data}",
            )
        return str(job_id)

    async def handle_parse_callback(self, payload: Dict[str, Any]) -> None:
        """
        Apply a parse status update from the document-parsing service.

        Expected payload keys:
            job_id:           ParsingJob id
            state:            "running" | "done" | "failed"
            pages_done:       int   (optional)
            pages_total:      int   (optional)
            s3_markdown_url:  str   (full s3://bucket/key, present when state=done)
            error:            str   (present when state=failed)

        Side-effects:
            - Updates parsing_status / parsing_progress / parsed_markdown_s3_key / parsing_error
            - On state=done: dispatches preprocess_document so chunking starts
            - On state=failed: rolls the legacy `status` column up to Failed
        """
        job_id = payload.get("job_id") or payload.get("id")
        if not job_id:
            logger.warning("parse-callback: missing job_id, payload=%s", payload)
            return

        doc = await self.document_repository.get_by_parsing_job_id(str(job_id))
        if doc is None:
            logger.warning("parse-callback: no document for parsing_job_id=%s", job_id)
            return

        state = (payload.get("state") or "").lower()
        pages_done = payload.get("pages_done") or 0
        pages_total = payload.get("pages_total") or 0
        progress_pct = 0
        if pages_total and pages_total > 0:
            progress_pct = min(100, max(0, int(round(100.0 * pages_done / pages_total))))

        if state == "running":
            await self.document_repository.update_fields(
                doc.id,
                {
                    "parsing_status": ParsingStatus.Parsing.value,
                    "parsing_progress": progress_pct,
                    "status": "Processing",
                },
            )
            return

        if state == "failed":
            await self.document_repository.update_fields(
                doc.id,
                {
                    "parsing_status": ParsingStatus.Failed.value,
                    "parsing_error": (payload.get("error") or "")[:8000],
                    "status": "Failed",
                },
            )
            return

        if state == "done":
            markdown_url = payload.get("s3_markdown_url") or payload.get("markdown_url")
            await self.document_repository.update_fields(
                doc.id,
                {
                    "parsing_status": ParsingStatus.Parsed.value,
                    "parsing_progress": 100,
                    "parsed_markdown_s3_key": markdown_url,
                    "status": "Processing",
                },
            )

            # Hand off to the chunk/embed pipeline.
            knowledge_base = await self.knowledge_base_repository.get(id=doc.kb_id)
            correlation_id = str(uuid4())
            logger.info(
                f"Parse done for document {doc.id} — dispatching preprocess "
                f"[markdown_url={markdown_url} correlation_id={correlation_id}]"
            )
            send_preprocess_task(
                document_id=doc.id,
                knowledge_base_id=doc.kb_id,
                name=doc.name,
                embedding_model_id=knowledge_base.embed_id,
                bucket=settings.BUCKET_NAME,
                correlation_id=correlation_id,
                parsed_markdown_s3_url=markdown_url,
            )
            return

        logger.warning("parse-callback: ignoring unknown state=%s for job %s", state, job_id)

    async def list_documents(self, kb_id: str) -> List[Dict[str, Any]]:
        """
        List all documents in a knowledge base.

        Raises:
            ResourceNotFoundError: If the knowledge base does not exist
        """
        await self.knowledge_base_repository.get(id=kb_id)

        documents = await self.document_repository.get(kb_id=kb_id)
        return [
            {
                "id": doc.id,
                "name": doc.name,
                "kb_id": doc.kb_id,
                "status": doc.status if isinstance(doc.status, str) else doc.status.value,
                "parsing_status": doc.parsing_status,
                "parsing_progress": doc.parsing_progress,
                "parsing_error": doc.parsing_error,
                "ingesting_status": doc.ingesting_status,
                "ingesting_progress": doc.ingesting_progress,
                "etag": doc.etag,
                "cmetadata": doc.cmetadata,
                "create_time": doc.create_time.isoformat() if doc.create_time else None,
                "update_time": doc.update_time.isoformat() if doc.update_time else None,
            }
            for doc in documents
        ]

    async def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents and their associated S3 files.

        Args:
            document_ids: List of document IDs to delete

        Raises:
            ResourceNotFoundError: If no documents found for the given IDs
            ExternalServiceError: If S3 deletion fails
        """
        documents = await self.document_repository.get_by_ids(document_ids)
        if not documents:
            raise ResourceNotFoundError("Document", str(document_ids))

        # Delete files from S3
        try:
            await asyncio.gather(*[
                self.s3_client_service.delete_file(
                    settings.BUCKET_NAME,
                    doc.kb_id,
                    doc.name,
                )
                for doc in documents
            ])
        except Exception as exc:
            raise ExternalServiceError("S3", f"Failed to delete files: {exc}") from exc

        await self.document_repository.bulk_delete(document_ids)

    async def delete_document(self, kb_id: str, doc_id: str) -> None:
        """
        Delete a single document: removes the original S3 file, all chunk S3 files,
        and the document record (chunks cascade automatically).

        For GraphRAG knowledge bases, also deletes from the Neo4j graph and
        GraphRAG vector/KV stores directly (synchronously).

        Args:
            kb_id: Knowledge base ID the document belongs to
            doc_id: Document ID to delete

        Raises:
            ResourceNotFoundError: If the document does not exist or does not belong to the KB
            ExternalServiceError: If any S3 deletion fails
        """
        documents = await self.document_repository.get(id=doc_id, kb_id=kb_id)
        if not documents:
            raise ResourceNotFoundError("Document", doc_id)

        doc = documents[0]

        # For GraphRAG KBs: delete from graph/vector/KV stores before removing from DB
        knowledge_base = await self.knowledge_base_repository.get(id=kb_id)
        rag_mode = (knowledge_base.parser_config or {}).get("rag_mode", "classic")

        if rag_mode == "graphrag":
            chunk_ids = await self.chunk_repository.get_ids_by_document_id(doc_id)
            if chunk_ids:
                await _graphrag_delete_document_graph(
                    kb_id=kb_id,
                    document_id=doc_id,
                    chunk_ids=chunk_ids,
                )

        # Fetch S3 URLs for all chunks of this document
        chunk_s3_urls = await self.chunk_repository.get_s3_urls_by_document_id(doc_id)

        # Delete chunk files from S3
        if chunk_s3_urls:
            try:
                await asyncio.gather(*[
                    self.s3_client_service.delete_file_by_key(*_parse_s3_url(url))
                    for url in chunk_s3_urls
                ])
            except Exception as exc:
                raise ExternalServiceError("S3", f"Failed to delete chunk files: {exc}") from exc

        # Delete the original document file from S3
        try:
            await self.s3_client_service.delete_file(settings.BUCKET_NAME, doc.kb_id, doc.name)
        except Exception as exc:
            raise ExternalServiceError("S3", f"Failed to delete document file: {exc}") from exc

        # Delete document from DB (chunks cascade)
        await self.document_repository.bulk_delete([doc_id])
