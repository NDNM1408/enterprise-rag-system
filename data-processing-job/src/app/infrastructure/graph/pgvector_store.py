"""
PGVector embedding upsert for graph entities and relations.

Writes to GRAPHRAG_VDB_ENTITY and GRAPHRAG_VDB_RELATION tables
using SQLAlchemy async sessions with raw SQL — no ORM models needed.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from sqlalchemy import text

from app.application.services.embedding_service import EmbeddingService
from app.infrastructure.graph.prompts import GRAPH_FIELD_SEP

logger = logging.getLogger(__name__)


class GraphVectorStore:
    """Upsert entity/relation embeddings into GraphRAG PGVector tables."""

    def __init__(self, session_factory: Any, embedding_service: EmbeddingService) -> None:
        self._session_factory = session_factory
        self._embedding_service = embedding_service

    async def upsert_entity(
        self,
        workspace: str,
        vector_id: str,
        entity_name: str,
        content: str,
        source_id: str,
        file_path: str,
    ) -> None:
        """Upsert an entity embedding into GRAPHRAG_VDB_ENTITY."""
        embedding = await self._embedding_service.get_embedding(content)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        chunk_ids = source_id.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in source_id else [source_id]
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

        sql = text("""
            INSERT INTO "GRAPHRAG_VDB_ENTITY"
                (workspace, id, entity_name, content, content_vector, chunk_ids, file_path, create_time, update_time)
            VALUES
                (:workspace, :id, :entity_name, :content,
                 CAST(:vec AS vector), :chunk_ids, :file_path, :create_time, :update_time)
            ON CONFLICT (workspace, id) DO UPDATE SET
                entity_name = EXCLUDED.entity_name,
                content = EXCLUDED.content,
                content_vector = EXCLUDED.content_vector,
                chunk_ids = EXCLUDED.chunk_ids,
                file_path = EXCLUDED.file_path,
                update_time = EXCLUDED.update_time
        """)

        async with self._session_factory() as session:
            await session.execute(sql, {
                "workspace": workspace,
                "id": vector_id,
                "entity_name": entity_name,
                "content": content,
                "vec": embedding_str,
                "chunk_ids": chunk_ids,
                "file_path": file_path,
                "create_time": now,
                "update_time": now,
            })
            await session.commit()

        logger.debug("Upserted entity embedding: %s/%s", workspace, vector_id)

    async def upsert_relation(
        self,
        workspace: str,
        vector_id: str,
        src_id: str,
        tgt_id: str,
        content: str,
        source_id: str,
        file_path: str,
    ) -> None:
        """Upsert a relation embedding into GRAPHRAG_VDB_RELATION."""
        embedding = await self._embedding_service.get_embedding(content)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        chunk_ids = source_id.split(GRAPH_FIELD_SEP) if GRAPH_FIELD_SEP in source_id else [source_id]
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

        sql = text("""
            INSERT INTO "GRAPHRAG_VDB_RELATION"
                (workspace, id, source_id, target_id, content, content_vector,
                 chunk_ids, file_path, create_time, update_time)
            VALUES
                (:workspace, :id, :source_id, :target_id, :content,
                 CAST(:vec AS vector), :chunk_ids, :file_path, :create_time, :update_time)
            ON CONFLICT (workspace, id) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                target_id = EXCLUDED.target_id,
                content = EXCLUDED.content,
                content_vector = EXCLUDED.content_vector,
                chunk_ids = EXCLUDED.chunk_ids,
                file_path = EXCLUDED.file_path,
                update_time = EXCLUDED.update_time
        """)

        async with self._session_factory() as session:
            await session.execute(sql, {
                "workspace": workspace,
                "id": vector_id,
                "source_id": src_id,
                "target_id": tgt_id,
                "content": content,
                "vec": embedding_str,
                "chunk_ids": chunk_ids,
                "file_path": file_path,
                "create_time": now,
                "update_time": now,
            })
            await session.commit()

        logger.debug("Upserted relation embedding: %s/%s", workspace, vector_id)
