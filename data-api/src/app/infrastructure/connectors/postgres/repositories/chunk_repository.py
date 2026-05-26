"""Repository for Chunk read operations."""

import logging
from typing import List

from sqlalchemy import select, delete as sqla_delete
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.connectors.postgres.database import db_session
from app.infrastructure.connectors.postgres.schema import Chunk
from app.exceptions import DatabaseError


logger = logging.getLogger(__name__)


class ChunkRepository:
    def __init__(self) -> None:
        self.async_session = db_session.get_session()

    async def get_full_by_document_id(self, document_id: str) -> List[dict]:
        """
        Return all chunks for a document with content + parent_text, ordered
        by insertion time. Used by the preview endpoint to show what the
        ingest pipeline actually produced.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(
                        Chunk.id,
                        Chunk.content,
                        Chunk.parent_text,
                        Chunk.status,
                        Chunk.create_time,
                    )
                    .where(Chunk.document_id == document_id)
                    .order_by(Chunk.create_time.asc(), Chunk.id.asc())
                )
                return [
                    {
                        "id": row.id,
                        "content": row.content or "",
                        "parent_text": row.parent_text or "",
                        "status": (
                            row.status.value if hasattr(row.status, "value") else row.status
                        ),
                    }
                    for row in result.fetchall()
                ]
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_full_by_document_id: {exc}")
            raise DatabaseError(f"Failed to retrieve chunks: {exc}") from exc

    async def get_contents_by_document_id(self, document_id: str) -> List[str]:
        """
        Return the text content of all chunks for a given document.

        Used when dispatching a LightRAG delete task so the worker can
        recompute LightRAG's internal doc_ids without a DB round-trip.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Chunk.content).where(Chunk.document_id == document_id)
                )
                return [c for c in result.scalars().all() if c]
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_contents_by_document_id: {exc}")
            raise DatabaseError(f"Failed to retrieve chunk contents: {exc}") from exc

    async def get_ids_by_document_id(self, document_id: str) -> list[str]:
        """Return all chunk IDs for a given document."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Chunk.id).where(Chunk.document_id == document_id)
                )
                return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_ids_by_document_id: {exc}")
            raise DatabaseError(f"Failed to retrieve chunk IDs: {exc}") from exc

    async def get_by_ids(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch chunks. Returns {chunk_id: {"content": ..., "doc_name": ...}}"""
        if not chunk_ids:
            return {}
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Chunk.id, Chunk.content, Chunk.doc_name)
                    .where(Chunk.id.in_(chunk_ids))
                )
                return {
                    row.id: {"content": row.content or "", "doc_name": row.doc_name or ""}
                    for row in result.fetchall()
                }
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_by_ids: {exc}")
            raise DatabaseError(f"Failed to retrieve chunks by IDs: {exc}") from exc

    async def get_s3_urls_by_document_id(self, document_id: str) -> List[str]:
        """
        Get all non-null chunk S3 URLs for a given document.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Chunk.chunk_s3_url).where(Chunk.document_id == document_id)
                )
                return [url for url in result.scalars().all() if url]
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_s3_urls_by_document_id: {exc}")
            raise DatabaseError(f"Failed to retrieve chunk S3 URLs: {exc}") from exc

    async def delete_by_kb_id(self, kb_id: str) -> int:
        """
        Delete all chunks for a knowledge base.

        Returns:
            Number of rows deleted

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        sqla_delete(Chunk).where(Chunk.kb_id == kb_id)
                    )
            logger.info(f"Deleted {result.rowcount} chunks for kb {kb_id}")
            return result.rowcount
        except SQLAlchemyError as exc:
            logger.error(f"Database error in delete_by_kb_id: {exc}")
            raise DatabaseError(f"Failed to delete chunks: {exc}") from exc

    async def get_s3_urls_by_kb_id(self, kb_id: str) -> List[str]:
        """
        Get all non-null chunk S3 URLs for a knowledge base.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Chunk.chunk_s3_url).where(
                        Chunk.kb_id == kb_id,
                        Chunk.chunk_s3_url.isnot(None)
                    )
                )
                return [url for url in result.scalars().all() if url]
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_s3_urls_by_kb_id: {exc}")
            raise DatabaseError(f"Failed to retrieve chunk S3 URLs: {exc}") from exc
