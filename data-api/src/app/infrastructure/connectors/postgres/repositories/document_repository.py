"""Repository for document CRUD operations."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import delete as sqla_delete, select, update as sqla_update, and_, or_, func
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.connectors.postgres.database import db_session
from app.infrastructure.connectors.postgres.schema import Document
from app.exceptions import DatabaseError, ResourceNotFoundError


logger = logging.getLogger(__name__)


class DocumentRepository:
    def __init__(self) -> None:
        self.async_session = db_session.get_session()

    async def get(self, **kwargs: Any) -> List[Document]:
        """
        Get documents matching filter criteria.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(select(Document).filter_by(**kwargs))
                return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get: {exc}")
            raise DatabaseError(f"Failed to retrieve documents: {exc}") from exc

    async def get_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Get documents by list of IDs.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Document).where(Document.id.in_(ids))
                )
                return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_by_ids: {exc}")
            raise DatabaseError(f"Failed to retrieve documents: {exc}") from exc

    async def find_conflicts(self, kb_id: str, names: List[str]) -> List[str]:
        """
        Return filenames that already exist in the knowledge base (non-Failed status).

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                stmt = select(Document.name).where(
                    and_(
                        Document.kb_id == kb_id,
                        Document.status != "Failed",
                        Document.name.in_(names),
                    )
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Database error in find_conflicts: {exc}")
            raise DatabaseError(f"Failed to check for document conflicts: {exc}") from exc

    async def find_etag_conflicts(self, kb_id: str, etags: List[str]) -> List[str]:
        """
        Return etags that already exist in the knowledge base (non-Failed status).
        Used to detect duplicate file content regardless of filename.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                stmt = select(Document.etag).where(
                    and_(
                        Document.kb_id == kb_id,
                        Document.status != "Failed",
                        Document.etag.isnot(None),
                        Document.etag.in_(etags),
                    )
                )
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Database error in find_etag_conflicts: {exc}")
            raise DatabaseError(f"Failed to check for etag conflicts: {exc}") from exc

    async def bulk_create(self, documents: List[Document]) -> None:
        """
        Bulk insert documents.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    session.add_all(documents)
            logger.info(f"Bulk created {len(documents)} documents")
        except SQLAlchemyError as exc:
            logger.error(f"Database error in bulk_create: {exc}")
            raise DatabaseError(f"Failed to bulk create documents: {exc}") from exc

    async def bulk_delete(self, document_ids: List[str]) -> int:
        """
        Bulk delete documents by IDs.

        Returns:
            Number of rows deleted

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        sqla_delete(Document).where(Document.id.in_(document_ids))
                    )
            logger.info(f"Bulk deleted {result.rowcount} documents")
            return result.rowcount
        except SQLAlchemyError as exc:
            logger.error(f"Database error in bulk_delete: {exc}")
            raise DatabaseError(f"Failed to bulk delete documents: {exc}") from exc

    async def count_by_kb_ids(self, kb_ids: List[str]) -> dict:
        """
        Count documents grouped by knowledge base IDs.

        Returns:
            Dict mapping kb_id to document count

        Raises:
            DatabaseError: On DB failure
        """
        if not kb_ids:
            return {}

        try:
            async with self.async_session() as session:
                stmt = (
                    select(Document.kb_id, func.count(Document.id))
                    .where(Document.kb_id.in_(kb_ids))
                    .group_by(Document.kb_id)
                )
                result = await session.execute(stmt)
                return {row[0]: row[1] for row in result.all()}
        except SQLAlchemyError as exc:
            logger.error(f"Database error in count_by_kb_ids: {exc}")
            raise DatabaseError(f"Failed to count documents: {exc}") from exc

    async def delete_by_kb_id(self, kb_id: str) -> int:
        """
        Delete all documents for a knowledge base.

        Returns:
            Number of rows deleted

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        sqla_delete(Document).where(Document.kb_id == kb_id)
                    )
            logger.info(f"Deleted {result.rowcount} documents for kb {kb_id}")
            return result.rowcount
        except SQLAlchemyError as exc:
            logger.error(f"Database error in delete_by_kb_id: {exc}")
            raise DatabaseError(f"Failed to delete documents: {exc}") from exc

    async def update_fields(self, doc_id: str, fields: Dict[str, Any]) -> bool:
        """
        Update a subset of columns on a document row.

        Returns True if at least one row matched.

        Raises:
            DatabaseError: On DB failure
        """
        if not fields:
            return False
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        sqla_update(Document)
                        .where(Document.id == doc_id)
                        .values(**fields)
                    )
            return result.rowcount > 0
        except SQLAlchemyError as exc:
            logger.error(f"Database error in update_fields: {exc}")
            raise DatabaseError(f"Failed to update document {doc_id}: {exc}") from exc

    async def get_by_parsing_job_id(self, job_id: str) -> Optional[Document]:
        """
        Fetch the document referenced by a document-parsing ParsingJob id.

        Returns None when no document references this job (stale callback).

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Document).where(Document.parsing_job_id == job_id)
                )
                return result.scalars().first()
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_by_parsing_job_id: {exc}")
            raise DatabaseError(f"Failed to lookup document by parsing_job_id: {exc}") from exc

    async def get_s3_urls_by_kb_id(self, kb_id: str) -> List[str]:
        """
        Get all non-null document S3 URLs for a knowledge base.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Document.s3_url).where(
                        Document.kb_id == kb_id,
                        Document.s3_url.isnot(None)
                    )
                )
                return [url for url in result.scalars().all() if url]
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_s3_urls_by_kb_id: {exc}")
            raise DatabaseError(f"Failed to retrieve document S3 URLs: {exc}") from exc
