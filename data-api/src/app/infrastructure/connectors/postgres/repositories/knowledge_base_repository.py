"""Repository for knowledge base CRUD operations."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import delete as sqla_delete, update as sqla_update, func, desc, asc, select
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.connectors.postgres.database import db_session
from app.infrastructure.connectors.postgres.schema import KnowledgeBase
from app.exceptions import DatabaseError, ResourceNotFoundError


logger = logging.getLogger(__name__)


class KnowledgeBaseRepository:
    def __init__(self) -> None:
        self.async_session = db_session.get_session()

    async def get(self, **kwargs: Any) -> KnowledgeBase:
        """
        Get a knowledge base by filter criteria.

        Raises:
            ResourceNotFoundError: If not found
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                result = await session.execute(select(KnowledgeBase).filter_by(**kwargs))
                record = result.scalars().first()

            if record is None:
                raise ResourceNotFoundError("KnowledgeBase", kwargs.get("id", "unknown"))
            return record

        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get: {exc}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {exc}") from exc

    async def create(self, **kwargs: Any) -> KnowledgeBase:
        """
        Create and persist a new knowledge base.

        Returns:
            The created KnowledgeBase instance

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    instance = KnowledgeBase(**kwargs)
                    session.add(instance)
                    await session.flush()
                    await session.refresh(instance)
            return instance
        except SQLAlchemyError as exc:
            logger.error(f"Database error in create: {exc}")
            raise DatabaseError(f"Failed to create knowledge base: {exc}") from exc

    async def update(self, data: Dict[str, Any], where: Dict[str, Any]) -> None:
        """
        Update knowledge base fields.

        Raises:
            ResourceNotFoundError: If no matching record found
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                async with session.begin():
                    stmt = (
                        sqla_update(KnowledgeBase)
                        .where(*(getattr(KnowledgeBase, k) == v for k, v in where.items()))
                        .values(**data)
                        .execution_options(synchronize_session="fetch")
                    )
                    result = await session.execute(stmt)

            if result.rowcount == 0:
                raise ResourceNotFoundError("KnowledgeBase", str(where))
        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"Database error in update: {exc}")
            raise DatabaseError(f"Failed to update knowledge base: {exc}") from exc

    async def delete(self, **kwargs: Any) -> None:
        """
        Delete a knowledge base by ID.

        Raises:
            ResourceNotFoundError: If not found
            DatabaseError: On DB failure
        """
        record_id = kwargs.get("id")
        if not record_id:
            raise DatabaseError("id is required to delete a knowledge base")

        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        sqla_delete(KnowledgeBase).where(KnowledgeBase.id == record_id)
                    )

            if result.rowcount == 0:
                raise ResourceNotFoundError("KnowledgeBase", record_id)
            logger.info(f"Deleted knowledge base {record_id}")

        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"Database error in delete: {exc}")
            raise DatabaseError(f"Failed to delete knowledge base: {exc}") from exc

    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        """
        Count knowledge bases matching optional filter.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                stmt = select(func.count()).select_from(KnowledgeBase)
                if where:
                    for k, v in where.items():
                        stmt = stmt.where(getattr(KnowledgeBase, k) == v)
                result = await session.execute(stmt)
                return result.scalar_one()
        except SQLAlchemyError as exc:
            logger.error(f"Database error in count: {exc}")
            raise DatabaseError(f"Failed to count knowledge bases: {exc}") from exc

    async def paging(
        self,
        skip: int = 0,
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[Dict[str, Any]] = None,
    ) -> List[KnowledgeBase]:
        """
        Return a paginated list of knowledge bases.

        Raises:
            DatabaseError: On DB failure
        """
        try:
            async with self.async_session() as session:
                stmt = select(KnowledgeBase)

                if where:
                    for field, value in where.items():
                        stmt = stmt.where(getattr(KnowledgeBase, field) == value)

                if order_by:
                    for field, direction in order_by.items():
                        col = getattr(KnowledgeBase, field)
                        stmt = stmt.order_by(desc(col) if direction.lower() == "desc" else asc(col))

                stmt = stmt.offset(skip).limit(limit)
                result = await session.execute(stmt)
                return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(f"Database error in paging: {exc}")
            raise DatabaseError(f"Failed to list knowledge bases: {exc}") from exc
