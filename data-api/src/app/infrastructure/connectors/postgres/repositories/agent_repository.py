"""Repository for agent CRUD operations."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import delete as sqla_delete, update as sqla_update, func, desc, asc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from app.infrastructure.connectors.postgres.database import db_session
from app.infrastructure.connectors.postgres.schema import Agent, AgentKnowledgeBase, KnowledgeBase
from app.exceptions import DatabaseError, ResourceNotFoundError


logger = logging.getLogger(__name__)


class AgentRepository:
    """Repository for Agent CRUD operations."""

    def __init__(self) -> None:
        self.async_session = db_session.get_session()

    async def get(self, **kwargs: Any) -> Agent:
        """Get an agent by filter criteria."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Agent)
                    .options(selectinload(Agent.knowledge_bases))
                    .filter_by(**kwargs)
                )
                record = result.scalars().first()

            if record is None:
                raise ResourceNotFoundError("Agent", kwargs.get("id", "unknown"))
            return record

        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get: {exc}")
            raise DatabaseError(f"Failed to retrieve agent: {exc}") from exc

    async def create(self, **kwargs: Any) -> Agent:
        """Create and persist a new agent."""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    instance = Agent(**kwargs)
                    session.add(instance)
                    await session.flush()
                    await session.refresh(instance)
            return instance
        except SQLAlchemyError as exc:
            logger.error(f"Database error in create: {exc}")
            raise DatabaseError(f"Failed to create agent: {exc}") from exc

    async def update(self, data: Dict[str, Any], where: Dict[str, Any]) -> Agent:
        """Update agent fields."""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    stmt = (
                        sqla_update(Agent)
                        .where(*(getattr(Agent, k) == v for k, v in where.items()))
                        .values(**data)
                        .execution_options(synchronize_session="fetch")
                    )
                    result = await session.execute(stmt)

            if result.rowcount == 0:
                raise ResourceNotFoundError("Agent", str(where))

            return await self.get(**where)
        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"Database error in update: {exc}")
            raise DatabaseError(f"Failed to update agent: {exc}") from exc

    async def delete(self, **kwargs: Any) -> None:
        """Delete an agent by ID."""
        record_id = kwargs.get("id")
        if not record_id:
            raise DatabaseError("id is required to delete an agent")

        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        sqla_delete(Agent).where(Agent.id == record_id)
                    )

            if result.rowcount == 0:
                raise ResourceNotFoundError("Agent", record_id)
            logger.info(f"Deleted agent {record_id}")

        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"Database error in delete: {exc}")
            raise DatabaseError(f"Failed to delete agent: {exc}") from exc

    async def paging(
        self,
        skip: int = 0,
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[Dict[str, Any]] = None,
    ) -> List[Agent]:
        """Return a paginated list of agents."""
        try:
            async with self.async_session() as session:
                stmt = select(Agent).options(selectinload(Agent.knowledge_bases))

                if where:
                    for field, value in where.items():
                        stmt = stmt.where(getattr(Agent, field) == value)

                if order_by:
                    for field, direction in order_by.items():
                        col = getattr(Agent, field)
                        stmt = stmt.order_by(desc(col) if direction.lower() == "desc" else asc(col))
                else:
                    stmt = stmt.order_by(desc(Agent.create_time))

                stmt = stmt.offset(skip).limit(limit)
                result = await session.execute(stmt)
                return list(result.scalars().all())

        except SQLAlchemyError as exc:
            logger.error(f"Database error in paging: {exc}")
            raise DatabaseError(f"Failed to list agents: {exc}") from exc

    async def get_linked_kb_ids(self, agent_id: str) -> List[str]:
        """Get list of knowledge base IDs linked to an agent."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(AgentKnowledgeBase.kb_id).where(
                        AgentKnowledgeBase.agent_id == agent_id
                    )
                )
                return [row[0] for row in result.all()]
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_linked_kb_ids: {exc}")
            raise DatabaseError(f"Failed to get linked KBs: {exc}") from exc

    async def get_linked_kbs(self, agent_id: str) -> List[Dict[str, str]]:
        """Return linked knowledge bases as {id, name} pairs (for UI rendering)."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(KnowledgeBase.id, KnowledgeBase.name)
                    .join(
                        AgentKnowledgeBase,
                        AgentKnowledgeBase.kb_id == KnowledgeBase.id,
                    )
                    .where(AgentKnowledgeBase.agent_id == agent_id)
                )
                return [{"id": row[0], "name": row[1]} for row in result.all()]
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_linked_kbs: {exc}")
            raise DatabaseError(f"Failed to get linked KBs: {exc}") from exc

    async def link_knowledge_base(self, agent_id: str, kb_id: str) -> AgentKnowledgeBase:
        """Link a knowledge base to an agent."""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    existing = await session.execute(
                        select(AgentKnowledgeBase).where(
                            AgentKnowledgeBase.agent_id == agent_id,
                            AgentKnowledgeBase.kb_id == kb_id,
                        )
                    )
                    existing_record = existing.scalars().first()
                    if existing_record:
                        logger.info(f"KB {kb_id} already linked to agent {agent_id}")
                        return existing_record

                    instance = AgentKnowledgeBase(agent_id=agent_id, kb_id=kb_id)
                    session.add(instance)
                    await session.flush()
                    await session.refresh(instance)
            return instance
        except SQLAlchemyError as exc:
            logger.error(f"Database error in link_knowledge_base: {exc}")
            raise DatabaseError(f"Failed to link knowledge base: {exc}") from exc

    async def unlink_knowledge_base(self, agent_id: str, kb_id: str) -> None:
        """Unlink a knowledge base from an agent."""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    result = await session.execute(
                        sqla_delete(AgentKnowledgeBase).where(
                            AgentKnowledgeBase.agent_id == agent_id,
                            AgentKnowledgeBase.kb_id == kb_id,
                        )
                    )
            if result.rowcount == 0:
                raise ResourceNotFoundError("AgentKnowledgeBase", f"{agent_id}/{kb_id}")
            logger.info(f"Unlinked KB {kb_id} from agent {agent_id}")
        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"Database error in unlink_knowledge_base: {exc}")
            raise DatabaseError(f"Failed to unlink knowledge base: {exc}") from exc
