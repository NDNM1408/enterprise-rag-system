"""Repository for conversation and message CRUD operations."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import delete as sqla_delete, update as sqla_update, desc, asc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from app.infrastructure.connectors.postgres.database import db_session
from app.infrastructure.connectors.postgres.schema import Conversation, Message
from app.exceptions import DatabaseError, ResourceNotFoundError


logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository for Conversation CRUD operations."""

    def __init__(self) -> None:
        self.async_session = db_session.get_session()

    async def get(self, **kwargs: Any) -> Conversation:
        """Get a conversation by filter criteria."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Conversation)
                    .options(selectinload(Conversation.messages))
                    .filter_by(**kwargs)
                )
                record = result.scalars().first()

            if record is None:
                raise ResourceNotFoundError("Conversation", kwargs.get("id", "unknown"))
            return record

        except ResourceNotFoundError:
            raise
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get: {exc}")
            raise DatabaseError(f"Failed to retrieve conversation: {exc}") from exc

    async def create(self, **kwargs: Any) -> Conversation:
        """Create and persist a new conversation."""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    instance = Conversation(**kwargs)
                    session.add(instance)
                    await session.flush()
                    await session.refresh(instance)
            return instance
        except SQLAlchemyError as exc:
            logger.error(f"Database error in create: {exc}")
            raise DatabaseError(f"Failed to create conversation: {exc}") from exc

    async def list_by_user(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> List[Conversation]:
        """List conversations for a user, optionally filtered by agent."""
        try:
            async with self.async_session() as session:
                stmt = select(Conversation).where(Conversation.user_id == user_id)
                if agent_id:
                    stmt = stmt.where(Conversation.agent_id == agent_id)
                stmt = stmt.order_by(desc(Conversation.update_time)).offset(skip).limit(limit)
                result = await session.execute(stmt)
                return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Database error in list_by_user: {exc}")
            raise DatabaseError(f"Failed to list conversations: {exc}") from exc

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> Message:
        """Add a message to a conversation."""
        try:
            async with self.async_session() as session:
                async with session.begin():
                    message = Message(
                        conversation_id=conversation_id,
                        role=role,
                        content=content,
                    )
                    session.add(message)
                    await session.flush()
                    await session.refresh(message)

                    await session.execute(
                        sqla_update(Conversation)
                        .where(Conversation.id == conversation_id)
                        .values(update_time=message.create_time)
                    )
            return message
        except SQLAlchemyError as exc:
            logger.error(f"Database error in add_message: {exc}")
            raise DatabaseError(f"Failed to add message: {exc}") from exc

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> List[Message]:
        """Get messages for a conversation, ordered by time."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(asc(Message.create_time))
                    .limit(limit)
                )
                return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.error(f"Database error in get_messages: {exc}")
            raise DatabaseError(f"Failed to get messages: {exc}") from exc

    async def get_or_create_conversation(
        self,
        agent_id: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        if conversation_id:
            try:
                return await self.get(id=conversation_id, user_id=user_id)
            except ResourceNotFoundError:
                logger.warning(f"Conversation {conversation_id} not found, creating new")

        return await self.create(agent_id=agent_id, user_id=user_id)
