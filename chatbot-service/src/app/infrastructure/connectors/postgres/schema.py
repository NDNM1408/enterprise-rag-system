"""
SQLAlchemy ORM models for chatbot service.
These tables will be created in the same database as data-api.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Text,
    ForeignKey,
    JSON,
    TIMESTAMP,
    Float,
    Boolean,
)
from sqlalchemy.orm import relationship, declarative_base
from app.configurations.settings import settings

Base = declarative_base()
schema = settings.PGSQL_SCHEMA


class Agent(Base):
    """Agent model for chatbot agents."""
    __tablename__ = "agent"
    __table_args__ = {'schema': schema}

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    system_prompt = Column(Text)
    llm_model = Column(String, nullable=False)
    llm_temperature = Column(Float, default=0.7)
    is_active = Column(Boolean, default=True)
    tenant_id = Column(String)
    created_by = Column(String)
    create_time = Column(TIMESTAMP, default=datetime.now, nullable=False)
    update_time = Column(TIMESTAMP, default=datetime.now, onupdate=datetime.now, nullable=True)

    # Relationships
    knowledge_bases = relationship(
        "AgentKnowledgeBase",
        back_populates="agent",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    conversations = relationship(
        "Conversation",
        back_populates="agent",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class AgentKnowledgeBase(Base):
    """Link table for Agent-KnowledgeBase many-to-many relationship."""
    __tablename__ = "agent_knowledge_base"
    __table_args__ = {'schema': schema}

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    agent_id = Column(String, ForeignKey(f"{schema}.agent.id", ondelete="CASCADE"), nullable=False)
    kb_id = Column(String, nullable=False)
    create_time = Column(TIMESTAMP, default=datetime.now, nullable=False)

    # Relationships
    agent = relationship("Agent", back_populates="knowledge_bases")


class Conversation(Base):
    """Conversation model for chat sessions."""
    __tablename__ = "conversation"
    __table_args__ = {'schema': schema}

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    agent_id = Column(String, ForeignKey(f"{schema}.agent.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=False)
    title = Column(String)
    cmetadata = Column(JSON, default={})
    create_time = Column(TIMESTAMP, default=datetime.now, nullable=False)
    update_time = Column(TIMESTAMP, default=datetime.now, onupdate=datetime.now, nullable=True)

    # Relationships
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Message(Base):
    """Message model for chat messages."""
    __tablename__ = "message"
    __table_args__ = {'schema': schema}

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    conversation_id = Column(String, ForeignKey(f"{schema}.conversation.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  # 'human', 'ai', 'system'
    content = Column(Text)
    create_time = Column(TIMESTAMP, default=datetime.now, nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
