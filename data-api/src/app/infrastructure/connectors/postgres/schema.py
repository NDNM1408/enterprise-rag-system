import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Enum as SQLAlchemyEnum,
    Text,
    ForeignKey,
    JSON,
    TIMESTAMP,
)
from sqlalchemy.orm import relationship, declarative_base
from app.configurations.configurations import settings

Base = declarative_base()
schema = settings.PGSQL_SCHEMA


class DocumentStatus(enum.Enum):
    Created = "Created"
    Processing = "Processing"
    Succeed = "Succeed"
    Failed = "Failed"

class ChunkStatus(enum.Enum):
    Processing = "Processing"
    Succeed = "Succeed"
    Failed = "Failed"

def get_enum_values(enum_class):
    return [member.value for member in enum_class]


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    __table_args__ = {'schema': schema}
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    tenant_id = Column(String)
    created_by = Column(String)
    embed_id = Column(String)
    parser_id = Column(String)
    parser_config = Column(JSON)
    create_time = Column(TIMESTAMP, default=datetime.now, nullable=False)
    update_time = Column(TIMESTAMP, default=datetime.now, onupdate=datetime.now, nullable=True)

    # Relationships
    documents = relationship(
        "Document",
        back_populates="knowledgebase",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class Document(Base):
    __tablename__ = "document"
    __table_args__ = {'schema': schema}

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    name = Column(String, nullable=False)
    kb_id = Column(String, ForeignKey(f"{schema}.knowledge_base.id", ondelete="CASCADE"))
    cmetadata = Column(JSON)
    create_time = Column(TIMESTAMP, default=datetime.now, nullable=False)
    update_time = Column(TIMESTAMP, default=datetime.now, onupdate=datetime.now, nullable=True)
    status = Column(SQLAlchemyEnum(DocumentStatus, values_callable=get_enum_values, name="DocumentStatus", schema=schema), nullable=False)
    s3_url = Column(String)
    etag = Column(String)

    # Relationships
    knowledgebase = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = {'schema': schema}

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True, nullable=False)
    content = Column(Text)
    document_id = Column(String, ForeignKey(f"{schema}.document.id", ondelete="CASCADE"))
    kb_id = Column(String, ForeignKey(f"{schema}.knowledge_base.id", ondelete="CASCADE"))
    doc_name = Column(String)
    status = Column(SQLAlchemyEnum(ChunkStatus, values_callable=get_enum_values, name="ChunkStatus", schema=schema), nullable=False)

    # S3 location for this chunk's text file
    chunk_s3_url = Column(String)

    create_time = Column(TIMESTAMP, default=datetime.now, nullable=False)
    update_time = Column(TIMESTAMP, default=datetime.now, onupdate=datetime.now, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="chunks")


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
    llm_temperature = Column(String, default="0.7")
    is_active = Column(String, default="true")
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