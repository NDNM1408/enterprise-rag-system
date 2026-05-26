"""add agent, agent_knowledge_base, conversation, message tables

Revision ID: 011
Revises: 010
Create Date: 2026-05-26

Adds the chatbot tables that were merged in from the former chatbot-service:
  - agent: chatbot agent definitions
  - agent_knowledge_base: M:N link agent ↔ knowledge_base
  - conversation: chat session per agent+user
  - message: chat turn within a conversation
"""
from alembic import op
from sqlalchemy import text


revision = '011'
down_revision = '010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.agent (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            description TEXT,
            system_prompt TEXT,
            llm_model VARCHAR NOT NULL,
            llm_temperature VARCHAR DEFAULT '0.7',
            is_active VARCHAR DEFAULT 'true',
            tenant_id VARCHAR,
            created_by VARCHAR,
            create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.agent_knowledge_base (
            id VARCHAR PRIMARY KEY,
            agent_id VARCHAR NOT NULL REFERENCES public.agent(id) ON DELETE CASCADE,
            kb_id VARCHAR NOT NULL,
            create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_agent_kb_agent_id
            ON public.agent_knowledge_base(agent_id)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_agent_kb_kb_id
            ON public.agent_knowledge_base(kb_id)
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.conversation (
            id VARCHAR PRIMARY KEY,
            agent_id VARCHAR NOT NULL REFERENCES public.agent(id) ON DELETE CASCADE,
            user_id VARCHAR NOT NULL,
            title VARCHAR,
            cmetadata JSON DEFAULT '{}'::json,
            create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_conversation_agent_user
            ON public.conversation(agent_id, user_id)
    """))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.message (
            id VARCHAR PRIMARY KEY,
            conversation_id VARCHAR NOT NULL REFERENCES public.conversation(id) ON DELETE CASCADE,
            role VARCHAR NOT NULL,
            content TEXT,
            create_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_message_conversation_id
            ON public.message(conversation_id)
    """))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(text("DROP TABLE IF EXISTS public.message"))
    conn.execute(text("DROP TABLE IF EXISTS public.conversation"))
    conn.execute(text("DROP TABLE IF EXISTS public.agent_knowledge_base"))
    conn.execute(text("DROP TABLE IF EXISTS public.agent"))
