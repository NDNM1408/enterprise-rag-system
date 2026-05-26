"""drop graphrag tables — graph mode removed entirely

Revision ID: 012
Revises: 011
Create Date: 2026-05-26

Graph/Neo4j integration was removed in the llm-wiki feature branch. The
former GraphRAG vector tables are dropped here; the ``rag_mode_enum`` value
``'graphrag'`` is left in place because Postgres can't safely drop enum
values without rebuilding the column (and ``knowledge_base.rag_mode`` may
still reference it on legacy rows). New KBs may pick ``'classic'`` or
``'llm-wiki'``; the DTO enforces this.
"""
from alembic import op
from sqlalchemy import text


revision = '012'
down_revision = '011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(text('DROP TABLE IF EXISTS public."GRAPHRAG_VDB_RELATION" CASCADE'))
    conn.execute(text('DROP TABLE IF EXISTS public."GRAPHRAG_VDB_ENTITY" CASCADE'))


def downgrade() -> None:
    # The original mappings live in 007; downgrading there would re-create them.
    pass
