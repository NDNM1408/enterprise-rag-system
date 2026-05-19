"""parent-child chunking — add parent_id, chunk_type, heading_path, token_count to chunk

Revision ID: 008
Revises: 007
Create Date: 2026-05-17

Splits each document into:
  • 'generate' chunks  — one per leaf markdown section, full content,
                          NOT embedded (skip vector upsert), status='Succeed' on insert.
  • 'retrieve' chunks  — paragraph/table groups within a section, capped at
                          gemini-embedding-001's 2048-token input limit,
                          embedded into the same `chunk.embedding` column.
                          ``parent_id`` points to the enclosing generate chunk.
"""
from alembic import op
from sqlalchemy import text


revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(text("""
        ALTER TABLE "chunk"
            ADD COLUMN IF NOT EXISTS parent_id VARCHAR
                REFERENCES "chunk"(id) ON DELETE CASCADE,
            ADD COLUMN IF NOT EXISTS chunk_type VARCHAR NOT NULL DEFAULT 'retrieve',
            ADD COLUMN IF NOT EXISTS heading_path TEXT,
            ADD COLUMN IF NOT EXISTS token_count INTEGER
    """))

    # Retrieve-side lookup: given a parent (generate) chunk, find its children.
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_chunk_parent_id ON "chunk" (parent_id)
    """))

    # Filter by type when dispatching embeddings / building retrieval prompts.
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_chunk_type ON "chunk" (chunk_type)
    """))


def downgrade() -> None:
    conn = op.get_bind()

    conn.execute(text('DROP INDEX IF EXISTS idx_chunk_type'))
    conn.execute(text('DROP INDEX IF EXISTS idx_chunk_parent_id'))
    conn.execute(text("""
        ALTER TABLE "chunk"
            DROP COLUMN IF EXISTS token_count,
            DROP COLUMN IF EXISTS heading_path,
            DROP COLUMN IF EXISTS chunk_type,
            DROP COLUMN IF EXISTS parent_id
    """))
