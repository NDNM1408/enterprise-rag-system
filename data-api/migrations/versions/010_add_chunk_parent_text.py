"""denormalized parent-child chunking — store parent_text inline on each chunk

Revision ID: 010
Revises: 009
Create Date: 2026-05-20

Switches the chunking model from "separate parent rows + parent_id FK"
(migration 008) to "one row per retrieve chunk, parent context inlined".

Each chunk now carries:
  • content      — text that gets embedded + matched
  • parent_text  — full enclosing section, injected into the LLM prompt
                   on retrieval (the column added by this migration)

Legacy columns (chunk_type, parent_id) are kept so existing rows and any
in-flight workers don't break — the new splitter simply leaves them at
their defaults ('retrieve' and NULL).
"""
from alembic import op
from sqlalchemy import text


revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(text("""
        ALTER TABLE "chunk"
            ADD COLUMN IF NOT EXISTS parent_text TEXT
    """))

    # Backfill: for legacy retrieve rows produced by migration 008's splitter,
    # copy the enclosing generate chunk's content into parent_text so old data
    # behaves the same way under the new query path.
    conn.execute(text("""
        UPDATE "chunk" c
        SET parent_text = p.content
        FROM "chunk" p
        WHERE c.parent_id = p.id
          AND c.parent_text IS NULL
          AND p.chunk_type = 'generate'
    """))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(text('ALTER TABLE "chunk" DROP COLUMN IF EXISTS parent_text'))
