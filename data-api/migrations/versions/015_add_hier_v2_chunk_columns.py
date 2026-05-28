"""hier_v2: per-chunk type tag + embed_text + table_id + table_dataframe

Revision ID: 015
Revises: 014
Create Date: 2026-05-28

The hier_v2 chunking method emits three chunk types that share the existing
``chunk`` row layout but need three new columns:

  * ``chunk_type``      — 'text_child' | 'table_summary' | 'table_segment'
  * ``embed_text``      — what the embedding model reads (section_path +
                          retrieval text); ``content`` stays the verbatim
                          chunk and feeds generation.
  * ``table_id``        — shared between a ``table_summary`` and all its
                          ``table_segment`` siblings; lets the selector
                          collapse parent/segment pairs.
  * ``table_dataframe`` — base64-pickled pandas dict for downstream code
                          (e.g. answer LLM that wants tabular access).

All columns are nullable — legacy rows (and llm-wiki rows) keep working.
"""
from alembic import op


revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        ALTER TABLE public.chunk
          ADD COLUMN IF NOT EXISTS chunk_type      VARCHAR,
          ADD COLUMN IF NOT EXISTS embed_text      TEXT,
          ADD COLUMN IF NOT EXISTS table_id        VARCHAR,
          ADD COLUMN IF NOT EXISTS table_dataframe TEXT
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_chunk_type ON public.chunk (chunk_type)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_table_id ON public.chunk (table_id)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_chunk_table_id")
    op.execute("DROP INDEX IF EXISTS idx_chunk_chunk_type")
    op.execute(
        """
        ALTER TABLE public.chunk
          DROP COLUMN IF EXISTS table_dataframe,
          DROP COLUMN IF EXISTS table_id,
          DROP COLUMN IF EXISTS embed_text,
          DROP COLUMN IF EXISTS chunk_type
        """
    )
