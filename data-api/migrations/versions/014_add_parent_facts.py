"""parent_facts — per-parent extracted-fact cache for extract-then-answer

Revision ID: 014
Revises: 013
Create Date: 2026-05-27

Stores the LLM-extracted atomic facts for each retrieval "part" (parent),
keyed by ``parent_id``. Extraction is query-independent (all facts in the
section), so the cache is reused across every query that retrieves the
parent — the chat path only extracts parents it has never seen.

``document_id`` FK cascades on delete so re-ingesting a document (which
drops its old chunks/parents) clears the stale fact rows automatically.
"""
from alembic import op


revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS public.parent_facts (
            parent_id    VARCHAR PRIMARY KEY,
            kb_id        VARCHAR,
            document_id  VARCHAR,
            facts        JSONB NOT NULL DEFAULT '[]'::jsonb,
            create_time  TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_parent_facts_kb_id ON public.parent_facts (kb_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_parent_facts_doc_id ON public.parent_facts (document_id)"
    )
    op.execute(
        """
        ALTER TABLE public.parent_facts
        ADD CONSTRAINT parent_facts_document_id_fkey
        FOREIGN KEY (document_id) REFERENCES public.document(id) ON DELETE CASCADE
        """
    )


def downgrade():
    op.execute("DROP TABLE IF EXISTS public.parent_facts")
