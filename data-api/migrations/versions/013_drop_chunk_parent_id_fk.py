"""drop chunk.parent_id self-FK — v4 uses parent_id as a part UUID

Revision ID: 013
Revises: 012
Create Date: 2026-05-27

v4 chunking stores ``parent_id`` as the UUID of a "part" (a section slice
≤1 table); parts are not chunk rows, so the legacy self-referential FK
``chunk.parent_id → chunk.id`` (added in migration 008) blocks every v4
insert with ForeignKeyViolationError. Drop it.

The ``parent_id`` column and its btree index stay — retrieval-time dedupe
uses both.
"""
from alembic import op


revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE public.chunk DROP CONSTRAINT IF EXISTS chunk_parent_id_fkey")


def downgrade():
    # Recreating the FK would require purging every v4 chunk (parent_id is no
    # longer a chunk row). Leave the rollback as a no-op — callers that need
    # the old self-FK back must wipe the chunk table first.
    pass
