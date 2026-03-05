"""add dlq_log table

Revision ID: 003
Revises: 002
Create Date: 2026-02-18

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Create dlq_log table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.dlq_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task_id TEXT NOT NULL,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """))

    # Create index on created_at for queries
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_dlq_log_created_at
        ON public.dlq_log (created_at DESC)
    """))


def downgrade() -> None:
    conn = op.get_bind()

    # Drop table
    conn.execute(text("DROP TABLE IF EXISTS public.dlq_log CASCADE"))
