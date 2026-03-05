"""add KB rag_mode and embedding_dim columns

Revision ID: 002
Revises: 001
Create Date: 2026-02-18

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Create rag_mode enum type
    conn.execute(text("""
        DO $$ BEGIN
            CREATE TYPE public.rag_mode_enum AS ENUM ('classic', 'lightrag');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """))

    # Add rag_mode column to knowledge_base
    conn.execute(text("""
        ALTER TABLE public.knowledge_base
        ADD COLUMN IF NOT EXISTS rag_mode public.rag_mode_enum DEFAULT 'classic'
    """))

    # Add embedding_dim column
    conn.execute(text("""
        ALTER TABLE public.knowledge_base
        ADD COLUMN IF NOT EXISTS embedding_dim INTEGER DEFAULT 1024
    """))


def downgrade() -> None:
    conn = op.get_bind()

    # Drop columns
    conn.execute(text("""
        ALTER TABLE public.knowledge_base
        DROP COLUMN IF EXISTS rag_mode
    """))

    conn.execute(text("""
        ALTER TABLE public.knowledge_base
        DROP COLUMN IF EXISTS embedding_dim
    """))

    # Drop enum type
    conn.execute(text("DROP TYPE IF EXISTS public.rag_mode_enum"))
