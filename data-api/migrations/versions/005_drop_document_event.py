"""drop document_event table

Revision ID: 001
Revises: 000
Create Date: 2026-02-19

"""
from alembic import op
from sqlalchemy import text


revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(text('DROP TABLE IF EXISTS "document_event" CASCADE'))
    conn.execute(text('DROP TYPE IF EXISTS "DocumentAction"'))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(text("""
        DO $$ BEGIN
            CREATE TYPE "DocumentAction" AS ENUM ('Created', 'Updated', 'Deleted');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS "document_event" (
            id VARCHAR PRIMARY KEY,
            document_id VARCHAR NOT NULL REFERENCES "document"(id),
            name VARCHAR NOT NULL,
            create_time TIMESTAMP NOT NULL DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW(),
            action "DocumentAction" NOT NULL
        )
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_document_event_document_id
        ON "document_event" (document_id)
    """))
