"""initial schema - create base tables

Revision ID: 000
Revises:
Create Date: 2026-02-19

"""
from alembic import op
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = '000'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Create enum types
    conn.execute(text("""
        DO $$ BEGIN
            CREATE TYPE "DocumentStatus" AS ENUM ('Created', 'Processing', 'Succeed', 'Failed');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """))

    conn.execute(text("""
        DO $$ BEGIN
            CREATE TYPE "ChunkStatus" AS ENUM ('Processing', 'Succeed', 'Failed');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """))

    # Create knowledge_base table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS "knowledge_base" (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            description TEXT,
            tenant_id VARCHAR,
            created_by VARCHAR,
            embed_id VARCHAR,
            parser_id VARCHAR,
            parser_config JSON,
            create_time TIMESTAMP NOT NULL DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW()
        )
    """))

    # Create document table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS "document" (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            kb_id VARCHAR REFERENCES "knowledge_base"(id) ON DELETE CASCADE,
            cmetadata JSON,
            create_time TIMESTAMP NOT NULL DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW(),
            status "DocumentStatus" NOT NULL
        )
    """))

    # Create chunk table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS "chunk" (
            id VARCHAR PRIMARY KEY,
            content TEXT,
            document_id VARCHAR REFERENCES "document"(id) ON DELETE CASCADE,
            doc_name VARCHAR,
            status "ChunkStatus" NOT NULL,
            create_time TIMESTAMP NOT NULL DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW()
        )
    """))


def downgrade() -> None:
    conn = op.get_bind()

    conn.execute(text('DROP TABLE IF EXISTS "chunk" CASCADE'))
    conn.execute(text('DROP TABLE IF EXISTS "document" CASCADE'))
    conn.execute(text('DROP TABLE IF EXISTS "knowledge_base" CASCADE'))
    conn.execute(text('DROP TYPE IF EXISTS "ChunkStatus"'))
    conn.execute(text('DROP TYPE IF EXISTS "DocumentStatus"'))
