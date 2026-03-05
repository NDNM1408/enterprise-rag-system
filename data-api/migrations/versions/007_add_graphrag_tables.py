"""add graphrag tables and update rag_mode enum

Revision ID: 007
Revises: 006
Create Date: 2026-02-21

Changes:
  - Add 'graphrag' value to rag_mode_enum
  - Migrate existing 'lightrag' rows in knowledge_base to 'graphrag'
  - Create GRAPHRAG_VDB_ENTITY table (pgvector entity embeddings)
  - Create GRAPHRAG_VDB_RELATION table (pgvector relation embeddings)
"""
from alembic import op
from sqlalchemy import text


revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # ------------------------------------------------------------------
    # 1. Extend rag_mode_enum with the new 'graphrag' value.
    #    PostgreSQL requires that ALTER TYPE ... ADD VALUE be committed
    #    before we can use the new value in the same session.
    #    Use raw DBAPI connection with autocommit mode.
    # ------------------------------------------------------------------
    raw_conn = conn.connection
    old_isolation_level = raw_conn.isolation_level

    try:
        # Set autocommit mode (0 = autocommit in psycopg2)
        raw_conn.set_isolation_level(0)

        # Execute ALTER TYPE (will auto-commit)
        cursor = raw_conn.cursor()
        cursor.execute("ALTER TYPE public.rag_mode_enum ADD VALUE IF NOT EXISTS 'graphrag'")
        cursor.close()

        # Commit the change
        raw_conn.commit()
    finally:
        # Restore original isolation level
        raw_conn.set_isolation_level(old_isolation_level)

    # ------------------------------------------------------------------
    # 2. Migrate existing 'lightrag' knowledge bases to 'graphrag'.
    # ------------------------------------------------------------------
    conn.execute(text("""
        UPDATE public.knowledge_base
        SET rag_mode = 'graphrag'
        WHERE rag_mode = 'lightrag'
    """))

    # ------------------------------------------------------------------
    # 3. Create GRAPHRAG_VDB_ENTITY — stores entity embeddings produced
    #    by the graph ingestion pipeline.
    #
    #    Schema mirrors the upstream LightRAG LIGHTRAG_VDB_ENTITY table
    #    so data-api can still use the LightRAG query engine.
    # ------------------------------------------------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS "GRAPHRAG_VDB_ENTITY" (
            workspace    VARCHAR NOT NULL,
            id           VARCHAR NOT NULL,
            entity_name  VARCHAR,
            content      TEXT,
            content_vector VECTOR(3072),
            chunk_ids    VARCHAR[],
            file_path    VARCHAR,
            create_time  TIMESTAMP,
            update_time  TIMESTAMP,
            CONSTRAINT pk_graphrag_vdb_entity PRIMARY KEY (workspace, id)
        )
    """))

    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_graphrag_vdb_entity_workspace
        ON "GRAPHRAG_VDB_ENTITY" (workspace)
    """))

    # ------------------------------------------------------------------
    # 4. Create GRAPHRAG_VDB_RELATION — stores relation embeddings.
    # ------------------------------------------------------------------
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS "GRAPHRAG_VDB_RELATION" (
            workspace    VARCHAR NOT NULL,
            id           VARCHAR NOT NULL,
            source_id    VARCHAR,
            target_id    VARCHAR,
            content      TEXT,
            content_vector VECTOR(3072),
            chunk_ids    VARCHAR[],
            file_path    VARCHAR,
            create_time  TIMESTAMP,
            update_time  TIMESTAMP,
            CONSTRAINT pk_graphrag_vdb_relation PRIMARY KEY (workspace, id)
        )
    """))

    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_graphrag_vdb_relation_workspace
        ON "GRAPHRAG_VDB_RELATION" (workspace)
    """))


def downgrade() -> None:
    conn = op.get_bind()

    # Drop GraphRAG tables
    conn.execute(text('DROP TABLE IF EXISTS "GRAPHRAG_VDB_RELATION"'))
    conn.execute(text('DROP TABLE IF EXISTS "GRAPHRAG_VDB_ENTITY"'))

    # Revert 'graphrag' rows back to 'lightrag' in knowledge_base.
    # Note: the 'graphrag' enum value cannot be removed without recreating
    # the type; the value is left in the enum but is no longer used.
    conn.execute(text("""
        UPDATE public.knowledge_base
        SET rag_mode = 'lightrag'
        WHERE rag_mode = 'graphrag'
    """))
