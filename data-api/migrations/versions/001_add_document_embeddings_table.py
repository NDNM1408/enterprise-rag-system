"""merge chunk and document_embeddings into single chunk table

Revision ID: 001
Revises: 000
Create Date: 2026-02-19

"""
from alembic import op
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision = '001'
down_revision = '000'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Enable pgvector extension
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Drop old basic chunk table created in 000 (no data yet)
    conn.execute(text('DROP TABLE IF EXISTS "chunk" CASCADE'))

    # Create merged chunk table (combines chunk + document_embeddings)
    conn.execute(text("""
        CREATE TABLE "chunk" (
            id VARCHAR PRIMARY KEY,
            document_id VARCHAR REFERENCES "document"(id) ON DELETE CASCADE,
            kb_id VARCHAR REFERENCES "knowledge_base"(id) ON DELETE CASCADE,
            doc_name VARCHAR,
            content TEXT,
            status "ChunkStatus" NOT NULL,

            -- S3 location for this chunk (for retry support)
            chunk_s3_url VARCHAR,

            -- Vector search fields (populated by upsert_chunk task)
            embedding vector(1024),
            metadata JSONB NOT NULL DEFAULT '{}',

            create_time TIMESTAMP NOT NULL DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW()
        )
    """))

    # Generated full-text search column
    conn.execute(text("""
        ALTER TABLE "chunk"
        ADD COLUMN text_tsvector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED
    """))

    # HNSW index for vector similarity search
    conn.execute(text("""
        CREATE INDEX idx_chunk_embedding_hnsw
        ON "chunk"
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """))

    # GIN index on metadata JSONB for filtering
    conn.execute(text("""
        CREATE INDEX idx_chunk_metadata
        ON "chunk"
        USING GIN (metadata jsonb_path_ops)
    """))

    # GIN index for full-text search
    conn.execute(text("""
        CREATE INDEX idx_chunk_fulltext
        ON "chunk"
        USING GIN (text_tsvector)
    """))

    # Index for document_id lookups
    conn.execute(text("""
        CREATE INDEX idx_chunk_document_id ON "chunk" (document_id)
    """))

    # Index for kb_id lookups (used by vector search queries)
    conn.execute(text("""
        CREATE INDEX idx_chunk_kb_id ON "chunk" (kb_id)
    """))

    # Index for retrying failed chunks
    conn.execute(text("""
        CREATE INDEX idx_chunk_status ON "chunk" (status)
        WHERE status = 'Failed'
    """))


def downgrade() -> None:
    conn = op.get_bind()

    # Drop merged chunk table
    conn.execute(text('DROP TABLE IF EXISTS "chunk" CASCADE'))

    # Restore original basic chunk table from 000
    conn.execute(text("""
        CREATE TABLE "chunk" (
            id VARCHAR PRIMARY KEY,
            content TEXT,
            document_id VARCHAR REFERENCES "document"(id) ON DELETE CASCADE,
            doc_name VARCHAR,
            status "ChunkStatus" NOT NULL,
            create_time TIMESTAMP NOT NULL DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW()
        )
    """))
