"""change embedding dimension from 1024 to 3072

Revision ID: 006
Revises: 005
Create Date: 2026-02-20

"""
from alembic import op
from sqlalchemy import text


revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Drop HNSW index — must be removed before altering the vector column type
    conn.execute(text("DROP INDEX IF EXISTS idx_chunk_embedding_hnsw"))

    # Alter column to 3072 dimensions.
    # Existing 1024-dim vectors cannot be cast to 3072-dim, so clear them with USING NULL.
    conn.execute(text("""
        ALTER TABLE "chunk"
        ALTER COLUMN embedding TYPE vector(3072) USING NULL
    """))

    # Reset chunks whose embeddings were cleared so they get re-processed
    conn.execute(text("""
        UPDATE "chunk"
        SET status = 'Processing', metadata = '{}'
        WHERE status = 'Succeed'
    """))

    # Both HNSW and IVFFlat in this version of pgvector have a 2000-dimension
    # limit and cannot index 3072-dim vectors. pgvector will fall back to an
    # exact brute-force scan using the <=> operator, which is correct for all
    # dataset sizes. Upgrade pgvector to >= 0.7.0 to re-enable ANN indexing.


def downgrade() -> None:
    conn = op.get_bind()

    conn.execute(text("DROP INDEX IF EXISTS idx_chunk_embedding_ivfflat"))

    conn.execute(text("""
        ALTER TABLE "chunk"
        ALTER COLUMN embedding TYPE vector(1024) USING NULL
    """))

    conn.execute(text("""
        CREATE INDEX idx_chunk_embedding_hnsw
        ON "chunk"
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """))
