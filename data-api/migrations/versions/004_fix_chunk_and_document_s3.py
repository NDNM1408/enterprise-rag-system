"""move s3/etag fields: drop from chunk, add to document

Revision ID: 004
Revises: 003
Create Date: 2026-02-19

"""
from alembic import op
from sqlalchemy import text


revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Remove incorrectly placed columns from chunk
    conn.execute(text('ALTER TABLE "chunk" DROP COLUMN IF EXISTS document_s3_url'))
    conn.execute(text('ALTER TABLE "chunk" DROP COLUMN IF EXISTS document_etag'))

    # Add s3_url and etag to document
    conn.execute(text('ALTER TABLE "document" ADD COLUMN IF NOT EXISTS s3_url VARCHAR'))
    conn.execute(text('ALTER TABLE "document" ADD COLUMN IF NOT EXISTS etag VARCHAR'))


def downgrade() -> None:
    conn = op.get_bind()

    conn.execute(text('ALTER TABLE "document" DROP COLUMN IF EXISTS etag'))
    conn.execute(text('ALTER TABLE "document" DROP COLUMN IF EXISTS s3_url'))

    conn.execute(text('ALTER TABLE "chunk" ADD COLUMN IF NOT EXISTS document_etag VARCHAR'))
    conn.execute(text('ALTER TABLE "chunk" ADD COLUMN IF NOT EXISTS document_s3_url VARCHAR'))
