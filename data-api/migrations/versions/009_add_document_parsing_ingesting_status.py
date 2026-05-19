"""split document processing into parsing + ingesting phases with progress tracking

Revision ID: 009
Revises: 008
Create Date: 2026-05-20

Adds dedicated parsing & ingesting tracking columns to the `document` table so the
processing pipeline can be split into two visible phases for the UI:

  • parsing_status / parsing_progress  - document-parsing service phase
  • ingesting_status / ingesting_progress - data-processing-job (chunk + embed) phase

The legacy `status` column is kept as a rollup (Created -> Processing -> Succeed/Failed)
for backward compatibility with existing readers.
"""
from alembic import op
from sqlalchemy import text


revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(text("""
        ALTER TABLE "document"
            ADD COLUMN IF NOT EXISTS parsing_status VARCHAR(32) NOT NULL DEFAULT 'Pending',
            ADD COLUMN IF NOT EXISTS parsing_progress INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS parsing_job_id VARCHAR,
            ADD COLUMN IF NOT EXISTS parsed_markdown_s3_key VARCHAR,
            ADD COLUMN IF NOT EXISTS parsing_error TEXT,
            ADD COLUMN IF NOT EXISTS ingesting_status VARCHAR(32) NOT NULL DEFAULT 'Pending',
            ADD COLUMN IF NOT EXISTS ingesting_progress INTEGER NOT NULL DEFAULT 0
    """))

    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_document_parsing_job_id
            ON "document" (parsing_job_id)
    """))

    # Backfill: any existing document already in a terminal state should reflect
    # that on both phase columns so the UI doesn't show "Pending" for old rows.
    conn.execute(text("""
        UPDATE "document"
        SET parsing_status = 'Parsed',
            parsing_progress = 100,
            ingesting_status = CASE
                WHEN status::text = 'Succeed' THEN 'Succeed'
                WHEN status::text = 'Failed'  THEN 'Failed'
                ELSE 'Pending'
            END,
            ingesting_progress = CASE
                WHEN status::text = 'Succeed' THEN 100
                ELSE 0
            END
        WHERE status::text IN ('Succeed', 'Failed')
    """))


def downgrade() -> None:
    conn = op.get_bind()

    conn.execute(text('DROP INDEX IF EXISTS idx_document_parsing_job_id'))
    conn.execute(text("""
        ALTER TABLE "document"
            DROP COLUMN IF EXISTS ingesting_progress,
            DROP COLUMN IF EXISTS ingesting_status,
            DROP COLUMN IF EXISTS parsing_error,
            DROP COLUMN IF EXISTS parsed_markdown_s3_key,
            DROP COLUMN IF EXISTS parsing_job_id,
            DROP COLUMN IF EXISTS parsing_progress,
            DROP COLUMN IF EXISTS parsing_status
    """))
