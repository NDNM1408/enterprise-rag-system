"""create parsing.parsing_job

Revision ID: 001_parsing_job
Revises:
Create Date: 2026-05-06 17:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_parsing_job"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS parsing")
    op.create_table(
        "parsing_job",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("state", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("parser", sa.String(64)),
        sa.Column("mode", sa.String(64)),
        sa.Column("pages_total", sa.Integer()),
        sa.Column("pages_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("s3_input_key", sa.Text()),
        sa.Column("s3_markdown_key", sa.Text()),
        sa.Column("s3_image_prefix", sa.Text()),
        sa.Column("image_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.BigInteger()),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb")),
        sa.Column("error", sa.Text()),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        schema="parsing",
    )
    op.create_index(
        "ix_parsing_job_state",
        "parsing_job",
        ["state"],
        schema="parsing",
    )
    op.create_index(
        "ix_parsing_job_submitted_at",
        "parsing_job",
        [sa.text("submitted_at DESC")],
        schema="parsing",
    )


def downgrade() -> None:
    op.drop_index("ix_parsing_job_submitted_at", table_name="parsing_job", schema="parsing")
    op.drop_index("ix_parsing_job_state", table_name="parsing_job", schema="parsing")
    op.drop_table("parsing_job", schema="parsing")
    # Don't drop schema — other migrations might use it.
