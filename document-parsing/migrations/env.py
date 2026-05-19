"""Alembic environment for document-parsing-service.

Reuses the running postgres (env DATABASE_URL). Schema is set per-table on
the SQLAlchemy models so migrations live alongside the existing ``datahub``
DB without colliding with data-api's tables.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make ``src`` importable so we can pull in the SQLAlchemy metadata.
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "..", "src"))

from infrastructure.db import Base  # noqa: E402

config = context.config


def _sync_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


db_url = os.getenv("DATABASE_URL", "")
if db_url:
    config.set_main_option("sqlalchemy.url", _sync_url(db_url))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    """Only manage objects in the ``parsing`` schema."""
    if type_ == "table" and obj.schema != "parsing":
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        include_object=include_object,
        version_table_schema="parsing",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        connection.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS parsing")
        connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            include_object=include_object,
            version_table_schema="parsing",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
