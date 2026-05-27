"""Repository for the per-parent extracted-fact cache.

Facts are extracted once per ``parent_id`` (query-independent) and reused
across every query that retrieves the parent. The chat path looks up cached
facts, extracts only the misses, then persists them here.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from sqlalchemy import text

from app.infrastructure.connectors.postgres.database import db_session

logger = logging.getLogger(__name__)


class ParentFactsRepository:
    """Read/write the ``parent_facts`` cache table."""

    def __init__(self) -> None:
        self.async_session = db_session.get_session()

    async def get_many(self, parent_ids: List[str]) -> Dict[str, List[str]]:
        """Return ``{parent_id: facts}`` for the parent_ids that are cached.

        Missing parent_ids are simply absent from the result."""
        if not parent_ids:
            return {}
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    text(
                        "SELECT parent_id, facts FROM parent_facts "
                        "WHERE parent_id = ANY(:ids)"
                    ),
                    {"ids": list(parent_ids)},
                )
                rows = result.fetchall()
            out: Dict[str, List[str]] = {}
            for row in rows:
                facts = row.facts
                if isinstance(facts, list):
                    out[str(row.parent_id)] = facts
            return out
        except Exception as exc:
            # Cache is best-effort: a read failure should degrade to
            # extract-everything, never break the chat turn.
            logger.warning("parent_facts read failed: %s", exc)
            return {}

    async def save_many(self, records: List[Dict]) -> None:
        """Upsert fact rows. Each record: {parent_id, kb_id, document_id, facts}."""
        if not records:
            return
        try:
            async with self.async_session() as session:
                async with session.begin():
                    for rec in records:
                        await session.execute(
                            text(
                                """
                                INSERT INTO parent_facts
                                    (parent_id, kb_id, document_id, facts)
                                VALUES
                                    (:parent_id, :kb_id, :document_id,
                                     CAST(:facts AS jsonb))
                                ON CONFLICT (parent_id) DO UPDATE
                                    SET facts = EXCLUDED.facts,
                                        kb_id = EXCLUDED.kb_id,
                                        document_id = EXCLUDED.document_id
                                """
                            ),
                            {
                                "parent_id": rec["parent_id"],
                                "kb_id": rec.get("kb_id"),
                                "document_id": rec.get("document_id"),
                                "facts": rec["facts_json"],
                            },
                        )
        except Exception as exc:
            # Persistence is best-effort — a write failure just means the next
            # query re-extracts. Don't fail the chat turn.
            logger.warning("parent_facts write failed: %s", exc)
