"""
Repository for document table reads/writes.

Each method opens its own session from the shared session_factory (NullPool
engine), executes one logical operation, and closes the session.  This keeps
transactions short and avoids holding connections across await boundaries.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


class DocumentRepository:
    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    async def get_status(self, document_id: str) -> Optional[str]:
        """Return the document's current status string, or None if not found."""
        async with self._sf() as session:
            result = await session.execute(
                text("SELECT status FROM document WHERE id = :id"),
                {"id": document_id},
            )
            row = result.one_or_none()
            return row[0] if row else None

    async def set_status(self, document_id: str, status: str) -> None:
        """Update the document status.  Commits immediately."""
        async with self._sf() as session:
            async with session.begin():
                await session.execute(
                    text("UPDATE document SET status = :status WHERE id = :id"),
                    {"status": status, "id": document_id},
                )
        logger.debug("doc=%s status→%s", document_id, status)

    async def set_ingesting(
        self,
        document_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
    ) -> None:
        """Update ingesting_status / ingesting_progress (one or both).

        Coalesce-on-NULL: when one argument is omitted, the existing column
        value is preserved. ``progress`` is clamped to [0, 100].
        """
        if status is None and progress is None:
            return
        if progress is not None:
            progress = max(0, min(100, int(progress)))
        async with self._sf() as session:
            async with session.begin():
                await session.execute(
                    text(
                        "UPDATE document SET "
                        "    ingesting_status   = COALESCE(:status,   ingesting_status), "
                        "    ingesting_progress = COALESCE(:progress, ingesting_progress) "
                        "WHERE id = :id"
                    ),
                    {"status": status, "progress": progress, "id": document_id},
                )

    async def recompute_ingesting_progress(self, document_id: str) -> int:
        """Recalculate ingesting_progress from chunk counts and write it back.

        Returns the new percentage (0-100). Atomic with a single SQL statement
        so concurrent upsert_chunk tasks don't race against each other.
        """
        async with self._sf() as session:
            async with session.begin():
                result = await session.execute(
                    text(
                        """
                        WITH counts AS (
                            SELECT
                                COUNT(*)::int                                       AS total,
                                COUNT(*) FILTER (WHERE status = 'Succeed')::int    AS succeeded
                            FROM chunk
                            WHERE document_id = :doc_id
                        )
                        UPDATE document
                        SET ingesting_progress = CASE
                            WHEN counts.total = 0 THEN ingesting_progress
                            ELSE GREATEST(
                                ingesting_progress,
                                (counts.succeeded * 100 / counts.total)::int
                            )
                        END
                        FROM counts
                        WHERE document.id = :doc_id
                        RETURNING document.ingesting_progress
                        """
                    ),
                    {"doc_id": document_id},
                )
                row = result.first()
                return int(row[0]) if row else 0

    async def finalize_ingesting(
        self,
        document_id: str,
        *,
        success: bool,
    ) -> None:
        """Mark ingesting_status / ingesting_progress / rollup status atomically.

        Rollup logic for the legacy ``status`` column:
            ingesting Succeed + parsing not Failed → status='Succeed'
            else → status='Failed'
        """
        async with self._sf() as session:
            async with session.begin():
                await session.execute(
                    text(
                        """
                        UPDATE document
                        SET ingesting_status   = :ing_status,
                            ingesting_progress = 100,
                            status = CASE
                                WHEN :success AND parsing_status <> 'Failed' THEN 'Succeed'
                                ELSE 'Failed'
                            END
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": document_id,
                        "ing_status": "Succeed" if success else "Failed",
                        "success": success,
                    },
                )
