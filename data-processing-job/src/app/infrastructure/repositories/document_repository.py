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
