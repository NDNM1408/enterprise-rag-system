"""
Repository for chunk table reads/writes.

All methods use the shared session_factory (NullPool engine) and open a new
session per operation.  SQL strings are kept here so task code stays thin.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


class ChunkRepository:
    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    async def get_status(self, chunk_id: str) -> Optional[str]:
        """Return chunk status, or None if the chunk is not found."""
        async with self._sf() as session:
            result = await session.execute(
                text("SELECT status FROM chunk WHERE id = :id"),
                {"id": chunk_id},
            )
            row = result.one_or_none()
            return row[0] if row else None

    async def set_status(self, chunk_id: str, status: str) -> None:
        """Update a single chunk's status.  Commits immediately."""
        async with self._sf() as session:
            async with session.begin():
                await session.execute(
                    text("UPDATE chunk SET status = :status WHERE id = :id"),
                    {"status": status, "id": chunk_id},
                )
        logger.debug("chunk=%s status→%s", chunk_id, status)

    async def count_non_succeeded(self, document_id: str) -> int:
        """Return the number of chunks for a document that are not 'Succeed'."""
        async with self._sf() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) FROM chunk "
                    "WHERE document_id = :document_id AND status != 'Succeed'"
                ),
                {"document_id": document_id},
            )
            return result.scalar() or 0

    async def get_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Return all chunks for a document as dicts with id, content,
        parent_text, status, heading_path."""
        async with self._sf() as session:
            result = await session.execute(
                text(
                    "SELECT id, content, parent_text, status, heading_path "
                    "FROM chunk WHERE document_id = :document_id"
                ),
                {"document_id": document_id},
            )
            return [dict(row._mapping) for row in result]

    async def batch_insert(self, records: List[Dict[str, Any]]) -> None:
        """
        Batch-insert chunk records in a single statement.

        Required keys: id, content, document_id, kb_id, doc_name, status.
        Optional: parent_text, chunk_s3_url, heading_path, token_count.

        Every chunk is a retrieve chunk in the denormalized model — the
        full enclosing section travels inline as ``parent_text`` so the
        LLM context query becomes a single-row read, no FK follow.
        """
        if not records:
            return

        cols = (
            "id", "content", "parent_text", "document_id", "kb_id", "doc_name",
            "status", "heading_path", "token_count", "chunk_s3_url",
        )
        placeholders = ", ".join(
            "(" + ", ".join(f":{c}_{i}" for c in cols) + ")"
            for i in range(len(records))
        )
        params: Dict[str, Any] = {}
        for i, rec in enumerate(records):
            params[f"id_{i}"] = rec["id"]
            params[f"content_{i}"] = rec["content"]
            params[f"parent_text_{i}"] = rec.get("parent_text")
            params[f"document_id_{i}"] = rec["document_id"]
            params[f"kb_id_{i}"] = rec["kb_id"]
            params[f"doc_name_{i}"] = rec["doc_name"]
            params[f"status_{i}"] = rec.get("status", "Processing")
            params[f"heading_path_{i}"] = rec.get("heading_path")
            params[f"token_count_{i}"] = rec.get("token_count")
            params[f"chunk_s3_url_{i}"] = rec.get("chunk_s3_url")

        query = text(
            "INSERT INTO chunk "
            f"({', '.join(cols)}) "
            f"VALUES {placeholders}"
        )
        async with self._sf() as session:
            async with session.begin():
                await session.execute(query, params)
        logger.debug("batch_insert: inserted %d chunks", len(records))
