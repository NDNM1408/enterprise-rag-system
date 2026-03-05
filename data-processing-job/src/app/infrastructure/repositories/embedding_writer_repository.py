"""
Embedding writer repository for INGESTION operations only.
Writes embeddings directly into the merged chunk table.

NOTE: Query/search operations belong in the data-api service.
"""

import json
import logging
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


logger = logging.getLogger(__name__)


class EmbeddingWriterRepository:
    """
    Repository for writing embeddings into the chunk table.

    Provides:
    - Update embedding on an existing chunk record
    - Delete embeddings (reset) by document

    Does NOT provide:
    - Query/search operations (those are in data-api service)
    """

    def __init__(self, async_session: AsyncSession):
        self.session = async_session

    async def upsert(
        self,
        chunk_id: str,
        document_id: str,
        kb_id: str,
        embedding: List[float],
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update the embedding fields on an existing chunk record.
        The chunk row must already exist (created by preprocess_document task).

        Args:
            chunk_id: ID of the chunk
            document_id: ID of the parent document
            kb_id: ID of the knowledge base
            embedding: 1024-dimensional vector
            text: Full text content of the chunk (must match chunk.content)
            metadata: JSONB metadata (chunk position, source, etc.)
        """
        # Use CAST(...) instead of ::type — the :: shorthand after a SQLAlchemy
        # named parameter confuses asyncpg's parameter parser and causes
        # "syntax error at or near ':'".
        query = text("""
            UPDATE "chunk"
            SET
                embedding = CAST(:embedding AS vector),
                metadata  = CAST(:metadata AS jsonb),
                update_time = NOW()
            WHERE id = :chunk_id
        """)

        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'

        params = {
            'chunk_id': chunk_id,
            'embedding': embedding_str,
            'metadata': json.dumps(metadata),
        }

        try:
            await self.session.execute(query, params)
            await self.session.commit()
            logger.info(f"Updated embedding for chunk {chunk_id}")
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update embedding for chunk {chunk_id}: {e}")
            raise

    async def delete_by_document(self, document_id: str) -> int:
        """
        Clear embeddings for all chunks belonging to a document
        (resets them back to Processing so they can be retried).

        Args:
            document_id: ID of the document

        Returns:
            Number of rows updated
        """
        query = text("""
            UPDATE "chunk"
            SET
                embedding   = NULL,
                metadata    = CAST('{}' AS jsonb),
                status      = 'Processing',
                update_time = NOW()
            WHERE document_id = :document_id
        """)

        try:
            result = await self.session.execute(query, {'document_id': document_id})
            await self.session.commit()
            rows_updated = result.rowcount
            logger.info(f"Reset embeddings for {rows_updated} chunks of document {document_id}")
            return rows_updated
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to reset embeddings for document {document_id}: {e}")
            raise
