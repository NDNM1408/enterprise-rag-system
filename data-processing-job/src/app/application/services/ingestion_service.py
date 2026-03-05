"""
Ingestion service for writing embeddings to vector store.
This service handles ONLY ingestion operations (upsert/delete).

NOTE: Query/search operations belong in the data-api service.
"""

import logging
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.repositories.embedding_writer_repository import EmbeddingWriterRepository


logger = logging.getLogger(__name__)


class IngestionService:
    """
    Service for ingesting document embeddings into the vector store.

    Coordinates between embedding generation and storage during
    the document processing pipeline.

    Provides:
    - Upsert embeddings (insert/update)
    - Delete embeddings

    Does NOT provide:
    - Query/search operations (those are in data-api)
    """

    def __init__(self, async_session: AsyncSession):
        self.repository = EmbeddingWriterRepository(async_session)

    async def upsert_embedding(
        self,
        chunk_id: str,
        document_id: str,
        kb_id: str,
        embedding: List[float],
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Upsert an embedding into the vector store.

        Args:
            chunk_id: UUID of the chunk
            document_id: UUID of the parent document
            kb_id: UUID of the knowledge base
            embedding: 1024-dimensional embedding vector
            content: Full text content
            metadata: JSONB metadata
        """
        await self.repository.upsert(
            chunk_id=chunk_id,
            document_id=document_id,
            kb_id=kb_id,
            embedding=embedding,
            content=content,
            metadata=metadata
        )

    async def delete_document_embeddings(self, document_id: str) -> int:
        """
        Delete all embeddings for a document.

        Args:
            document_id: UUID of the document

        Returns:
            Number of embeddings deleted
        """
        return await self.repository.delete_by_document(document_id)
