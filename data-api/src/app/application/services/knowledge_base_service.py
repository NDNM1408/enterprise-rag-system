"""Service for knowledge base management operations."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.exceptions import ValidationError
from app.infrastructure.connectors.postgres.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.infrastructure.connectors.postgres.repositories.document_repository import DocumentRepository
from app.infrastructure.connectors.postgres.repositories.chunk_repository import ChunkRepository
from app.infrastructure.clients.s3_client_service import S3ClientService
from app.application.dtos.requests.knowledge_base_request import CreateKnowledgeBaseRequest
from app.infrastructure.connectors.postgres.schema import KnowledgeBase
from app.configurations.configurations import settings

logger = logging.getLogger(__name__)


def _parse_s3_url(url: str) -> Tuple[str, str]:
    """Parse 's3://bucket/key' into (bucket, key)."""
    without_scheme = url[5:]  # strip "s3://"
    bucket, key = without_scheme.split("/", 1)
    return bucket, key


class KnowledgeBaseService:
    def __init__(
        self,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        s3_client_service: S3ClientService,
    ):
        self.knowledge_base_repository = knowledge_base_repository
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.s3_client_service = s3_client_service

    async def paging(
        self,
        page: int = 1,
        page_size: int = 10,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a paginated list of knowledge bases.

        Args:
            page: 1-indexed page number
            page_size: Items per page
            where: Optional equality filter dict
            order_by: Optional sort dict, e.g. {"create_time": "desc"}

        Returns:
            Dict with total, page, page_size, and items list
        """
        skip = (page - 1) * page_size
        items = await self.knowledge_base_repository.paging(
            skip=skip, limit=page_size, where=where, order_by=order_by
        )
        total = await self.knowledge_base_repository.count(where=where)

        # Get document counts for all knowledge bases
        kb_ids = [item.id for item in items]
        doc_counts = await self.document_repository.count_by_kb_ids(kb_ids)

        # Add document_count to each item
        items_with_count = []
        for item in items:
            item_dict = item.to_dict()
            item_dict["document_count"] = doc_counts.get(item.id, 0)
            items_with_count.append(item_dict)

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items_with_count,
        }

    async def create(self, data: CreateKnowledgeBaseRequest) -> KnowledgeBase:
        """
        Create a new knowledge base.

        Args:
            data: Creation request (embed_id defaults to 'rag-embedding-model')

        Returns:
            Created KnowledgeBase ORM object
        """
        data_dict = data.model_dump(exclude_unset=True)
        # Ensure embed_id has a default value
        if "embed_id" not in data_dict:
            data_dict["embed_id"] = "rag-embedding-model"
        return await self.knowledge_base_repository.create(**data_dict)

    async def get(self, kb_id: str) -> KnowledgeBase:
        """
        Get a knowledge base by ID.

        Args:
            kb_id: Knowledge base identifier

        Returns:
            KnowledgeBase ORM object

        Raises:
            ResourceNotFoundError: If not found
        """
        return await self.knowledge_base_repository.get(id=kb_id)

    async def update(self, kb_id: str, patch: dict) -> KnowledgeBase:
        """Partial update — writes only the keys present in ``patch``.

        ``parser_config`` is replaced wholesale (not merged) so callers must
        send the complete config they want stored. The DTO already serialises
        ``ParserConfig`` to dict for us; ``rag_mode`` and ``agentic_search``
        both round-trip naturally."""
        if not patch:
            # No-op update — just return current state.
            return await self.knowledge_base_repository.get(id=kb_id)
        await self.knowledge_base_repository.update(
            data=patch, where={"id": kb_id},
        )
        return await self.knowledge_base_repository.get(id=kb_id)

    async def delete(self, kb_id: str) -> None:
        """
        Delete a knowledge base by ID, including all related chunks, documents,
        and all S3 files in the KB folder.

        Args:
            kb_id: Knowledge base identifier

        Raises:
            ResourceNotFoundError: If not found
        """
        from app.configurations.configurations import settings

        # Delete entire KB folder from S3 (includes all documents and chunks)
        folder_prefix = f"{kb_id}/"
        try:
            deleted_count = await self.s3_client_service.delete_folder(
                settings.BUCKET_NAME,
                folder_prefix
            )
            logger.info(f"Deleted {deleted_count} S3 objects for kb {kb_id}")
        except Exception as exc:
            logger.error(f"Failed to delete S3 folder for kb {kb_id}: {exc}")
            # Continue with DB deletion even if S3 deletion fails

        # Delete all chunks from DB (includes embeddings)
        await self.chunk_repository.delete_by_kb_id(kb_id)

        # Delete all documents from DB
        await self.document_repository.delete_by_kb_id(kb_id)

        # Finally delete the knowledge base
        await self.knowledge_base_repository.delete(id=kb_id)

        logger.info(f"Deleted knowledge base {kb_id} with all related data and S3 files")
