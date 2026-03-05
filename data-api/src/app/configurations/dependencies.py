from fastapi import Depends

import httpx
from app.infrastructure.clients.s3_client_service import S3ClientService
from app.infrastructure.connectors.rabbitmq.rabbitmq import RabbitMQService
from app.infrastructure.connectors.postgres.repositories.document_repository import DocumentRepository
from app.infrastructure.connectors.postgres.repositories.knowledge_base_repository import KnowledgeBaseRepository
from app.infrastructure.connectors.postgres.repositories.chunk_repository import ChunkRepository
from app.application.services.document_service import DocumentsService
from app.application.services.knowledge_base_service import KnowledgeBaseService

async def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient()

async def get_s3_client_service() -> S3ClientService:
    return S3ClientService()

async def get_rabbitmq_service() -> RabbitMQService:
    return RabbitMQService()

async def get_document_repository() -> DocumentRepository:
    return DocumentRepository()

async def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository()

async def get_knowledge_base_repository() -> KnowledgeBaseRepository:
    return KnowledgeBaseRepository()

async def get_knowledge_base_service(
    knowledge_base_repository: KnowledgeBaseRepository = Depends(get_knowledge_base_repository),
    document_repository: DocumentRepository = Depends(get_document_repository),
    chunk_repository: ChunkRepository = Depends(get_chunk_repository),
    s3_client_service: S3ClientService = Depends(get_s3_client_service),
) -> KnowledgeBaseService:
    return KnowledgeBaseService(
        knowledge_base_repository,
        document_repository,
        chunk_repository,
        s3_client_service,
    )

async def get_document_service(
    knowledge_base_repository: KnowledgeBaseRepository = Depends(get_knowledge_base_repository),
    document_repository: DocumentRepository = Depends(get_document_repository),
    s3_client_service: S3ClientService = Depends(get_s3_client_service),
    chunk_repository: ChunkRepository = Depends(get_chunk_repository),
) -> DocumentsService:
    return DocumentsService(
        knowledge_base_repository,
        document_repository,
        s3_client_service,
        chunk_repository,
    )

