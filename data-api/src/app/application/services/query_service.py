"""
Query service — handles both classic RAG and graph RAG queries.

Detects the knowledge base type (classic vs graphrag) and routes to the
appropriate search implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from app.configurations.configurations import settings
from app.configurations.dependencies import get_knowledge_base_repository
from app.exceptions import DatabaseError, ResourceNotFoundError, ValidationError
from app.infrastructure.clients.embedding_client import EmbeddingClient
from app.infrastructure.connectors.postgres.repositories.chunk_repository import ChunkRepository
from app.infrastructure.connectors.postgres.repositories.knowledge_base_repository import (
    KnowledgeBaseRepository,
)
from app.infrastructure.connectors.postgres.database import db_session
from app.infrastructure.graph.graph_query_engine import GraphQueryEngine
from app.infrastructure.graph.llm_client import LLMClient
from app.infrastructure.graph.neo4j_store import Neo4jStore
from app.infrastructure.repositories.document_embeddings_repository import DocumentEmbeddingsRepository
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class QueryService:
    """
    Query service that handles both classic RAG and graph RAG queries.

    Automatically detects the KB type based on parser_config.rag_mode and
    routes to the appropriate search implementation.
    """

    def __init__(
        self,
        knowledge_base_repository: KnowledgeBaseRepository,
        graph_query_engine: GraphQueryEngine | None = None,
    ):
        self.knowledge_base_repository = knowledge_base_repository
        self._graph_engine = graph_query_engine

        # Classic RAG components
        self._classic_engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=20,
            max_overflow=10,
            pool_timeout=60
        )
        self._classic_session_factory = sessionmaker(
            self._classic_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        self._embedding_client = EmbeddingClient()

    async def query(
        self,
        kb_id: str,
        query_text: str,
        # Common parameters
        top_k: int = 10,
        # Classic RAG parameters
        search_type: Literal["semantic", "hybrid", "fuzzy"] = "semantic",
        alpha: float = 0.5,
        # Graph RAG parameters
        mode: Literal["local", "global", "hybrid", "naive", "mix"] = "hybrid",
        chunk_top_k: int = 10,
        max_entity_tokens: int = 4000,
        max_relation_tokens: int = 4000,
        max_total_tokens: int = 16000,
    ) -> Dict[str, Any]:
        """
        Execute a query on the knowledge base.

        Detects the KB type (classic vs graphrag) and routes accordingly.
        Returns retrieved context/chunks only (no LLM-generated answer).

        Args:
            kb_id: Knowledge base UUID.
            query_text: Raw query string.
            top_k: Number of results to return.
            search_type: Search type for classic RAG (semantic/hybrid/fuzzy).
            alpha: Weight for vector similarity in hybrid search.
            mode: GraphRAG retrieval mode (local/global/hybrid/naive/mix).
            chunk_top_k: Number of chunks to retrieve in graph RAG.
            max_entity_tokens: Max tokens for entity context.
            max_relation_tokens: Max tokens for relation context.
            max_total_tokens: Max total tokens for context.

        Returns:
            dict with query results and metadata.

        Raises:
            ValidationError: If inputs are invalid.
            ResourceNotFoundError: If KB does not exist.
            DatabaseError: If the query fails.
        """
        if not query_text or not query_text.strip():
            raise ValidationError("query_text cannot be empty")

        # Verify KB exists and get its type
        kb = await self.knowledge_base_repository.get(id=kb_id)
        if not kb:
            raise ResourceNotFoundError(f"Knowledge base {kb_id} not found")

        rag_mode = (kb.parser_config or {}).get("rag_mode", "classic")

        if rag_mode == "graphrag":
            return await self._query_graph(
                kb_id=kb_id,
                query_text=query_text,
                top_k=top_k,
                mode=mode,
                chunk_top_k=chunk_top_k,
                max_entity_tokens=max_entity_tokens,
                max_relation_tokens=max_relation_tokens,
                max_total_tokens=max_total_tokens,
            )
        else:
            return await self._query_classic(
                kb_id=kb_id,
                query_text=query_text,
                top_k=top_k,
                search_type=search_type,
                alpha=alpha,
            )

    async def _query_classic(
        self,
        kb_id: str,
        query_text: str,
        top_k: int,
        search_type: str,
        alpha: float,
    ) -> Dict[str, Any]:
        """Execute classic RAG query (semantic/hybrid/fuzzy)."""
        try:
            query_embedding = await self._embedding_client.get_embedding(query_text)
        except Exception as e:
            logger.error(f"Failed to embed query for kb {kb_id}: {e}", exc_info=True)
            raise DatabaseError(f"Embedding service error: {str(e)}")

        async with self._classic_session_factory() as session:
            try:
                repository = DocumentEmbeddingsRepository(session)

                if search_type == "semantic":
                    results = await repository.query_by_vector(
                        kb_id=kb_id,
                        query_embedding=query_embedding,
                        top_k=top_k,
                    )
                elif search_type == "hybrid":
                    results = await repository.hybrid_search(
                        kb_id=kb_id,
                        query_embedding=query_embedding,
                        query_text=query_text,
                        top_k=top_k,
                        alpha=alpha
                    )
                elif search_type == "fuzzy":
                    results = await repository.fuzzy_search(
                        kb_id=kb_id,
                        query_text=query_text,
                        top_k=top_k,
                    )
                else:
                    raise ValidationError(f"Invalid search_type: {search_type}")

                return {
                    "kb_id": kb_id,
                    "query_type": "classic",
                    "search_type": search_type,
                    "query_text": query_text,
                    "results": results,
                    "result_count": len(results),
                }
            except ValidationError:
                raise
            except Exception as e:
                logger.error(f"Classic search failed for kb {kb_id}: {e}", exc_info=True)
                raise DatabaseError(f"Search failed: {str(e)}")

    async def _query_graph(
        self,
        kb_id: str,
        query_text: str,
        top_k: int,
        mode: str,
        chunk_top_k: int,
        max_entity_tokens: int,
        max_relation_tokens: int,
        max_total_tokens: int,
    ) -> Dict[str, Any]:
        """Execute graph RAG query. Returns context only (no LLM answer)."""
        engine = await self._get_graph_engine()

        try:
            result = await engine.query(
                kb_id=kb_id,
                query_text=query_text,
                mode=mode,
                top_k=top_k,
                only_context=True,  # Always return context only
                chunk_top_k=chunk_top_k,
                max_entity_tokens=max_entity_tokens,
                max_relation_tokens=max_relation_tokens,
                max_total_tokens=max_total_tokens,
            )
        except Exception as exc:
            logger.error(
                "GraphRAG query failed for kb=%s mode=%s: %s",
                kb_id, mode, exc, exc_info=True,
            )
            raise DatabaseError(f"GraphRAG query failed: {exc}") from exc

        return {
            "kb_id": kb_id,
            "query_type": "graph",
            "mode": mode,
            "query_text": query_text,
            "context": result.get("context", ""),
            "keywords": result.get("keywords", {}),
            "entity_count": result.get("entity_count", 0),
            "relation_count": result.get("relation_count", 0),
            "chunk_count": result.get("chunk_count", 0),
        }

    async def _get_graph_engine(self) -> GraphQueryEngine:
        """Get or create the graph query engine."""
        if self._graph_engine is None:
            neo4j_store = Neo4jStore(
                uri=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD,
                database=settings.NEO4J_DATABASE,
            )
            llm_client = LLMClient(
                api_base=settings.GRAPHRAG_LLM_API_BASE,
                model=settings.GRAPHRAG_LLM_MODEL,
                api_key=settings.GRAPHRAG_LLM_API_KEY,
            )
            session_factory = db_session.get_session()

            self._graph_engine = GraphQueryEngine(
                neo4j_store=neo4j_store,
                embedding_client=self._embedding_client,
                llm_client=llm_client,
                session_factory=session_factory,
                chunk_repository=ChunkRepository(),
            )
        return self._graph_engine

    async def close(self):
        """Close database connections."""
        await self._classic_engine.dispose()


async def get_query_service() -> QueryService:
    """Factory function to create a QueryService instance."""
    kb_repo = await get_knowledge_base_repository()
    return QueryService(knowledge_base_repository=kb_repo)
