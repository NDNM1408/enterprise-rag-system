"""Query service — routes by KB ``rag_mode``.

Modes:
  - ``classic``  — pgvector semantic / hybrid / fuzzy search on the ``chunk`` table.
  - ``llm-wiki`` — Elasticsearch BM25 + kNN with client-side RRF fusion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.configurations.configurations import settings
from app.configurations.dependencies import get_knowledge_base_repository
from app.exceptions import DatabaseError, ResourceNotFoundError, ValidationError
from app.infrastructure.clients.embedding_client import EmbeddingClient
from app.infrastructure.connectors.postgres.repositories.knowledge_base_repository import (
    KnowledgeBaseRepository,
)
from app.infrastructure.repositories.document_embeddings_repository import (
    DocumentEmbeddingsRepository,
)
from app.infrastructure.search.es_search_service import ElasticsearchSearchService

logger = logging.getLogger(__name__)


def _dedupe_by_parent(raw: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Walk the score-ordered raw rows; collapse children sharing a parent
    into one entry. Falls back to ``parent_text`` hash when ``parent_id`` is
    NULL (legacy chunks ingested before v4).

    Returns ≤ ``top_k`` rows — fewer if many children share parents."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for row in raw:
        pid = row.get("parent_id")
        if pid:
            key = ("pid", pid)
        else:
            pt = row.get("parent_text") or row.get("content") or row.get("text") or ""
            if not pt:
                continue
            key = ("ptxt", hash(pt))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= top_k:
            break
    return out


class QueryService:
    """Dispatch by ``parser_config.rag_mode`` set on the knowledge base."""

    def __init__(
        self,
        knowledge_base_repository: KnowledgeBaseRepository,
    ):
        self.knowledge_base_repository = knowledge_base_repository

        # Pgvector engine for the classic path. NullPool would be safer in a
        # forked-worker context, but data-api is async-only so a small pool is
        # fine here.
        self._classic_engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=20,
            max_overflow=10,
            pool_timeout=60,
        )
        self._classic_session_factory = sessionmaker(
            self._classic_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._embedding_client = EmbeddingClient()
        self._es_search: ElasticsearchSearchService | None = None

    async def query(
        self,
        kb_id: str,
        query_text: str,
        top_k: int = 10,
        search_type: Literal["semantic", "hybrid", "fuzzy"] = "semantic",
        alpha: float = 0.5,
        **_: Any,
    ) -> Dict[str, Any]:
        """Execute a query on the knowledge base — auto-routed by mode."""
        if not query_text or not query_text.strip():
            raise ValidationError("query_text cannot be empty")

        kb = await self.knowledge_base_repository.get(id=kb_id)
        if not kb:
            raise ResourceNotFoundError(f"Knowledge base {kb_id} not found")

        rag_mode = (kb.parser_config or {}).get("rag_mode", "classic")

        if rag_mode == "llm-wiki":
            return await self._query_llm_wiki(kb_id=kb_id, query_text=query_text, top_k=top_k)
        return await self._query_classic(
            kb_id=kb_id,
            query_text=query_text,
            top_k=top_k,
            search_type=search_type,
            alpha=alpha,
        )

    # ------------------------------------------------------------------
    # classic — pgvector
    # ------------------------------------------------------------------

    async def _query_classic(
        self,
        kb_id: str,
        query_text: str,
        top_k: int,
        search_type: str,
        alpha: float,
    ) -> Dict[str, Any]:
        try:
            query_embedding = await self._embedding_client.get_embedding(query_text)
        except Exception as exc:
            logger.error("Embedding failed for kb %s: %s", kb_id, exc, exc_info=True)
            raise DatabaseError(f"Embedding service error: {exc}")

        # v4: over-fetch children so dedupe-by-parent still leaves a usable
        # top-N after sibling collapse.
        fetch_k = max(top_k * 3, top_k)

        async with self._classic_session_factory() as session:
            repository = DocumentEmbeddingsRepository(session)
            try:
                if search_type == "semantic":
                    raw = await repository.query_by_vector(
                        kb_id=kb_id, query_embedding=query_embedding, top_k=fetch_k,
                    )
                elif search_type == "hybrid":
                    raw = await repository.hybrid_search(
                        kb_id=kb_id, query_embedding=query_embedding,
                        query_text=query_text, top_k=fetch_k, alpha=alpha,
                    )
                elif search_type == "fuzzy":
                    raw = await repository.fuzzy_search(
                        kb_id=kb_id, query_text=query_text, top_k=fetch_k,
                    )
                else:
                    raise ValidationError(f"Invalid search_type: {search_type}")
            except ValidationError:
                raise
            except Exception as exc:
                logger.error("Classic search failed for kb %s: %s", kb_id, exc, exc_info=True)
                raise DatabaseError(f"Search failed: {exc}")

        results = _dedupe_by_parent(raw, top_k)

        return {
            "kb_id": kb_id,
            "query_type": "classic",
            "search_type": search_type,
            "query_text": query_text,
            "results": results,
            "result_count": len(results),
        }

    # ------------------------------------------------------------------
    # llm-wiki — Elasticsearch hybrid
    # ------------------------------------------------------------------

    async def _query_llm_wiki(
        self,
        kb_id: str,
        query_text: str,
        top_k: int,
    ) -> Dict[str, Any]:
        try:
            query_embedding = await self._embedding_client.get_embedding(query_text)
        except Exception as exc:
            logger.error("Embedding failed for kb %s: %s", kb_id, exc, exc_info=True)
            raise DatabaseError(f"Embedding service error: {exc}")

        es = self._get_es_service()
        try:
            results = await es.hybrid_search(
                kb_id=kb_id,
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
            )
        except Exception as exc:
            logger.error("llm-wiki search failed for kb %s: %s", kb_id, exc, exc_info=True)
            raise DatabaseError(f"Elasticsearch query failed: {exc}")

        return {
            "kb_id": kb_id,
            "query_type": "llm-wiki",
            "query_text": query_text,
            "results": results,
            "result_count": len(results),
        }

    def _get_es_service(self) -> ElasticsearchSearchService:
        if self._es_search is None:
            self._es_search = ElasticsearchSearchService()
        return self._es_search

    async def close(self):
        await self._classic_engine.dispose()
        if self._es_search is not None:
            await self._es_search.close()


async def get_query_service() -> QueryService:
    kb_repo = await get_knowledge_base_repository()
    return QueryService(knowledge_base_repository=kb_repo)
