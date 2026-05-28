"""Query service — routes by KB ``rag_mode``.

Modes:
  - ``classic``  — hier_v2 pipeline: pgvector single-shot retrieve →
                   parent/table dedupe → LLM selector → return top-N.
  - ``llm-wiki`` — Elasticsearch BM25 + kNN with client-side RRF fusion.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Tuple

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.configurations.configurations import settings
from app.configurations.dependencies import get_knowledge_base_repository
from app.exceptions import DatabaseError, ResourceNotFoundError, ValidationError
from app.infrastructure.clients.embedding_client import EmbeddingClient
from app.infrastructure.clients.litellm_client import LiteLLMClient
from app.infrastructure.connectors.postgres.repositories.knowledge_base_repository import (
    KnowledgeBaseRepository,
)
from app.infrastructure.repositories.document_embeddings_repository import (
    DocumentEmbeddingsRepository,
)
from app.infrastructure.search.es_search_service import ElasticsearchSearchService
from app.application.services.agentic_search_service import AgenticSearchService

logger = logging.getLogger(__name__)


def _dedupe_chunks(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Two-pass dedupe (score-ordered input):

    1. ``parent_id`` — collapse text_child siblings of the same parent
       window, and table_segment siblings of the same table (their
       parent_id = table_id) to one representative each.
    2. ``table_id`` — pair a ``table_summary`` and its segment siblings.
       The first one seen wins (highest score per table).
    3. Fallback ``hash(parent_text)`` for legacy chunks with NULL parent_id.

    Preserves order; keeps the highest-scored representative."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for row in raw:
        keys = []
        pid = row.get("parent_id")
        tid = row.get("table_id")
        if pid:
            keys.append(("pid", pid))
        if tid:
            keys.append(("tid", tid))
        if not keys:
            pt = row.get("parent_text") or row.get("content") or row.get("text") or ""
            if not pt:
                continue
            keys.append(("ptxt", hash(pt)))
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        out.append(row)
    return out


def _strip_json_fence(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s


async def _llm_select(
    llm: LiteLLMClient,
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Ask gemini-2.5-flash to pick the top_n most relevant chunks.

    Returns ``(selected, fallback)``. ``fallback=True`` means the LLM
    response couldn't be parsed and we returned the score-order prefix."""
    if not candidates:
        return [], False
    if len(candidates) <= top_n:
        return candidates, False

    # Build a numbered candidate list — preview the content (≤ 280 chars).
    lines: List[str] = []
    for i, r in enumerate(candidates):
        preview = (r.get("text") or r.get("parent_text") or "").strip()
        preview = " ".join(preview.split())[:280]
        src = r.get("doc_name") or "?"
        sec = (r.get("heading_path") or "")[:60]
        lines.append(f"[{i}] ({src} · {sec}) {preview}")

    system = (
        "You are a chunk relevance filter for a RAG pipeline. Given a user "
        "query and a numbered list of candidate chunks, output the indices "
        f"of the {top_n} chunks most likely to help answer the query, "
        "ordered by relevance (most relevant first). Output STRICT JSON only."
    )
    user = (
        f"User query:\n{query}\n\n"
        f"Candidate chunks ({len(candidates)} total):\n"
        + "\n".join(lines)
        + "\n\nOutput JSON only:\n"
        f'{{"selected_indices": [int, ...]}}  '
        f"— exactly {top_n} indices, most-relevant first, no duplicates, "
        f"all in [0, {len(candidates) - 1}]."
    )

    try:
        raw = await llm.chat(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            model=settings.SELECTOR_MODEL,
            temperature=0.0,
            enable_thinking=False,
        )
        data = json.loads(_strip_json_fence(raw))
        indices = data.get("selected_indices") or []
        seen: set = set()
        picks: List[Dict[str, Any]] = []
        for ix in indices:
            if not isinstance(ix, int) or ix < 0 or ix >= len(candidates):
                continue
            if ix in seen:
                continue
            seen.add(ix)
            picks.append(candidates[ix])
            if len(picks) >= top_n:
                break
        if picks:
            return picks, False
    except Exception as exc:
        logger.warning("LLM selector failed (%s); falling back to score order", exc)

    return candidates[:top_n], True


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
        self._llm = LiteLLMClient()
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

        parser_config = kb.parser_config or {}
        rag_mode = parser_config.get("rag_mode", "classic")
        agentic = bool(parser_config.get("agentic_search", False))
        # Per-KB knob overrides. None → fall back to global defaults.
        kb_top_n = parser_config.get("top_n")
        kb_max_iter = parser_config.get("agentic_max_iter")
        kb_per_iter_k = parser_config.get("agentic_top_k_per_iter")

        if rag_mode == "llm-wiki":
            return await self._query_llm_wiki(kb_id=kb_id, query_text=query_text, top_k=top_k)
        if agentic:
            return await self._query_agentic(
                kb_id=kb_id, query_text=query_text, top_k=top_k,
                top_n_override=kb_top_n,
                max_iter_override=kb_max_iter,
                per_iter_k_override=kb_per_iter_k,
            )
        return await self._query_classic(
            kb_id=kb_id,
            query_text=query_text,
            top_k=top_k,
            search_type=search_type,
            alpha=alpha,
            top_n_override=kb_top_n,
        )

    # ------------------------------------------------------------------
    # hier_v2 Phase 2 — agentic basin-pivot loop.
    # ------------------------------------------------------------------

    async def _query_agentic(
        self,
        kb_id: str,
        query_text: str,
        top_k: int,
        top_n_override: int | None = None,
        max_iter_override: int | None = None,
        per_iter_k_override: int | None = None,
    ) -> Dict[str, Any]:
        top_n = top_n_override or (top_k if top_k > 0 else settings.SELECTOR_TOP_N)

        svc = AgenticSearchService(
            embedding_client=self._embedding_client,
            llm_client=self._llm,
            repository_factory=self._classic_session_factory,
            selector_fn=_llm_select,
        )
        out = await svc.run(
            kb_id=kb_id, query_text=query_text, top_n=top_n,
            max_iter=max_iter_override, per_iter_k=per_iter_k_override,
        )
        return {
            "kb_id": kb_id,
            "query_type": "hier_v2_agentic",
            "search_type": "semantic",
            "query_text": query_text,
            "results": out["results"],
            "result_count": len(out["results"]),
            "agentic": out["agentic"],
        }

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
        top_n_override: int | None = None,
    ) -> Dict[str, Any]:
        try:
            query_embedding = await self._embedding_client.get_embedding(query_text)
        except Exception as exc:
            logger.error("Embedding failed for kb %s: %s", kb_id, exc, exc_info=True)
            raise DatabaseError(f"Embedding service error: {exc}")

        # hier_v2: over-fetch so parent/table dedup leaves a useful pool for
        # the LLM selector to pick from.
        top_n = top_n_override or (top_k if top_k > 0 else settings.SELECTOR_TOP_N)
        fetch_k = max(top_n * settings.SELECTOR_OVERFETCH_MULT, top_n)

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

        deduped = _dedupe_chunks(raw)
        results, selector_fallback = await _llm_select(
            self._llm, query_text, deduped, top_n=top_n,
        )

        return {
            "kb_id": kb_id,
            "query_type": "hier_v2",
            "search_type": search_type,
            "query_text": query_text,
            "results": results,
            "result_count": len(results),
            "selector": {
                "raw_count": len(raw),
                "dedup_count": len(deduped),
                "model": settings.SELECTOR_MODEL,
                "fallback_score_order": selector_fallback,
            },
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
