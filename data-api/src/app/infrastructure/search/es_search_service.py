"""Elasticsearch async search service for llm-wiki KBs.

Implements hybrid retrieval: BM25 (``multi_match`` over heading + content) +
kNN over the embedding vector, fused client-side via RRF. Matches the design
of NDNM1408/llm-wiki-elasticsearch but operates on a single per-KB index
(no separate raw vs wiki surfaces — synthesised wiki pages are out of scope
for this iteration).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch, NotFoundError

from app.configurations.configurations import settings
from app.infrastructure.search.es_index_schema import kb_index_name

logger = logging.getLogger(__name__)


# RRF tuning. ``k`` is the standard RRF constant; weights come from the
# reference repo which favours BM25 slightly (1.5) over kNN (1.0) for legal
# corpora — heading/content matches tend to be stronger signals than vector
# similarity alone.
RRF_K = 60
BM25_WEIGHT = 1.5
KNN_WEIGHT = 1.0
# Retrieve a few times the requested top_k from each lane so RRF has enough
# overlap to fuse meaningfully.
LANE_K_MULTIPLIER = 4
LANE_K_MIN = 40


class ElasticsearchSearchService:
    """Async ES client wrapper for retrieval and per-document deletes."""

    def __init__(self):
        self._client = AsyncElasticsearch(
            hosts=[settings.ELASTICSEARCH_URL],
            basic_auth=(
                (settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD)
                if settings.ELASTICSEARCH_USERNAME
                else None
            ),
            request_timeout=30,
        )
        self._index_prefix = settings.ELASTICSEARCH_INDEX_PREFIX

    async def hybrid_search(
        self,
        kb_id: str,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` chunks ranked by weighted RRF over BM25 + kNN."""
        index = kb_index_name(kb_id, self._index_prefix)
        lane_k = max(LANE_K_MIN, top_k * LANE_K_MULTIPLIER)

        try:
            bm25_resp = await self._client.search(
                index=index,
                query={
                    "multi_match": {
                        "query": query_text,
                        "fields": ["heading_path^2", "content"],
                    }
                },
                size=lane_k,
                _source=[
                    "chunk_id", "document_id", "kb_id", "doc_name",
                    "section_label", "heading_path", "ordinal",
                    "start_line", "end_line", "content",
                ],
            )
            knn_resp = await self._client.search(
                index=index,
                knn={
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": lane_k,
                    "num_candidates": max(lane_k * 4, 100),
                },
                size=lane_k,
                _source=[
                    "chunk_id", "document_id", "kb_id", "doc_name",
                    "section_label", "heading_path", "ordinal",
                    "start_line", "end_line", "content",
                ],
            )
        except NotFoundError:
            logger.warning("ES index %s missing — KB has no indexed chunks", index)
            return []

        bm25_hits = bm25_resp["hits"]["hits"]
        knn_hits = knn_resp["hits"]["hits"]

        fused = self._rrf_fuse(bm25_hits, knn_hits)
        return [self._format_hit(hit, score) for hit, score in fused[:top_k]]

    @staticmethod
    def _rrf_fuse(
        bm25_hits: List[Dict[str, Any]],
        knn_hits: List[Dict[str, Any]],
    ) -> List[tuple]:
        """Weighted RRF: ``score = sum_lanes(weight / (k + rank))``.

        Returns ``[(hit, score)]`` sorted by score desc. ``hit`` is the first
        occurrence we saw (BM25 wins ties so the BM25 ``_score`` surfaces).
        """
        merged: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(bm25_hits):
            chunk_id = hit["_source"]["chunk_id"]
            merged.setdefault(chunk_id, {"hit": hit, "score": 0.0})
            merged[chunk_id]["score"] += BM25_WEIGHT / (RRF_K + rank + 1)

        for rank, hit in enumerate(knn_hits):
            chunk_id = hit["_source"]["chunk_id"]
            merged.setdefault(chunk_id, {"hit": hit, "score": 0.0})
            merged[chunk_id]["score"] += KNN_WEIGHT / (RRF_K + rank + 1)

        return sorted(
            ((m["hit"], m["score"]) for m in merged.values()),
            key=lambda x: x[1],
            reverse=True,
        )

    @staticmethod
    def _format_hit(hit: Dict[str, Any], score: float) -> Dict[str, Any]:
        src = hit["_source"]
        return {
            "chunk_id": src["chunk_id"],
            "document_id": src["document_id"],
            "kb_id": src.get("kb_id"),
            "doc_name": src.get("doc_name"),
            "section_label": src.get("section_label"),
            "heading_path": src.get("heading_path"),
            "ordinal": src.get("ordinal"),
            "start_line": src.get("start_line"),
            "end_line": src.get("end_line"),
            "text": src.get("content"),
            "content": src.get("content"),
            # The chunk *content* IS the parent here — llm-wiki chunks are
            # already sized to a legal article, so we re-use the field name
            # the RAG node consumes for parent dedupe to keep the retrieval
            # surface uniform across rag_modes.
            "parent_text": src.get("content"),
            "similarity": float(score),
        }

    async def delete_document_chunks(self, kb_id: str, document_id: str) -> int:
        """Remove every chunk belonging to ``document_id`` from the KB index."""
        index = kb_index_name(kb_id, self._index_prefix)
        try:
            resp = await self._client.delete_by_query(
                index=index,
                query={"term": {"document_id": document_id}},
                refresh=True,
            )
            return int(resp.get("deleted", 0))
        except NotFoundError:
            return 0

    async def close(self) -> None:
        await self._client.close()
