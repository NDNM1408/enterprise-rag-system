"""
Graph ingestion orchestrator — replaces ``rag.ainsert()``.

Pipeline: extract entities/relations → merge with graph → return counts.
"""

from __future__ import annotations

import logging

from app.infrastructure.graph.entity_extractor import EntityExtractor
from app.infrastructure.graph.graph_merger import GraphMerger

logger = logging.getLogger(__name__)


class GraphIngestor:
    """Orchestrate LLM extraction and graph merge for a single chunk."""

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        graph_merger: GraphMerger,
    ) -> None:
        self._extractor = entity_extractor
        self._merger = graph_merger

    async def ingest(
        self,
        content: str,
        kb_id: str,
        file_path: str,
        chunk_key: str | None = None,
    ) -> dict:
        """Extract entities/relations from content and merge into the graph.

        Args:
            content:    Text content to ingest.
            kb_id:      Knowledge base ID (Neo4j workspace label).
            file_path:  File path / document ID for provenance.
            chunk_key:  Optional chunk identifier for logging.

        Returns:
            Dict with entity_count and relation_count.
        """
        chunk_key = chunk_key or "unknown"

        logger.info("graph_ingestor: starting ingestion for chunk=%s kb=%s", chunk_key, kb_id)

        # 1. Extract entities and relations from text
        entities, relations = await self._extractor.extract(
            text=content,
            chunk_key=chunk_key,
            file_path=file_path,
        )

        # 2. Merge entities into graph
        if entities:
            await self._merger.merge_and_upsert_entities(kb_id, entities)

        # 3. Merge relations into graph
        if relations:
            await self._merger.merge_and_upsert_relations(kb_id, relations)

        result = {
            "chunk_key": chunk_key,
            "entity_count": len(entities),
            "relation_count": len(relations),
        }

        logger.info(
            "graph_ingestor: finished chunk=%s entities=%d relations=%d",
            chunk_key, result["entity_count"], result["relation_count"],
        )

        return result
