"""
Merge extracted entities/relations with existing graph data.

Ported from lightrag/operate.py (_merge_nodes_then_upsert, _merge_edges_then_upsert).

Simplified: no source_ids limit, no entity_chunks_storage, no pipeline_status.
We keep the core merge logic: deduplicate descriptions, majority-vote entity type,
LLM summarize when descriptions exceed threshold, sum relation weights.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter

from app.infrastructure.graph.entity_extractor import ExtractedEntity, ExtractedRelation
from app.infrastructure.graph.llm_client import LLMClient
from app.infrastructure.graph.neo4j_store import Neo4jStore
from app.infrastructure.graph.pgvector_store import GraphVectorStore
from app.infrastructure.graph.prompts import (
    DEFAULT_LANGUAGE,
    GRAPH_FIELD_SEP,
    SUMMARIZE_DESCRIPTIONS_PROMPT,
)

logger = logging.getLogger(__name__)

DESCRIPTION_SUMMARIZE_THRESHOLD = 8


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute a unique ID: prefix + md5(content)."""
    return prefix + hashlib.md5(content.encode()).hexdigest()


def _merge_source_ids(existing: list[str], new: list[str]) -> list[str]:
    """Merge two lists of source IDs preserving order and removing duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for seq in (existing, new):
        for sid in seq:
            if sid and sid not in seen:
                seen.add(sid)
                merged.append(sid)
    return merged


class GraphMerger:
    """Merge extracted entities/relations into Neo4j + PGVector."""

    def __init__(
        self,
        neo4j_store: Neo4jStore,
        vector_store: GraphVectorStore,
        llm_client: LLMClient,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self._neo4j = neo4j_store
        self._vector = vector_store
        self._llm = llm_client
        self._language = language

    # ------------------------------------------------------------------
    # Description summarization
    # ------------------------------------------------------------------

    async def _summarize_descriptions(
        self, description_type: str, name: str, descriptions: list[str]
    ) -> str:
        """Summarize a list of descriptions using the LLM."""
        description_list_str = "\n".join(
            f'{{"description": "{d}"}}' for d in descriptions
        )
        prompt = SUMMARIZE_DESCRIPTIONS_PROMPT.format(
            description_type=description_type,
            description_name=name,
            description_list=description_list_str,
            summary_length=600,
            language=self._language,
        )
        return await self._llm.complete(prompt=prompt)

    async def _handle_description_merge(
        self, description_type: str, name: str, descriptions: list[str]
    ) -> str:
        """Merge descriptions: LLM summarize if >= threshold, else join with SEP."""
        if not descriptions:
            return ""
        if len(descriptions) == 1:
            return descriptions[0]
        if len(descriptions) >= DESCRIPTION_SUMMARIZE_THRESHOLD:
            logger.info("LLM summarizing %d descriptions for %s", len(descriptions), name)
            return await self._summarize_descriptions(description_type, name, descriptions)
        return GRAPH_FIELD_SEP.join(descriptions)

    # ------------------------------------------------------------------
    # Entity merge
    # ------------------------------------------------------------------

    async def merge_and_upsert_entities(
        self,
        kb_id: str,
        entities: dict[str, list[ExtractedEntity]],
    ) -> list[dict]:
        """Merge extracted entities with existing graph nodes and upsert."""
        results = []

        for entity_name, entity_list in entities.items():
            existing_node = await self._neo4j.get_node(kb_id, entity_name)

            # Collect existing data
            existing_types: list[str] = []
            existing_source_ids: list[str] = []
            existing_descriptions: list[str] = []
            existing_file_paths: list[str] = []

            if existing_node:
                if existing_node.get("entity_type"):
                    existing_types.append(existing_node["entity_type"])
                if existing_node.get("source_id"):
                    existing_source_ids = existing_node["source_id"].split(GRAPH_FIELD_SEP)
                if existing_node.get("description"):
                    existing_descriptions = existing_node["description"].split(GRAPH_FIELD_SEP)
                if existing_node.get("file_path"):
                    existing_file_paths = existing_node["file_path"].split(GRAPH_FIELD_SEP)

            # Merge source_ids
            new_source_ids = [e.source_id for e in entity_list if e.source_id]
            merged_source_ids = _merge_source_ids(existing_source_ids, new_source_ids)

            # Entity type: majority vote
            all_types = existing_types + [e.entity_type for e in entity_list]
            entity_type = Counter(all_types).most_common(1)[0][0]

            # Deduplicate new descriptions
            seen_desc: set[str] = set()
            new_descriptions: list[str] = []
            for e in entity_list:
                if e.description and e.description not in seen_desc:
                    seen_desc.add(e.description)
                    new_descriptions.append(e.description)

            # Merge descriptions
            all_descriptions = existing_descriptions + new_descriptions
            description = await self._handle_description_merge(
                "Entity", entity_name, all_descriptions
            )

            # Merge file paths
            seen_fps: set[str] = set()
            all_fps: list[str] = []
            for fp in existing_file_paths + [e.file_path for e in entity_list]:
                if fp and fp not in seen_fps:
                    seen_fps.add(fp)
                    all_fps.append(fp)
            file_path = GRAPH_FIELD_SEP.join(all_fps)

            source_id_str = GRAPH_FIELD_SEP.join(merged_source_ids)

            # Upsert to Neo4j
            node_data = {
                "entity_id": entity_name,
                "entity_type": entity_type,
                "description": description,
                "source_id": source_id_str,
                "file_path": file_path,
                "created_at": int(time.time()),
            }
            await self._neo4j.upsert_node(kb_id, entity_name, node_data)

            # Upsert entity embedding to PGVector
            vdb_id = compute_mdhash_id(str(entity_name), prefix="ent-")
            entity_content = f"{entity_name}\n{description}"
            await self._vector.upsert_entity(
                workspace=kb_id,
                vector_id=vdb_id,
                entity_name=entity_name,
                content=entity_content,
                source_id=source_id_str,
                file_path=file_path,
            )

            results.append(node_data)

        return results

    # ------------------------------------------------------------------
    # Relation merge
    # ------------------------------------------------------------------

    async def merge_and_upsert_relations(
        self,
        kb_id: str,
        relations: dict[tuple[str, str], list[ExtractedRelation]],
    ) -> list[dict]:
        """Merge extracted relations with existing graph edges and upsert."""
        results = []

        for (src, tgt), rel_list in relations.items():
            if src == tgt:
                continue

            existing_edge = await self._neo4j.get_edge(kb_id, src, tgt)

            # Collect existing data
            existing_source_ids: list[str] = []
            existing_descriptions: list[str] = []
            existing_keywords: set[str] = set()
            existing_weights: list[float] = []
            existing_file_paths: list[str] = []

            if existing_edge:
                existing_weights.append(existing_edge.get("weight", 1.0))
                if existing_edge.get("source_id"):
                    existing_source_ids = existing_edge["source_id"].split(GRAPH_FIELD_SEP)
                if existing_edge.get("description"):
                    existing_descriptions = existing_edge["description"].split(GRAPH_FIELD_SEP)
                if existing_edge.get("keywords"):
                    for kw in existing_edge["keywords"].split(GRAPH_FIELD_SEP):
                        existing_keywords.update(k.strip() for k in kw.split(",") if k.strip())
                if existing_edge.get("file_path"):
                    existing_file_paths = existing_edge["file_path"].split(GRAPH_FIELD_SEP)

            # Merge source_ids
            new_source_ids = [r.source_id for r in rel_list if r.source_id]
            merged_source_ids = _merge_source_ids(existing_source_ids, new_source_ids)
            source_id_str = GRAPH_FIELD_SEP.join(merged_source_ids)

            # Sum weights
            weight = sum([r.weight for r in rel_list] + existing_weights)

            # Merge keywords
            all_keywords = set(existing_keywords)
            for r in rel_list:
                if r.keywords:
                    all_keywords.update(k.strip() for k in r.keywords.split(",") if k.strip())
            keywords_str = ",".join(sorted(all_keywords))

            # Deduplicate new descriptions
            seen_desc: set[str] = set()
            new_descriptions: list[str] = []
            for r in rel_list:
                if r.description and r.description not in seen_desc:
                    seen_desc.add(r.description)
                    new_descriptions.append(r.description)

            # Merge descriptions
            all_descriptions = existing_descriptions + new_descriptions
            description = await self._handle_description_merge(
                "Relation", f"({src}, {tgt})", all_descriptions
            )

            # Merge file paths
            seen_fps: set[str] = set()
            all_fps: list[str] = []
            for fp in existing_file_paths + [r.file_path for r in rel_list]:
                if fp and fp not in seen_fps:
                    seen_fps.add(fp)
                    all_fps.append(fp)
            file_path = GRAPH_FIELD_SEP.join(all_fps)

            # Ensure src/tgt nodes exist (create with UNKNOWN type if not)
            for node_name in (src, tgt):
                existing = await self._neo4j.get_node(kb_id, node_name)
                if existing is None:
                    placeholder_node = {
                        "entity_id": node_name,
                        "source_id": source_id_str,
                        "description": description,
                        "entity_type": "UNKNOWN",
                        "file_path": file_path,
                        "created_at": int(time.time()),
                    }
                    await self._neo4j.upsert_node(kb_id, node_name, placeholder_node)

                    # Also create entity embedding for placeholder
                    vdb_id = compute_mdhash_id(node_name, prefix="ent-")
                    entity_content = f"{node_name}\n{description}"
                    await self._vector.upsert_entity(
                        workspace=kb_id,
                        vector_id=vdb_id,
                        entity_name=node_name,
                        content=entity_content,
                        source_id=source_id_str,
                        file_path=file_path,
                    )

            # Upsert edge to Neo4j
            edge_data = {
                "weight": weight,
                "description": description,
                "keywords": keywords_str,
                "source_id": source_id_str,
                "file_path": file_path,
                "created_at": int(time.time()),
            }
            await self._neo4j.upsert_edge(kb_id, src, tgt, edge_data)

            # Upsert relation embedding to PGVector
            # Sort src/tgt for consistent ID (LightRAG does this)
            sorted_src, sorted_tgt = (src, tgt) if src <= tgt else (tgt, src)
            rel_vdb_id = compute_mdhash_id(sorted_src + sorted_tgt, prefix="rel-")
            rel_content = f"{keywords_str}\t{sorted_src}\n{sorted_tgt}\n{description}"
            await self._vector.upsert_relation(
                workspace=kb_id,
                vector_id=rel_vdb_id,
                src_id=sorted_src,
                tgt_id=sorted_tgt,
                content=rel_content,
                source_id=source_id_str,
                file_path=file_path,
            )

            results.append(edge_data)

        return results
