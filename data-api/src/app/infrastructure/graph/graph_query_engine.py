"""
GraphRAG query engine — replaces LightRAG's aquery() with direct
Neo4j + PGVector queries.

Reads from the same tables/graph that the ingestion pipeline writes to:
- Neo4j nodes (label = kb_id, property entity_id)
- Neo4j edges (type = DIRECTED)
- GRAPHRAG_VDB_ENTITY (PGVector entity embeddings)
- GRAPHRAG_VDB_RELATION (PGVector relation embeddings)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import json_repair
from sqlalchemy import text

from app.infrastructure.clients.embedding_client import EmbeddingClient
from app.infrastructure.graph.llm_client import LLMClient
from app.infrastructure.graph.neo4j_store import Neo4jStore
from app.infrastructure.graph.prompts import (
    FAIL_RESPONSE,
    KG_QUERY_CONTEXT_TEMPLATE,
    KEYWORDS_EXTRACTION_EXAMPLES,
    KEYWORDS_EXTRACTION_PROMPT,
    RAG_RESPONSE_PROMPT,
)

logger = logging.getLogger(__name__)

# Cosine distance threshold (distance space: lower = more similar).
# 0.8 ≈ similarity > 0.2 — permissive to avoid missing results.
_COSINE_DISTANCE_THRESHOLD = 0.8


class GraphQueryEngine:
    """Orchestrates graph-based retrieval without LightRAG."""

    def __init__(
        self,
        neo4j_store: Neo4jStore,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        session_factory: Any,
        chunk_repository: Any = None,
    ) -> None:
        self._neo4j = neo4j_store
        self._embedding = embedding_client
        self._llm = llm_client
        self._session_factory = session_factory
        self._chunk_repo = chunk_repository

    async def query(
        self,
        kb_id: str,
        query_text: str,
        mode: str = "hybrid",
        top_k: int = 40,
        only_context: bool = True,
        chunk_top_k: int = 10,
        max_entity_tokens: int = 4000,
        max_relation_tokens: int = 4000,
        max_total_tokens: int = 16000,
        response_type: str = "Multiple Paragraphs",
        hl_keywords: list[str] | None = None,
        ll_keywords: list[str] | None = None,
        user_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a graph-based query.

        Args:
            kb_id:              Knowledge base UUID (Neo4j label / PGVector workspace).
            query_text:         Raw user query.
            mode:               local | global | hybrid | naive | mix.
            top_k:              Max entities/relations to retrieve per search.
            only_context:       If True return context string; else generate LLM answer.
            chunk_top_k:        Number of document chunks to retrieve.
            max_entity_tokens:  Max tokens for entity context.
            max_relation_tokens: Max tokens for relation context.
            max_total_tokens:   Max total tokens for the entire context.
            response_type:      Response format (Multiple Paragraphs, Single Paragraph, etc.).
            hl_keywords:        Pre-computed high-level keywords (skips extraction if provided).
            ll_keywords:        Pre-computed low-level keywords (skips extraction if provided).
            user_prompt:        Additional instructions for LLM.

        Returns:
            dict with key "context" or "answer".
        """
        # 1. Extract keywords (or use provided ones)
        if hl_keywords is None or ll_keywords is None:
            extracted_hl, extracted_ll = await self._extract_keywords(query_text)
            hl_keywords = hl_keywords if hl_keywords is not None else extracted_hl
            ll_keywords = ll_keywords if ll_keywords is not None else extracted_ll

        logger.debug("High-level keywords: %s", hl_keywords)
        logger.debug("Low-level keywords: %s", ll_keywords)

        # Fallback: if both empty and query is short, use query as ll_keywords
        if not hl_keywords and not ll_keywords:
            if len(query_text) < 50:
                logger.warning("Forced ll_keywords to origin query: %s", query_text)
                ll_keywords = [query_text]
            else:
                return {"context" if only_context else "answer": FAIL_RESPONSE}

        ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
        hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

        # Store keywords for response metadata
        keywords_meta = {
            "hl_keywords": hl_keywords,
            "ll_keywords": ll_keywords,
        }

        # 2. Search based on mode
        local_entities: list[dict] = []
        local_relations: list[dict] = []
        global_entities: list[dict] = []
        global_relations: list[dict] = []

        if mode == "local" or mode == "naive":
            if ll_keywords_str:
                local_entities, local_relations = await self._search_entities(
                    kb_id, ll_keywords_str, top_k
                )
        elif mode == "global":
            if hl_keywords_str:
                global_relations, global_entities = await self._search_relations(
                    kb_id, hl_keywords_str, top_k
                )
        else:  # hybrid or mix
            tasks = []
            if ll_keywords_str:
                tasks.append(self._search_entities(kb_id, ll_keywords_str, top_k))
            if hl_keywords_str:
                tasks.append(self._search_relations(kb_id, hl_keywords_str, top_k))

            results = await asyncio.gather(*tasks)

            idx = 0
            if ll_keywords_str:
                local_entities, local_relations = results[idx]
                idx += 1
            if hl_keywords_str:
                global_relations, global_entities = results[idx]

        # 3. Round-robin merge
        entities, relations = self._merge_results(
            local_entities, local_relations, global_entities, global_relations
        )

        if not entities and not relations:
            return {"context" if only_context else "answer": FAIL_RESPONSE}

        # 4. Fetch document chunks referenced by entities/relations
        chunk_ids = self._collect_chunk_ids(entities, relations, max_chunks=chunk_top_k)
        chunk_map: dict[str, dict] = {}
        if self._chunk_repo is not None and chunk_ids:
            chunk_map = await self._chunk_repo.get_by_ids(chunk_ids)
        chunks = [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]
        # 5. Build context
        context = self._build_context(
            entities, relations, chunks,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
        )

        if only_context:
            return {
                "context": context,
                "keywords": keywords_meta,
                "entity_count": len(entities),
                "relation_count": len(relations),
                "chunk_count": len(chunks),
            }

        # 6. Generate LLM answer
        answer = await self._generate_answer(
            query_text, context, response_type=response_type, user_prompt=user_prompt
        )
        return {
            "answer": answer,
            "keywords": keywords_meta,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "chunk_count": len(chunks),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _extract_keywords(
        self, query_text: str
    ) -> tuple[list[str], list[str]]:
        """Call LLM to extract high/low level keywords from the query."""
        examples = "\n".join(KEYWORDS_EXTRACTION_EXAMPLES)
        prompt = KEYWORDS_EXTRACTION_PROMPT.format(
            query=query_text, examples=examples, language="English"
        )

        result = await self._llm.complete(prompt)

        # Strip <think>...</think> tags if present
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        try:
            keywords_data = json_repair.loads(result)
            if not keywords_data:
                return [], []
        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse keyword extraction response: %s", result)
            return [], []

        hl = keywords_data.get("high_level_keywords", [])
        ll = keywords_data.get("low_level_keywords", [])
        return hl, ll

    async def _search_entities(
        self, kb_id: str, keywords_str: str, top_k: int
    ) -> tuple[list[dict], list[dict]]:
        """
        Local search: vector-search entities → get nodes from Neo4j →
        find connected edges.

        Returns (node_datas, edge_datas).
        """
        # Vector search GRAPHRAG_VDB_ENTITY
        embedding = await self._embedding.get_embedding(keywords_str)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        sql = text("""
            SELECT e.entity_name, e.chunk_ids,
                   EXTRACT(EPOCH FROM e.create_time)::BIGINT AS created_at
            FROM "GRAPHRAG_VDB_ENTITY" e
            WHERE e.workspace = :workspace
              AND e.content_vector <=> CAST(:embedding AS vector) < :threshold
            ORDER BY e.content_vector <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {
                    "workspace": kb_id,
                    "embedding": embedding_str,
                    "threshold": _COSINE_DISTANCE_THRESHOLD,
                    "top_k": top_k,
                },
            )
            vdb_results = [
                {
                    "entity_name": row.entity_name,
                    "chunk_ids": row.chunk_ids or [],
                    "created_at": row.created_at,
                }
                for row in result.fetchall()
            ]

        if not vdb_results:
            return [], []

        node_ids = [r["entity_name"] for r in vdb_results]

        # Get node details and degrees from Neo4j concurrently
        nodes_dict, degrees_dict = await asyncio.gather(
            self._neo4j.get_nodes_batch(kb_id, node_ids),
            self._neo4j.node_degrees_batch(kb_id, node_ids),
        )

        # Enrich node data
        node_datas = []
        for vdb_row in vdb_results:
            name = vdb_row["entity_name"]
            node = nodes_dict.get(name)
            if node is None:
                continue
            node_datas.append(
                {
                    **node,
                    "entity_name": name,
                    "rank": degrees_dict.get(name, 0),
                    "created_at": vdb_row.get("created_at"),
                    "chunk_ids": vdb_row.get("chunk_ids") or [],
                }
            )

        # Find related edges
        edge_datas = await self._find_edges_from_entities(kb_id, node_datas)

        logger.info(
            "Local search: %d entities, %d relations", len(node_datas), len(edge_datas)
        )
        return node_datas, edge_datas

    async def _find_edges_from_entities(
        self, kb_id: str, node_datas: list[dict]
    ) -> list[dict]:
        """Get all edges connected to a set of entities, deduplicated and ranked."""
        if not node_datas:
            return []

        node_names = [d["entity_name"] for d in node_datas]
        batch_edges_dict = await self._neo4j.get_nodes_edges_batch(kb_id, node_names)

        all_edges: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        for node_name in node_names:
            for e in batch_edges_dict.get(node_name, []):
                sorted_edge = tuple(sorted(e))
                if sorted_edge not in seen:
                    seen.add(sorted_edge)
                    all_edges.append(sorted_edge)

        if not all_edges:
            return []

        edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
        edge_pairs_tuples = list(all_edges)

        edge_data_dict, edge_degrees_dict = await asyncio.gather(
            self._neo4j.get_edges_batch(kb_id, edge_pairs_dicts),
            self._neo4j.edge_degrees_batch(kb_id, edge_pairs_tuples),
        )

        all_edges_data: list[dict] = []
        for pair in all_edges:
            edge_props = edge_data_dict.get(pair)
            if edge_props is not None:
                if "weight" not in edge_props:
                    edge_props["weight"] = 1.0
                all_edges_data.append(
                    {
                        "src_tgt": pair,
                        "rank": edge_degrees_dict.get(pair, 0),
                        **edge_props,
                    }
                )

        all_edges_data.sort(key=lambda x: (x["rank"], x["weight"]), reverse=True)
        return all_edges_data

    async def _search_relations(
        self, kb_id: str, keywords_str: str, top_k: int
    ) -> tuple[list[dict], list[dict]]:
        """
        Global search: vector-search relations → get edges from Neo4j →
        find connected nodes.

        Returns (edge_datas, node_datas).
        """
        embedding = await self._embedding.get_embedding(keywords_str)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        sql = text("""
            SELECT r.source_id AS src_id, r.target_id AS tgt_id, r.chunk_ids,
                   EXTRACT(EPOCH FROM r.create_time)::BIGINT AS created_at
            FROM "GRAPHRAG_VDB_RELATION" r
            WHERE r.workspace = :workspace
              AND r.content_vector <=> CAST(:embedding AS vector) < :threshold
            ORDER BY r.content_vector <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        async with self._session_factory() as session:
            result = await session.execute(
                sql,
                {
                    "workspace": kb_id,
                    "embedding": embedding_str,
                    "threshold": _COSINE_DISTANCE_THRESHOLD,
                    "top_k": top_k,
                },
            )
            vdb_results = [
                {
                    "src_id": row.src_id,
                    "tgt_id": row.tgt_id,
                    "chunk_ids": row.chunk_ids or [],
                    "created_at": row.created_at,
                }
                for row in result.fetchall()
            ]

        if not vdb_results:
            return [], []

        # Get edge details from Neo4j
        edge_pairs_dicts = [
            {"src": r["src_id"], "tgt": r["tgt_id"]} for r in vdb_results
        ]
        edge_data_dict = await self._neo4j.get_edges_batch(kb_id, edge_pairs_dicts)

        edge_datas: list[dict] = []
        for vdb_row in vdb_results:
            pair = (vdb_row["src_id"], vdb_row["tgt_id"])
            edge_props = edge_data_dict.get(pair)
            if edge_props is not None:
                if "weight" not in edge_props:
                    edge_props["weight"] = 1.0
                edge_datas.append(
                    {
                        "src_id": vdb_row["src_id"],
                        "tgt_id": vdb_row["tgt_id"],
                        "chunk_ids": vdb_row.get("chunk_ids") or [],
                        "created_at": vdb_row.get("created_at"),
                        **edge_props,
                    }
                )

        # Find related entities
        entity_names: list[str] = []
        seen_entities: set[str] = set()
        for e in edge_datas:
            for key in ("src_id", "tgt_id"):
                name = e[key]
                if name not in seen_entities:
                    entity_names.append(name)
                    seen_entities.add(name)

        nodes_dict = await self._neo4j.get_nodes_batch(kb_id, entity_names)

        node_datas: list[dict] = []
        for name in entity_names:
            node = nodes_dict.get(name)
            if node is None:
                continue
            node_datas.append({**node, "entity_name": name})

        logger.info(
            "Global search: %d entities, %d relations",
            len(node_datas),
            len(edge_datas),
        )
        return edge_datas, node_datas

    @staticmethod
    def _merge_results(
        local_entities: list[dict],
        local_relations: list[dict],
        global_entities: list[dict],
        global_relations: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Round-robin interleave with deduplication."""
        # Merge entities
        final_entities: list[dict] = []
        seen_entities: set[str] = set()
        max_len = max(len(local_entities), len(global_entities))
        for i in range(max_len):
            if i < len(local_entities):
                entity = local_entities[i]
                name = entity.get("entity_name")
                if name and name not in seen_entities:
                    final_entities.append(entity)
                    seen_entities.add(name)
            if i < len(global_entities):
                entity = global_entities[i]
                name = entity.get("entity_name")
                if name and name not in seen_entities:
                    final_entities.append(entity)
                    seen_entities.add(name)

        # Merge relations
        final_relations: list[dict] = []
        seen_relations: set[tuple[str, str]] = set()
        max_len = max(len(local_relations), len(global_relations))
        for i in range(max_len):
            if i < len(local_relations):
                relation = local_relations[i]
                rel_key = _relation_key(relation)
                if rel_key not in seen_relations:
                    final_relations.append(relation)
                    seen_relations.add(rel_key)
            if i < len(global_relations):
                relation = global_relations[i]
                rel_key = _relation_key(relation)
                if rel_key not in seen_relations:
                    final_relations.append(relation)
                    seen_relations.add(rel_key)

        return final_entities, final_relations

    @staticmethod
    def _collect_chunk_ids(
        entities: list[dict], relations: list[dict], max_chunks: int = 10
    ) -> list[str]:
        """Count occurrences of each chunk across entities+relations. Return top max_chunks IDs."""
        occurrence: dict[str, int] = {}
        for item in (*entities, *relations):
            for cid in (item.get("chunk_ids") or []):
                occurrence[cid] = occurrence.get(cid, 0) + 1
        sorted_ids = sorted(occurrence, key=occurrence.__getitem__, reverse=True)
        return sorted_ids[:max_chunks]

    def _build_context(
        self,
        entities: list[dict],
        relations: list[dict],
        chunks: list[dict] | None = None,
        max_entity_tokens: int = 4000,
        max_relation_tokens: int = 4000,
        max_total_tokens: int = 16000,
    ) -> str:
        """Format entities, relations and chunks into the KG context template.

        Applies token limits to prevent context overflow.
        """
        # Simple token estimation: ~4 chars per token
        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        entities_list = []
        entity_tokens = 0
        for e in entities:
            if entity_tokens >= max_entity_tokens:
                break
            entry = {
                "entity": e.get("entity_name", ""),
                "type": e.get("entity_type", "UNKNOWN"),
                "description": e.get("description", ""),
            }
            entry_tokens = estimate_tokens(json.dumps(entry))
            if entity_tokens + entry_tokens <= max_entity_tokens:
                entities_list.append(entry)
                entity_tokens += entry_tokens
            else:
                break

        relations_list = []
        relation_tokens = 0
        for r in relations:
            if relation_tokens >= max_relation_tokens:
                break
            src = r.get("src_id") or (r.get("src_tgt", ("", ""))[0] if "src_tgt" in r else "")
            tgt = r.get("tgt_id") or (r.get("src_tgt", ("", ""))[1] if "src_tgt" in r else "")
            entry = {
                "entity1": src,
                "entity2": tgt,
                "description": r.get("description", ""),
            }
            entry_tokens = estimate_tokens(json.dumps(entry))
            if relation_tokens + entry_tokens <= max_relation_tokens:
                relations_list.append(entry)
                relation_tokens += entry_tokens
            else:
                break

        chunks = chunks or []
        chunks_list = []
        chunk_tokens = 0
        remaining_tokens = max_total_tokens - entity_tokens - relation_tokens
        for i, c in enumerate(chunks):
            if chunk_tokens >= remaining_tokens:
                break
            entry = {"reference_id": str(i + 1), "content": c["content"]}
            entry_tokens = estimate_tokens(c["content"])
            if chunk_tokens + entry_tokens <= remaining_tokens:
                chunks_list.append(entry)
                chunk_tokens += entry_tokens
            else:
                break

        reference_list = "\n".join(
            f"[{i + 1}] {c['doc_name']}" for i, c in enumerate(chunks[:len(chunks_list)])
        )

        return KG_QUERY_CONTEXT_TEMPLATE.format(
            entities_str=json.dumps(entities_list, ensure_ascii=False, indent=2),
            relations_str=json.dumps(relations_list, ensure_ascii=False, indent=2),
            chunks_str=json.dumps(chunks_list, ensure_ascii=False, indent=2),
            reference_list_str=reference_list,
        )

    async def _generate_answer(
        self,
        query_text: str,
        context: str,
        response_type: str = "Multiple Paragraphs",
        user_prompt: str | None = None,
    ) -> str:
        """Send context + query to LLM and return the answer."""
        sys_prompt = RAG_RESPONSE_PROMPT.format(
            context_data=context,
            response_type=response_type,
        )
        if user_prompt:
            sys_prompt += f"\n\nAdditional instructions: {user_prompt}"
        answer = await self._llm.complete(query_text, system_prompt=sys_prompt)
        return answer


def _relation_key(relation: dict) -> tuple[str, str]:
    """Build a canonical (sorted) key for deduplication."""
    if "src_tgt" in relation:
        return tuple(sorted(relation["src_tgt"]))
    return tuple(sorted([relation.get("src_id", ""), relation.get("tgt_id", "")]))
