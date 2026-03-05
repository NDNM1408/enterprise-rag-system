"""
Unit tests for GraphMerger with mocked stores.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.graph.entity_extractor import ExtractedEntity, ExtractedRelation
from app.infrastructure.graph.graph_merger import GraphMerger, compute_mdhash_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_neo4j_store():
    store = MagicMock()
    store.get_node = AsyncMock(return_value=None)
    store.get_edge = AsyncMock(return_value=None)
    store.upsert_node = AsyncMock()
    store.upsert_edge = AsyncMock()
    return store


def _make_vector_store():
    store = MagicMock()
    store.upsert_entity = AsyncMock()
    store.upsert_relation = AsyncMock()
    return store


def _make_llm_client():
    client = MagicMock()
    client.complete = AsyncMock(return_value="Summarized description.")
    return client


# ---------------------------------------------------------------------------
# compute_mdhash_id
# ---------------------------------------------------------------------------

class TestComputeMdhashId:

    def test_returns_prefixed_md5(self):
        result = compute_mdhash_id("Tokyo", prefix="ent-")
        assert result.startswith("ent-")
        assert len(result) == 4 + 32  # "ent-" + 32 hex chars

    def test_consistent(self):
        a = compute_mdhash_id("test", prefix="rel-")
        b = compute_mdhash_id("test", prefix="rel-")
        assert a == b

    def test_different_content(self):
        a = compute_mdhash_id("Tokyo", prefix="ent-")
        b = compute_mdhash_id("Japan", prefix="ent-")
        assert a != b


# ---------------------------------------------------------------------------
# Entity merge
# ---------------------------------------------------------------------------

class TestMergeAndUpsertEntities:

    @pytest.mark.asyncio
    async def test_new_entity_created(self):
        neo4j = _make_neo4j_store()
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        entities = {
            "Tokyo": [
                ExtractedEntity(
                    entity_name="Tokyo",
                    entity_type="location",
                    description="Capital of Japan",
                    source_id="chunk-1",
                    file_path="doc-1",
                )
            ]
        }

        results = await merger.merge_and_upsert_entities("kb-1", entities)

        assert len(results) == 1
        neo4j.upsert_node.assert_awaited_once()
        vector.upsert_entity.assert_awaited_once()

        # Verify node data
        call_args = neo4j.upsert_node.call_args
        assert call_args[0][0] == "kb-1"  # kb_id
        assert call_args[0][1] == "Tokyo"  # node_id
        node_data = call_args[0][2]
        assert node_data["entity_type"] == "location"
        assert node_data["description"] == "Capital of Japan"

    @pytest.mark.asyncio
    async def test_entity_merged_with_existing(self):
        neo4j = _make_neo4j_store()
        neo4j.get_node = AsyncMock(return_value={
            "entity_type": "location",
            "source_id": "chunk-0",
            "description": "A city in Japan",
            "file_path": "doc-0",
        })
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        entities = {
            "Tokyo": [
                ExtractedEntity(
                    entity_name="Tokyo",
                    entity_type="location",
                    description="Capital of Japan",
                    source_id="chunk-1",
                    file_path="doc-1",
                )
            ]
        }

        results = await merger.merge_and_upsert_entities("kb-1", entities)

        assert len(results) == 1
        node_data = neo4j.upsert_node.call_args[0][2]
        # Source IDs should be merged
        assert "chunk-0" in node_data["source_id"]
        assert "chunk-1" in node_data["source_id"]
        # Descriptions joined (below threshold, no LLM)
        assert "A city in Japan" in node_data["description"]
        assert "Capital of Japan" in node_data["description"]

    @pytest.mark.asyncio
    async def test_entity_type_majority_vote(self):
        neo4j = _make_neo4j_store()
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        entities = {
            "Test": [
                ExtractedEntity("Test", "concept", "desc1", "c1", "d1"),
                ExtractedEntity("Test", "method", "desc2", "c2", "d1"),
                ExtractedEntity("Test", "concept", "desc3", "c3", "d1"),
            ]
        }

        await merger.merge_and_upsert_entities("kb-1", entities)
        node_data = neo4j.upsert_node.call_args[0][2]
        assert node_data["entity_type"] == "concept"

    @pytest.mark.asyncio
    async def test_descriptions_deduplicated(self):
        neo4j = _make_neo4j_store()
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        entities = {
            "Test": [
                ExtractedEntity("Test", "concept", "same desc", "c1", "d1"),
                ExtractedEntity("Test", "concept", "same desc", "c2", "d1"),
                ExtractedEntity("Test", "concept", "unique desc", "c3", "d1"),
            ]
        }

        await merger.merge_and_upsert_entities("kb-1", entities)
        node_data = neo4j.upsert_node.call_args[0][2]
        # Should have 2 unique descriptions, not 3
        desc_parts = node_data["description"].split("<SEP>")
        assert len(desc_parts) == 2

    @pytest.mark.asyncio
    async def test_llm_summarize_above_threshold(self):
        neo4j = _make_neo4j_store()
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        # Create 9 unique descriptions (above threshold of 8)
        entities = {
            "Test": [
                ExtractedEntity("Test", "concept", f"description {i}", f"c{i}", "d1")
                for i in range(9)
            ]
        }

        await merger.merge_and_upsert_entities("kb-1", entities)
        # LLM should have been called for summarization
        llm.complete.assert_awaited_once()
        node_data = neo4j.upsert_node.call_args[0][2]
        assert node_data["description"] == "Summarized description."


# ---------------------------------------------------------------------------
# Relation merge
# ---------------------------------------------------------------------------

class TestMergeAndUpsertRelations:

    @pytest.mark.asyncio
    async def test_new_relation_created(self):
        neo4j = _make_neo4j_store()
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        relations = {
            ("Tokyo", "Japan"): [
                ExtractedRelation(
                    src_id="Tokyo",
                    tgt_id="Japan",
                    weight=1.0,
                    description="Tokyo is the capital",
                    keywords="capital, government",
                    source_id="chunk-1",
                    file_path="doc-1",
                )
            ]
        }

        results = await merger.merge_and_upsert_relations("kb-1", relations)

        assert len(results) == 1
        neo4j.upsert_edge.assert_awaited_once()
        vector.upsert_relation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_creates_missing_nodes(self):
        neo4j = _make_neo4j_store()
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        relations = {
            ("A", "B"): [
                ExtractedRelation("A", "B", 1.0, "desc", "kw", "c1", "d1")
            ]
        }

        await merger.merge_and_upsert_relations("kb-1", relations)

        # Both nodes should be created (get_node returns None)
        assert neo4j.upsert_node.await_count == 2
        # And their entity embeddings
        assert vector.upsert_entity.await_count == 2

    @pytest.mark.asyncio
    async def test_weights_summed(self):
        neo4j = _make_neo4j_store()
        neo4j.get_edge = AsyncMock(return_value={
            "weight": 2.0,
            "source_id": "c0",
            "description": "existing desc",
            "keywords": "kw1",
        })
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        relations = {
            ("A", "B"): [
                ExtractedRelation("A", "B", 3.0, "new desc", "kw2", "c1", "d1")
            ]
        }

        await merger.merge_and_upsert_relations("kb-1", relations)
        # upsert_edge(kb_id, src, tgt, edge_data) — positional args
        edge_data = neo4j.upsert_edge.call_args[0][3]
        assert edge_data["weight"] == 5.0  # 2.0 + 3.0

    @pytest.mark.asyncio
    async def test_keywords_merged(self):
        neo4j = _make_neo4j_store()
        neo4j.get_edge = AsyncMock(return_value={
            "weight": 1.0,
            "source_id": "c0",
            "description": "desc",
            "keywords": "alpha, beta",
        })
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        relations = {
            ("A", "B"): [
                ExtractedRelation("A", "B", 1.0, "desc2", "beta, gamma", "c1", "d1")
            ]
        }

        await merger.merge_and_upsert_relations("kb-1", relations)
        # upsert_edge(kb_id, src, tgt, edge_data) — positional args
        edge_data = neo4j.upsert_edge.call_args[0][3]
        keywords = set(edge_data["keywords"].split(","))
        assert {"alpha", "beta", "gamma"} == keywords

    @pytest.mark.asyncio
    async def test_self_relation_skipped(self):
        neo4j = _make_neo4j_store()
        vector = _make_vector_store()
        llm = _make_llm_client()
        merger = GraphMerger(neo4j, vector, llm)

        relations = {
            ("A", "A"): [
                ExtractedRelation("A", "A", 1.0, "self", "kw", "c1", "d1")
            ]
        }

        results = await merger.merge_and_upsert_relations("kb-1", relations)
        assert results == []
        neo4j.upsert_edge.assert_not_awaited()
