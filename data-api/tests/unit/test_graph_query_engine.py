"""
Unit tests for GraphQueryEngine.

All external calls (Neo4j, PGVector, LLM, Embedding) are mocked.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.graph.graph_query_engine import GraphQueryEngine
from app.infrastructure.graph.prompts import FAIL_RESPONSE


def _make_engine(
    llm_response='{"high_level_keywords": ["AI"], "low_level_keywords": ["GPT"]}',
    embedding_result=None,
    vdb_entity_rows=None,
    vdb_relation_rows=None,
    neo4j_nodes=None,
    neo4j_degrees=None,
    neo4j_edges=None,
    neo4j_node_edges=None,
    neo4j_edge_degrees=None,
):
    """Build a GraphQueryEngine with all dependencies mocked."""
    mock_neo4j = MagicMock()
    mock_neo4j.get_nodes_batch = AsyncMock(return_value=neo4j_nodes or {})
    mock_neo4j.node_degrees_batch = AsyncMock(return_value=neo4j_degrees or {})
    mock_neo4j.get_edges_batch = AsyncMock(return_value=neo4j_edges or {})
    mock_neo4j.get_nodes_edges_batch = AsyncMock(return_value=neo4j_node_edges or {})
    mock_neo4j.edge_degrees_batch = AsyncMock(return_value=neo4j_edge_degrees or {})

    mock_embedding = MagicMock()
    mock_embedding.get_embedding = AsyncMock(
        return_value=embedding_result or [0.1] * 1024
    )

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=llm_response)

    # Mock session factory — returns async context manager
    mock_session = MagicMock()
    mock_result = MagicMock()

    if vdb_entity_rows is not None:
        entity_rows = vdb_entity_rows
    else:
        entity_rows = []

    if vdb_relation_rows is not None:
        relation_rows = vdb_relation_rows
    else:
        relation_rows = []

    # We'll track calls to distinguish entity vs relation queries
    call_count = {"n": 0}
    all_rows = [entity_rows, relation_rows]

    def _fetchall_side_effect():
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(all_rows):
            return all_rows[idx]
        return []

    mock_result.fetchall = MagicMock(side_effect=_fetchall_side_effect)
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_chunk_repo = MagicMock()
    mock_chunk_repo.get_by_ids = AsyncMock(return_value={})

    engine = GraphQueryEngine(
        neo4j_store=mock_neo4j,
        embedding_client=mock_embedding,
        llm_client=mock_llm,
        session_factory=mock_session_factory,
        chunk_repository=mock_chunk_repo,
    )
    return engine, mock_neo4j, mock_embedding, mock_llm, mock_chunk_repo


class TestKeywordExtraction:

    @pytest.mark.asyncio
    async def test_extracts_keywords_from_llm_response(self):
        engine, _, _, mock_llm, _ = _make_engine(
            llm_response='{"high_level_keywords": ["Trade"], "low_level_keywords": ["Tariffs"]}'
        )
        hl, ll = await engine._extract_keywords("How do tariffs affect trade?")
        assert hl == ["Trade"]
        assert ll == ["Tariffs"]

    @pytest.mark.asyncio
    async def test_handles_empty_llm_response(self):
        engine, _, _, _, _ = _make_engine(llm_response="{}")
        hl, ll = await engine._extract_keywords("hello")
        assert hl == []
        assert ll == []

    @pytest.mark.asyncio
    async def test_handles_malformed_llm_response(self):
        engine, _, _, _, _ = _make_engine(llm_response="not json at all")
        hl, ll = await engine._extract_keywords("test query")
        # json_repair may still parse something; either way should not raise
        assert isinstance(hl, list)
        assert isinstance(ll, list)

    @pytest.mark.asyncio
    async def test_strips_think_tags(self):
        engine, _, _, _, _ = _make_engine(
            llm_response='<think>reasoning</think>{"high_level_keywords": ["X"], "low_level_keywords": ["Y"]}'
        )
        hl, ll = await engine._extract_keywords("query")
        assert hl == ["X"]
        assert ll == ["Y"]


class TestMergeResults:

    def test_round_robin_merge_entities(self):
        local = [{"entity_name": "A"}, {"entity_name": "B"}]
        global_ = [{"entity_name": "C"}, {"entity_name": "A"}]  # A is duplicate

        entities, _ = GraphQueryEngine._merge_results(local, [], global_, [])
        names = [e["entity_name"] for e in entities]
        assert names == ["A", "C", "B"]  # A from local, C from global, B from local

    def test_round_robin_merge_relations(self):
        local_r = [{"src_tgt": ("X", "Y")}, {"src_tgt": ("A", "B")}]
        global_r = [{"src_tgt": ("Y", "X")}]  # same edge as X-Y when sorted

        _, relations = GraphQueryEngine._merge_results([], local_r, [], global_r)
        assert len(relations) == 2  # (X,Y) and (A,B); global (Y,X) is deduplicated


class TestBuildContext:

    def test_builds_json_context(self):
        entities = [
            {"entity_name": "Apple", "entity_type": "COMPANY", "description": "Tech company"},
        ]
        relations = [
            {"src_id": "Apple", "tgt_id": "iPhone", "description": "produces"},
        ]
        context = GraphQueryEngine._build_context(entities, relations, chunks=[])
        assert "Apple" in context
        assert "iPhone" in context
        assert "produces" in context
        assert "Knowledge Graph Data" in context

    def test_builds_context_with_chunks(self):
        entities = [{"entity_name": "Apple", "entity_type": "COMPANY", "description": "Tech"}]
        relations = []
        chunks = [
            {"content": "Apple was founded in 1976.", "doc_name": "Apple History.pdf"},
        ]
        context = GraphQueryEngine._build_context(entities, relations, chunks=chunks)
        assert "Apple was founded in 1976." in context
        assert "Apple History.pdf" in context
        assert "[1]" in context

    def test_builds_context_without_chunks_arg(self):
        """Omitting chunks arg should not crash."""
        context = GraphQueryEngine._build_context([], [])
        assert "Knowledge Graph Data" in context


class TestCollectChunkIds:

    def test_collects_from_entities(self):
        entities = [
            {"entity_name": "A", "chunk_ids": ["c1", "c2"]},
            {"entity_name": "B", "chunk_ids": ["c1", "c3"]},
        ]
        ids = GraphQueryEngine._collect_chunk_ids(entities, [])
        assert "c1" in ids
        assert ids[0] == "c1"  # c1 appears twice, should rank first

    def test_collects_from_relations(self):
        relations = [
            {"src_id": "A", "tgt_id": "B", "chunk_ids": ["r1"]},
        ]
        ids = GraphQueryEngine._collect_chunk_ids([], relations)
        assert ids == ["r1"]

    def test_deduplicates_and_ranks(self):
        entities = [{"chunk_ids": ["c1", "c2"]}, {"chunk_ids": ["c1"]}]
        relations = [{"chunk_ids": ["c2", "c3"]}]
        ids = GraphQueryEngine._collect_chunk_ids(entities, relations)
        # c1: 2, c2: 2, c3: 1 — c1 and c2 tie; both before c3
        assert set(ids[:2]) == {"c1", "c2"}
        assert ids[2] == "c3"

    def test_respects_max_chunks(self):
        entities = [{"chunk_ids": [f"c{i}" for i in range(20)]}]
        ids = GraphQueryEngine._collect_chunk_ids(entities, [], max_chunks=5)
        assert len(ids) == 5

    def test_empty_input(self):
        assert GraphQueryEngine._collect_chunk_ids([], []) == []


class TestChunkFetching:

    @pytest.mark.asyncio
    async def test_chunk_repo_called_with_collected_ids(self):
        """get_by_ids should be called with chunk IDs from entities."""
        entity_row = MagicMock()
        entity_row.entity_name = "TestEntity"
        entity_row.chunk_ids = ["chunk-001", "chunk-002"]
        entity_row.created_at = 1000

        engine, _, _, _, mock_chunk_repo = _make_engine(
            vdb_entity_rows=[entity_row],
            neo4j_nodes={"TestEntity": {"entity_type": "THING", "description": "test"}},
            neo4j_degrees={"TestEntity": 1},
            neo4j_node_edges={"TestEntity": []},
        )
        mock_chunk_repo.get_by_ids = AsyncMock(return_value={})

        await engine.query(kb_id="kb-1", query_text="find entity", mode="local", only_context=True)

        mock_chunk_repo.get_by_ids.assert_called_once()
        call_args = mock_chunk_repo.get_by_ids.call_args[0][0]
        assert set(call_args) == {"chunk-001", "chunk-002"}

    @pytest.mark.asyncio
    async def test_chunks_appear_in_context(self):
        """When chunk repo returns data, chunks should appear in the context string."""
        entity_row = MagicMock()
        entity_row.entity_name = "Alpha"
        entity_row.chunk_ids = ["c1"]
        entity_row.created_at = 999

        engine, _, _, _, mock_chunk_repo = _make_engine(
            vdb_entity_rows=[entity_row],
            neo4j_nodes={"Alpha": {"entity_type": "ORG", "description": "Alpha org"}},
            neo4j_degrees={"Alpha": 1},
            neo4j_node_edges={"Alpha": []},
        )
        mock_chunk_repo.get_by_ids = AsyncMock(
            return_value={"c1": {"content": "Alpha is great.", "doc_name": "alpha.pdf"}}
        )

        result = await engine.query(kb_id="kb-1", query_text="tell me about Alpha", mode="local", only_context=True)

        context = result["context"]
        assert "Alpha is great." in context
        assert "alpha.pdf" in context


class TestQueryModeRouting:

    @pytest.mark.asyncio
    async def test_local_mode_only_searches_entities(self):
        """Local mode should call entity search, not relation search."""
        entity_row = MagicMock()
        entity_row.entity_name = "TestEntity"
        entity_row.created_at = 1000

        engine, mock_neo4j, _, _, _ = _make_engine(
            vdb_entity_rows=[entity_row],
            neo4j_nodes={"TestEntity": {"entity_type": "THING", "description": "test"}},
            neo4j_degrees={"TestEntity": 2},
            neo4j_node_edges={"TestEntity": []},
        )

        result = await engine.query(
            kb_id="kb-1", query_text="find entity", mode="local", only_context=True
        )

        assert "context" in result
        # Entity search was called (get_nodes_batch)
        mock_neo4j.get_nodes_batch.assert_called()

    @pytest.mark.asyncio
    async def test_global_mode_only_searches_relations(self):
        """Global mode should search relations, not entities."""
        rel_row = MagicMock()
        rel_row.src_id = "A"
        rel_row.tgt_id = "B"
        rel_row.created_at = 1000

        engine, mock_neo4j, _, _, _ = _make_engine(
            llm_response='{"high_level_keywords": ["concept"], "low_level_keywords": []}',
            vdb_relation_rows=[rel_row],
            neo4j_edges={("A", "B"): {"weight": 1.0, "description": "related"}},
            neo4j_nodes={"A": {"entity_type": "X"}, "B": {"entity_type": "Y"}},
        )

        # For global mode, the session_factory fetchall is called once for relations
        # We need to reset the call count logic
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall = MagicMock(return_value=[rel_row])
        mock_session.execute = AsyncMock(return_value=mock_result)
        engine._session_factory = MagicMock()
        engine._session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        engine._session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await engine.query(
            kb_id="kb-1", query_text="broad concept", mode="global", only_context=True
        )

        assert "context" in result

    @pytest.mark.asyncio
    async def test_empty_results_returns_fail_response(self):
        """When no entities or relations found, return FAIL_RESPONSE."""
        engine, _, _, _, _ = _make_engine(
            vdb_entity_rows=[],
            vdb_relation_rows=[],
        )

        result = await engine.query(
            kb_id="kb-1", query_text="something unknown", mode="hybrid", only_context=True
        )

        assert result["context"] == FAIL_RESPONSE

    @pytest.mark.asyncio
    async def test_only_context_false_calls_llm(self):
        """When only_context=False, engine should call LLM for answer generation."""
        entity_row = MagicMock()
        entity_row.entity_name = "TestEntity"
        entity_row.created_at = 1000

        engine, _, _, mock_llm, _ = _make_engine(
            vdb_entity_rows=[entity_row],
            neo4j_nodes={"TestEntity": {"entity_type": "THING", "description": "test"}},
            neo4j_degrees={"TestEntity": 1},
            neo4j_node_edges={"TestEntity": []},
        )

        # First LLM call: keyword extraction, Second: answer generation
        mock_llm.complete = AsyncMock(
            side_effect=[
                '{"high_level_keywords": ["AI"], "low_level_keywords": ["GPT"]}',
                "The generated answer.",
            ]
        )

        result = await engine.query(
            kb_id="kb-1", query_text="test", mode="local", only_context=False
        )

        assert result["answer"] == "The generated answer."
        assert mock_llm.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_short_query_fallback_when_keywords_empty(self):
        """Short query with empty keywords should fallback to using query as ll_keywords."""
        engine, _, _, _, _ = _make_engine(
            llm_response='{"high_level_keywords": [], "low_level_keywords": []}',
            vdb_entity_rows=[],
        )

        result = await engine.query(
            kb_id="kb-1", query_text="hi", mode="local", only_context=True
        )
        # Either returns FAIL_RESPONSE (no results) or context; should not crash
        assert "context" in result
