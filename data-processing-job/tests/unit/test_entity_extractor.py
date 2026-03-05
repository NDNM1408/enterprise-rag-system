"""
Unit tests for EntityExtractor and parsing logic.
"""

import pytest

from app.infrastructure.graph.entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelation,
    _fix_delimiter_corruption,
    _parse_entity,
    _parse_extraction_result,
    _parse_relation,
    _sanitize_text,
)
from app.infrastructure.graph.prompts import COMPLETION_DELIMITER, TUPLE_DELIMITER
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# _sanitize_text
# ---------------------------------------------------------------------------

class TestSanitizeText:

    def test_strips_html_tags(self):
        assert _sanitize_text("<p>hello</p>") == "hello"
        assert _sanitize_text("<br>world<br/>") == "world"

    def test_fullwidth_to_halfwidth(self):
        assert _sanitize_text("Ｈｅｌｌｏ") == "Hello"
        assert _sanitize_text("１２３") == "123"

    def test_removes_outer_quotes(self):
        assert _sanitize_text('"hello"') == "hello"
        assert _sanitize_text("'hello'") == "hello"

    def test_chinese_quotes_removed(self):
        assert _sanitize_text("\u201chello\u201d") == "hello"

    def test_filters_short_numeric(self):
        assert _sanitize_text("42") == ""
        assert _sanitize_text("1.2") == ""

    def test_preserves_normal_text(self):
        assert _sanitize_text("Tokyo") == "Tokyo"

    def test_empty_input(self):
        assert _sanitize_text("") == ""
        assert _sanitize_text(None) == ""

    def test_remove_inner_quotes(self):
        result = _sanitize_text("\u201ctest\u201d value", remove_inner_quotes=True)
        assert "\u201c" not in result
        assert "\u201d" not in result


# ---------------------------------------------------------------------------
# _fix_delimiter_corruption
# ---------------------------------------------------------------------------

class TestFixDelimiterCorruption:

    def test_double_hash(self):
        result = _fix_delimiter_corruption("entity<|##|>name", "#", "<|#|>")
        assert result == "entity<|#|>name"

    def test_escaped_hash(self):
        result = _fix_delimiter_corruption(r"entity<|\#|>name", "#", "<|#|>")
        assert result == "entity<|#|>name"

    def test_empty_delimiter(self):
        result = _fix_delimiter_corruption("entity<|>name", "#", "<|#|>")
        assert result == "entity<|#|>name"

    def test_no_corruption(self):
        result = _fix_delimiter_corruption("entity<|#|>name", "#", "<|#|>")
        assert result == "entity<|#|>name"


# ---------------------------------------------------------------------------
# _parse_entity
# ---------------------------------------------------------------------------

class TestParseEntity:

    def test_valid_entity(self):
        attrs = ["entity", "Tokyo", "location", "Capital of Japan"]
        result = _parse_entity(attrs, "chunk-1", "doc-1")
        assert result is not None
        assert result.entity_name == "Tokyo"
        assert result.entity_type == "location"
        assert result.description == "Capital of Japan"

    def test_type_normalized_lowercase_no_spaces(self):
        attrs = ["entity", "Test", "Natural Object", "Description"]
        result = _parse_entity(attrs, "chunk-1", "doc-1")
        assert result.entity_type == "naturalobject"

    def test_wrong_field_count(self):
        attrs = ["entity", "Tokyo", "location"]
        assert _parse_entity(attrs, "chunk-1", "doc-1") is None

    def test_invalid_type_chars(self):
        attrs = ["entity", "Test", "type<bad>", "desc"]
        assert _parse_entity(attrs, "chunk-1", "doc-1") is None

    def test_empty_name(self):
        attrs = ["entity", "", "type", "desc"]
        assert _parse_entity(attrs, "chunk-1", "doc-1") is None

    def test_empty_description(self):
        attrs = ["entity", "Name", "type", ""]
        assert _parse_entity(attrs, "chunk-1", "doc-1") is None


# ---------------------------------------------------------------------------
# _parse_relation
# ---------------------------------------------------------------------------

class TestParseRelation:

    def test_valid_relation(self):
        attrs = ["relation", "Tokyo", "Japan", "capital, government", "Tokyo is the capital of Japan"]
        result = _parse_relation(attrs, "chunk-1", "doc-1")
        assert result is not None
        assert result.src_id == "Tokyo"
        assert result.tgt_id == "Japan"
        assert result.keywords == "capital, government"

    def test_same_src_tgt_rejected(self):
        attrs = ["relation", "Tokyo", "Tokyo", "self", "self-relation"]
        assert _parse_relation(attrs, "chunk-1", "doc-1") is None

    def test_wrong_field_count(self):
        attrs = ["relation", "A", "B", "kw"]
        assert _parse_relation(attrs, "chunk-1", "doc-1") is None

    def test_default_weight(self):
        attrs = ["relation", "A", "B", "kw", "desc"]
        result = _parse_relation(attrs, "chunk-1", "doc-1")
        assert result.weight == 1.0


# ---------------------------------------------------------------------------
# _parse_extraction_result
# ---------------------------------------------------------------------------

class TestParseExtractionResult:

    def test_parses_entities_and_relations(self):
        llm_output = (
            "entity<|#|>Tokyo<|#|>location<|#|>Capital of Japan\n"
            "entity<|#|>Japan<|#|>location<|#|>An island nation\n"
            "relation<|#|>Tokyo<|#|>Japan<|#|>capital<|#|>Tokyo is the capital of Japan\n"
            "<|COMPLETE|>"
        )
        entities, relations = _parse_extraction_result(llm_output, "chunk-1", "doc-1")

        assert "Tokyo" in entities
        assert "Japan" in entities
        assert ("Tokyo", "Japan") in relations

    def test_handles_missing_completion_delimiter(self):
        llm_output = "entity<|#|>Test<|#|>concept<|#|>A test entity\n"
        entities, relations = _parse_extraction_result(llm_output, "chunk-1", "doc-1")
        assert "Test" in entities

    def test_handles_corrupted_delimiters(self):
        llm_output = "entity<|##|>Tokyo<|#|>location<|#|>Capital\n<|COMPLETE|>"
        entities, _ = _parse_extraction_result(llm_output, "chunk-1", "doc-1")
        assert "Tokyo" in entities

    def test_empty_output(self):
        entities, relations = _parse_extraction_result("", "chunk-1", "doc-1")
        assert entities == {}
        assert relations == {}


# ---------------------------------------------------------------------------
# EntityExtractor (integration with mocked LLM)
# ---------------------------------------------------------------------------

class TestEntityExtractor:

    @pytest.mark.asyncio
    async def test_extract_calls_llm_and_parses(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=[
            # Initial extraction
            (
                "entity<|#|>Tokyo<|#|>location<|#|>Capital of Japan\n"
                "relation<|#|>Tokyo<|#|>Japan<|#|>capital<|#|>Tokyo is the capital\n"
                "<|COMPLETE|>"
            ),
            # Gleaning (no new items)
            "<|COMPLETE|>",
        ])

        extractor = EntityExtractor(llm_client=mock_llm, max_gleaning=1)
        entities, relations = await extractor.extract(
            text="Tokyo is the capital of Japan",
            chunk_key="chunk-1",
            file_path="doc-1",
        )

        assert "Tokyo" in entities
        assert len(entities["Tokyo"]) == 1
        assert entities["Tokyo"][0].entity_type == "location"
        assert ("Tokyo", "Japan") in relations

        # LLM called twice: initial + 1 gleaning
        assert mock_llm.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_extract_merges_gleaning_results(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(side_effect=[
            # Initial
            "entity<|#|>Tokyo<|#|>location<|#|>Capital of Japan\n<|COMPLETE|>",
            # Gleaning finds more
            "entity<|#|>Japan<|#|>location<|#|>An island nation\n<|COMPLETE|>",
        ])

        extractor = EntityExtractor(llm_client=mock_llm, max_gleaning=1)
        entities, _ = await extractor.extract("text", "chunk-1", "doc-1")

        assert "Tokyo" in entities
        assert "Japan" in entities

    @pytest.mark.asyncio
    async def test_extract_no_gleaning(self):
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=(
            "entity<|#|>Test<|#|>concept<|#|>A test\n<|COMPLETE|>"
        ))

        extractor = EntityExtractor(llm_client=mock_llm, max_gleaning=0)
        entities, _ = await extractor.extract("text", "chunk-1", "doc-1")

        assert "Test" in entities
        assert mock_llm.complete.await_count == 1
