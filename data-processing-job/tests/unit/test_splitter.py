"""Unit tests for DocumentSplitter (tiktoken-based)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from unittest.mock import MagicMock, patch
from app.application.core.splitter import DocumentSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_splitter(chunk_size=10, chunk_overlap=2, model_name="gpt-4o-mini"):
    """Create a DocumentSplitter whose tiktoken encoding is fully mocked."""
    with patch("app.application.core.splitter.tiktoken") as mock_tiktoken:
        mock_encoding = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        splitter = DocumentSplitter(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    # splitter.encoding is now the mock_encoding instance; expose it for tests
    return splitter


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestDocumentSplitterInit:
    def test_tiktoken_called_with_model_name(self):
        with patch("app.application.core.splitter.tiktoken") as mock_tiktoken:
            mock_tiktoken.encoding_for_model.return_value = MagicMock()
            DocumentSplitter(model_name="gpt-4o-mini")
            mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4o-mini")

    def test_default_chunk_size(self):
        with patch("app.application.core.splitter.tiktoken") as mock_tiktoken:
            mock_tiktoken.encoding_for_model.return_value = MagicMock()
            splitter = DocumentSplitter()
            assert splitter.chunk_size == 1200

    def test_default_chunk_overlap(self):
        with patch("app.application.core.splitter.tiktoken") as mock_tiktoken:
            mock_tiktoken.encoding_for_model.return_value = MagicMock()
            splitter = DocumentSplitter()
            assert splitter.chunk_overlap == 100


# ---------------------------------------------------------------------------
# split — return structure
# ---------------------------------------------------------------------------

class TestSplitReturnStructure:
    def test_returns_list(self):
        splitter = make_splitter(chunk_size=5, chunk_overlap=1)
        splitter.encoding.encode.return_value = [0, 1, 2, 3, 4]
        splitter.encoding.decode.return_value = "hello"
        result = splitter.split("hello")
        assert isinstance(result, list)

    def test_each_item_has_required_keys(self):
        splitter = make_splitter(chunk_size=5, chunk_overlap=1)
        splitter.encoding.encode.return_value = [0, 1, 2, 3, 4]
        splitter.encoding.decode.return_value = "hello"
        result = splitter.split("hello")
        for item in result:
            assert "content" in item
            assert "tokens" in item
            assert "chunk_order_index" in item

    def test_chunk_order_index_is_sequential(self):
        # 10 tokens, chunk_size=4, overlap=1 → stride=3 → starts: 0,3,6,9
        splitter = make_splitter(chunk_size=4, chunk_overlap=1)
        splitter.encoding.encode.return_value = list(range(10))
        splitter.encoding.decode.side_effect = lambda t: "x" * len(t)
        result = splitter.split("dummy")
        indices = [item["chunk_order_index"] for item in result]
        assert indices == list(range(len(result)))


# ---------------------------------------------------------------------------
# split — sliding window correctness
# ---------------------------------------------------------------------------

class TestSplitSlidingWindow:
    def test_single_chunk_when_text_fits(self):
        # 5 tokens, chunk_size=10, overlap=2 → stride=8 → range(0,5,8) = [0] → 1 chunk
        splitter = make_splitter(chunk_size=10, chunk_overlap=2)
        splitter.encoding.encode.return_value = list(range(5))
        splitter.encoding.decode.return_value = "short"
        result = splitter.split("short text")
        assert len(result) == 1

    def test_multiple_chunks_with_overlap(self):
        # 8 tokens, chunk_size=4, overlap=2 → stride=2
        # starts: range(0,8,2) = [0,2,4,6] → 4 chunks
        splitter = make_splitter(chunk_size=4, chunk_overlap=2)
        splitter.encoding.encode.return_value = list(range(8))
        splitter.encoding.decode.side_effect = lambda t: "x" * len(t)
        result = splitter.split("text")
        assert len(result) == 4

    def test_last_chunk_has_correct_token_count(self):
        # 6 tokens, chunk_size=4, overlap=1 → stride=3
        # starts: [0,3] → chunks [0:4]=4 tokens, [3:7]=3 tokens
        splitter = make_splitter(chunk_size=4, chunk_overlap=1)
        splitter.encoding.encode.return_value = list(range(6))
        splitter.encoding.decode.side_effect = lambda t: "x" * len(t)
        result = splitter.split("text")
        assert len(result) == 2
        assert result[0]["tokens"] == 4
        assert result[-1]["tokens"] == 3

    def test_empty_text_returns_empty_list(self):
        splitter = make_splitter(chunk_size=10, chunk_overlap=2)
        splitter.encoding.encode.return_value = []
        result = splitter.split("")
        assert result == []

    def test_content_is_stripped(self):
        # decode returns padded string; split() must strip it
        splitter = make_splitter(chunk_size=5, chunk_overlap=1)
        splitter.encoding.encode.return_value = list(range(5))
        splitter.encoding.decode.return_value = "  hello  "
        result = splitter.split("  hello  ")
        assert result[0]["content"] == "hello"
