"""Unit tests for the denormalized parent-child MarkdownSplitter.

Each leaf section produces N retrieve chunks (depending on token budget),
and EVERY chunk carries the full enclosing section in ``parent_text``.
"""

import pytest

from app.application.core.markdown_splitter import MarkdownSplitter, ChunkRow


# ---------------------------------------------------------------------------
# Heading tree shape
# ---------------------------------------------------------------------------

class TestHeadingTree:

    def test_single_section_produces_at_least_one_chunk(self):
        s = MarkdownSplitter()
        rows = s.split("# Intro\n\nHello world.")
        assert len(rows) >= 1
        assert all(isinstance(r, ChunkRow) for r in rows)

    def test_heading_path_uses_arrow_separator(self):
        s = MarkdownSplitter()
        rows = s.split("# Top\n\n## Sub\n\nbody text here")
        # Leaf is "Top > Sub" — Top has no body of its own.
        leaf_paths = {r.heading_path for r in rows}
        assert leaf_paths == {"Top > Sub"}

    def test_multiple_leaves_yield_chunks_in_each(self):
        text = (
            "# A\n\nbody A.\n\n"
            "# B\n\nbody B.\n"
        )
        rows = MarkdownSplitter().split(text)
        paths = {r.heading_path for r in rows}
        assert paths == {"A", "B"}

    def test_no_headings_falls_back_to_synthetic_root(self):
        rows = MarkdownSplitter().split("Just plain text without any headings.")
        assert len(rows) == 1
        # Synthetic root has no path.
        assert rows[0].heading_path is None

    def test_empty_or_whitespace_input_returns_empty(self):
        s = MarkdownSplitter()
        assert s.split("") == []
        assert s.split("   \n\n   ") == []


# ---------------------------------------------------------------------------
# Parent-text inlining
# ---------------------------------------------------------------------------

class TestParentTextInlining:

    def test_every_chunk_has_parent_text(self):
        rows = MarkdownSplitter().split(
            "# Section\n\nfirst paragraph.\n\nsecond paragraph.\n"
        )
        assert all(r.parent_text for r in rows)

    def test_parent_text_includes_heading_prefix(self):
        rows = MarkdownSplitter().split("# Title\n\nbody paragraph.")
        assert rows[0].parent_text.startswith("# Title")

    def test_parent_text_contains_full_section_body(self):
        """All paragraphs of the leaf section must appear in parent_text,
        not just the paragraph that became this chunk."""
        text = (
            "# Sec\n\n"
            "alpha paragraph.\n\n"
            "beta paragraph.\n\n"
            "gamma paragraph.\n"
        )
        rows = MarkdownSplitter().split(text)
        for r in rows:
            assert "alpha paragraph" in r.parent_text
            assert "beta paragraph" in r.parent_text
            assert "gamma paragraph" in r.parent_text

    def test_chunks_in_same_section_share_parent_text(self):
        """Force two retrieve pieces in one section with a tight token cap
        so we know they share the same parent."""
        s = MarkdownSplitter(retrieve_max_tokens=20, retrieve_target_tokens=15)
        text = (
            "# Sec\n\n"
            "one two three four five six seven eight nine ten.\n\n"
            "more more more more more more more more more more.\n"
        )
        rows = s.split(text)
        # At least two pieces under such a tight cap.
        assert len(rows) >= 2
        # All from the same section → identical parent_text.
        assert len({r.parent_text for r in rows}) == 1


# ---------------------------------------------------------------------------
# Embed content vs parent
# ---------------------------------------------------------------------------

class TestEmbedContent:

    def test_content_has_heading_prefix(self):
        rows = MarkdownSplitter().split("# H\n\nbody.")
        assert rows[0].content.startswith("# H")

    def test_chunk_order_index_is_monotonic(self):
        rows = MarkdownSplitter().split(
            "# A\n\nbody a.\n\n# B\n\nbody b.\n"
        )
        indices = [r.chunk_order_index for r in rows]
        assert indices == sorted(indices)
        assert indices[0] == 0

    def test_each_chunk_has_unique_id(self):
        rows = MarkdownSplitter().split(
            "# A\n\nbody a.\n\n# B\n\nbody b.\n"
        )
        ids = [r.id for r in rows]
        assert len(ids) == len(set(ids))

    def test_no_generate_chunk_type_field(self):
        """The denormalized model has a single chunk type — the row class
        must no longer expose a chunk_type field."""
        rows = MarkdownSplitter().split("# H\n\nbody.")
        assert not hasattr(rows[0], "chunk_type")
        assert not hasattr(rows[0], "parent_id")


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------

class TestTableHandling:

    def test_table_appears_in_a_chunk(self):
        """Tables are part of their containing section, not chopped out
        into a separate chunk. When the section fits in budget, ONE chunk
        carries both the prose and the table."""
        text = (
            "# Sec\n\n"
            "prose paragraph.\n\n"
            "| h1 | h2 |\n"
            "| --- | --- |\n"
            "| a | b |\n"
            "| c | d |\n"
        )
        rows = MarkdownSplitter().split(text)
        # Section is small — should fit in default budget as ONE chunk.
        assert len(rows) == 1
        assert "| h1 |" in rows[0].content
        assert "prose paragraph" in rows[0].content


# ---------------------------------------------------------------------------
# Section-as-paragraph semantics (new model)
# ---------------------------------------------------------------------------

class TestSectionAsParagraph:

    def test_small_section_yields_single_chunk_with_content_equal_parent(self):
        """A section that fits in retrieve_max_tokens becomes ONE chunk.
        ``content`` and ``parent_text`` are the same string in this case
        (modulo content_prefix_mode differences)."""
        rows = MarkdownSplitter(content_prefix_mode="full").split(
            "# Sec\n\nshort body paragraph.\n"
        )
        assert len(rows) == 1
        assert rows[0].content == rows[0].parent_text

    def test_oversized_section_splits_into_children_sharing_parent(self):
        """Force overflow with a tight cap. Children share parent_text."""
        text = (
            "# Sec\n\n"
            "alpha paragraph with several words.\n\n"
            "beta paragraph with several words.\n\n"
            "gamma paragraph with several words.\n"
        )
        rows = MarkdownSplitter(
            retrieve_max_tokens=20,
            retrieve_target_tokens=15,
        ).split(text)
        assert len(rows) >= 2
        assert len({r.parent_text for r in rows}) == 1

    def test_preamble_before_first_heading_emits_a_chunk(self):
        """Title pages / status blocks above the first ``#`` heading are
        retrievable. heading_path is None for such chunks."""
        text = (
            "**Status:** owner is Alice.\n\n"
            "# I. Intro\n\n"
            "body.\n"
        )
        rows = MarkdownSplitter().split(text)
        preamble = [r for r in rows if r.heading_path is None]
        assert preamble, "preamble must produce its own chunk"
        assert "Alice" in preamble[0].content


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_target_must_not_exceed_max(self):
        with pytest.raises(ValueError):
            MarkdownSplitter(retrieve_max_tokens=100, retrieve_target_tokens=200)
