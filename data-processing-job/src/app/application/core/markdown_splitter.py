"""Markdown-aware parent-child splitter.

A leaf section under a ``#`` heading is the natural unit ("paragraph") of
this splitter. Tables, images, bullet lists, code blocks — anything that
appears under that heading — belong to that section.

Per section, the splitter emits:

  • ONE chunk when ``heading_prefix + body`` fits ``retrieve_max_tokens``.
    ``content == parent_text``: there's nothing else for the LLM to see
    after a retrieval hit, so the embed text IS the context.

  • MULTIPLE children when the section overflows. Children are produced
    by greedily packing block-level units (paragraphs separated by blank
    lines; tables kept intact and row-split only when a single table
    overflows alone). EVERY child of the same section shares the same
    ``parent_text`` — the full section body with heading prefix —
    inlined into the chunk payload so retrieval does NOT need a second
    DB hop to fetch context.

Heading-prefix policy (``content_prefix_mode``):
  • ``full``    — chunk content begins with the full heading trail
                  (``# A\\n## B\\n### C``). Backward-compatible default.
  • ``deepest`` — chunk content begins with the leaf heading only
                  (``### C``). ``parent_text`` STILL carries the full
                  trail, so LLM context is unchanged.
  • ``none``    — chunk content has no heading; heading info lives only
                  in ``parent_text``.

Document preamble (text before the first ``#``) is emitted as its own
chunk with empty heading path — without this, title pages / status
blocks above the first heading are silently dropped.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken

logger = logging.getLogger(__name__)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


@dataclass
class ChunkRow:
    """One row emitted by the splitter.

    Mirrors the columns the preprocess service inserts into ``chunk``.
    ``content`` is what gets embedded; ``parent_text`` is what the LLM
    sees after a retrieval hit.
    """
    id: str
    content: str                       # embed + match text
    parent_text: str                   # full leaf-section body for LLM
    chunk_order_index: int             # 0-based position in the document
    tokens: int                        # token count of ``content``
    heading_path: Optional[str] = None


@dataclass
class _Section:
    level: int                         # 1..6 ; 0 = synthetic root for preamble
    title: str
    path: List[str] = field(default_factory=list)
    body_lines: List[str] = field(default_factory=list)
    children: List["_Section"] = field(default_factory=list)


class MarkdownSplitter:
    """Markdown-aware splitter producing denormalized parent-child chunks."""

    def __init__(
        self,
        tokenizer_model: str = "gpt-4o-mini",
        retrieve_max_tokens: int = 2048,
        retrieve_target_tokens: int = 1800,
        content_prefix_mode: str = "full",
    ):
        """
        Args:
            tokenizer_model:        tiktoken model used to count tokens.
            retrieve_max_tokens:    hard upper bound for embed pieces.
                                    Default 2048 matches
                                    ``gemini-embedding-001`` input limit.
                                    Smaller values (e.g. 256) push the
                                    splitter into a hierarchical-style
                                    child-chunk regime — better embedding
                                    precision, more chunks per doc.
            retrieve_target_tokens: greedy pack target — stop adding more
                                    paragraphs once a chunk crosses this
                                    threshold. Margin under max_tokens
                                    absorbs tokenizer drift between
                                    tiktoken (gpt-4o) and Gemini.
            content_prefix_mode:    how much heading-path context to inline
                                    at the START of ``content`` (the embed
                                    text). ``parent_text`` always carries
                                    the FULL prefix regardless of this
                                    setting — only the embed-side string
                                    changes.
                                      • ``full``    — every level
                                        (``# A\\n## B\\n### C``). Backward-
                                        compatible default.
                                      • ``deepest`` — leaf heading only
                                        (``### C``). Reduces boilerplate
                                        per chunk; recommended when
                                        retrieve_max_tokens <= 512.
                                      • ``none``    — embed body alone;
                                        heading info lives only in
                                        parent_text.
        """
        if retrieve_target_tokens > retrieve_max_tokens:
            raise ValueError("retrieve_target_tokens must be <= retrieve_max_tokens")
        if content_prefix_mode not in ("full", "deepest", "none"):
            raise ValueError(
                f"content_prefix_mode must be one of full|deepest|none, got {content_prefix_mode!r}"
            )
        try:
            self.encoding = tiktoken.encoding_for_model(tokenizer_model)
        except KeyError:
            for name in ("o200k_base", "cl100k_base"):
                try:
                    self.encoding = tiktoken.get_encoding(name)
                    break
                except (KeyError, ValueError):
                    continue
            else:
                raise
        self.retrieve_max_tokens = retrieve_max_tokens
        self.retrieve_target_tokens = retrieve_target_tokens
        self.content_prefix_mode = content_prefix_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str) -> List[ChunkRow]:
        if not text or not text.strip():
            return []

        root = self._build_tree(text)
        rows: List[ChunkRow] = []
        order = 0

        for section in self._iter_leaf_sections(root):
            body = "\n".join(section.body_lines).strip()
            if not body:
                continue

            heading_path = " > ".join(section.path) if section.path else None
            # parent_text always carries the FULL heading trail — that's
            # what the LLM sees on retrieve hit.
            parent_prefix = self._render_prefix(section.path, "full")
            # content (embed string) prefix is configurable; only affects
            # the embed-side string, never parent_text.
            content_prefix = self._render_prefix(section.path, self.content_prefix_mode)
            parent_text = (parent_prefix + body).strip()

            # ── Fast path: whole section fits — emit ONE chunk.
            single = (content_prefix + body).strip()
            if self._count(single) <= self.retrieve_max_tokens:
                rows.append(
                    ChunkRow(
                        id=str(uuid.uuid4()),
                        content=single,
                        parent_text=parent_text,
                        chunk_order_index=order,
                        tokens=self._count(single),
                        heading_path=heading_path,
                    )
                )
                order += 1
                continue

            # ── Overflow path: split into children, every child shares
            #     parent_text = full section.
            for piece in self._split_oversized_section(body, prefix=content_prefix):
                content = (content_prefix + piece).strip()
                rows.append(
                    ChunkRow(
                        id=str(uuid.uuid4()),
                        content=content,
                        parent_text=parent_text,
                        chunk_order_index=order,
                        tokens=self._count(content),
                        heading_path=heading_path,
                    )
                )
                order += 1

        return rows

    @staticmethod
    def _render_prefix(path: List[str], mode: str) -> str:
        if mode == "none" or not path:
            return ""
        if mode == "deepest":
            return f"{'#' * len(path)} {path[-1]}\n\n"
        return "\n".join(f"{'#' * (i + 1)} {h}" for i, h in enumerate(path)) + "\n\n"

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, text: str) -> _Section:
        """Parse markdown into a heading tree.

        Lines preceding the first heading become the body of a synthetic
        root section (level=0, title=""). Subsequent sections are nested
        by ``#`` count.
        """
        root = _Section(level=0, title="", path=[])
        stack: List[_Section] = [root]

        for raw_line in text.splitlines():
            m = HEADING_RE.match(raw_line)
            if not m:
                stack[-1].body_lines.append(raw_line)
                continue

            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1] if stack else root
            node = _Section(
                level=level,
                title=title,
                path=parent.path + [title],
            )
            parent.children.append(node)
            stack.append(node)

        return root

    def _iter_leaf_sections(self, node: _Section):
        """Yield every section that owns body text — including non-leaf
        nodes whose body precedes their children.

        The previous implementation only yielded true leaves, so document
        PREAMBLE (text before the first ``#`` heading) and SECTION INTROS
        (paragraph between ``# H1`` and ``## H2``) were silently dropped.
        Hit against DAB1 where the ``**Status**`` block (containing the
        document owner) sat above the first heading and never produced a
        chunk — Q1 in the benchmark dataset was unretrievable as a result.
        """
        has_body = bool("".join(node.body_lines).strip())
        if has_body:
            yield node
        for child in node.children:
            yield from self._iter_leaf_sections(child)

    # ------------------------------------------------------------------
    # Retrieve-piece extraction
    # ------------------------------------------------------------------

    def _split_oversized_section(self, body: str, *, prefix: str) -> List[str]:
        """Split an oversized section body into children, each fitting
        ``retrieve_max_tokens`` once the heading ``prefix`` is prepended.

        Block-level units (paragraphs / tables / bullet groups) are
        identified by blank-line boundaries and packed greedily up to the
        per-child budget. Tables stay intact within their packed group
        whenever possible; row-split only when a SINGLE table block
        overflows the budget on its own. Long uninterrupted prose hard-
        splits with a sliding window.
        """
        blocks = self._split_blocks(body)
        prefix_tokens = self._count(prefix) if prefix else 0
        budget = max(1, self.retrieve_max_tokens - prefix_tokens)

        out: List[str] = []
        current: List[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if current:
                joined = "\n\n".join(current).strip()
                if joined:
                    out.append(joined)
            current = []
            current_tokens = 0

        for block in blocks:
            block_tokens = self._count(block)

            # Block alone exceeds the per-child budget — flush pending
            # pack, then break this block down.
            if block_tokens > budget:
                flush()
                if self._looks_like_table(block):
                    out.extend(self._split_table_by_rows(block, budget))
                else:
                    out.extend(self._hard_split(block, budget))
                continue

            # Would adding this block push us over budget? Flush first.
            if current_tokens + block_tokens > budget and current:
                flush()

            current.append(block)
            current_tokens += block_tokens

        flush()
        return out

    def _split_blocks(self, body: str) -> List[str]:
        """Group lines into paragraph / table blocks, separated by blank lines."""
        blocks: List[str] = []
        buf: List[str] = []
        for line in body.splitlines():
            if line.strip() == "":
                if buf:
                    blocks.append("\n".join(buf).rstrip())
                    buf = []
            else:
                buf.append(line)
        if buf:
            blocks.append("\n".join(buf).rstrip())
        return [b for b in blocks if b]

    def _looks_like_table(self, block: str) -> bool:
        lines = block.splitlines()
        if len(lines) < 2:
            return False
        if not TABLE_LINE_RE.match(lines[0]):
            return False
        return bool(TABLE_SEP_RE.match(lines[1]))

    def _split_table_by_rows(self, block: str, budget: int) -> List[str]:
        """Split a too-big table by data rows, replicating the header +
        separator at the top of each piece. Each piece stays under
        ``budget`` tokens (already net of prefix overhead)."""
        lines = block.splitlines()
        header = lines[0]
        sep = lines[1]
        rows = lines[2:]
        if not rows:
            return [block]

        out: List[str] = []
        current = [header, sep]
        current_tokens = self._count("\n".join(current))
        for row in rows:
            row_tokens = self._count(row)
            if current_tokens + row_tokens > budget and len(current) > 2:
                out.append("\n".join(current))
                current = [header, sep]
                current_tokens = self._count("\n".join(current))
            current.append(row)
            current_tokens += row_tokens
        if len(current) > 2:
            out.append("\n".join(current))
        return out

    def _hard_split(self, block: str, budget: int) -> List[str]:
        """Token-level sliding window for a paragraph that overshoots
        ``budget`` on its own (rare — long uninterrupted prose, usually
        OCR'd PDFs with no paragraph breaks)."""
        tokens = self.encoding.encode(block)
        out: List[str] = []
        size = max(1, budget)
        for start in range(0, len(tokens), size):
            slice_ = tokens[start : start + size]
            out.append(self.encoding.decode(slice_).strip())
        return [s for s in out if s]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count(self, text: str) -> int:
        return len(self.encoding.encode(text))
