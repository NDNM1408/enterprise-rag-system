"""Markdown-aware parent-child splitter (denormalized layout).

One markdown document becomes a flat list of retrieve chunks. Each row
carries both pieces the rest of the pipeline needs:

  • ``content``      — the embed text: heading path prefix + a paragraph /
                       table group capped at ``retrieve_max_tokens`` so it
                       fits the embedding model's input limit.
  • ``parent_text``  — the full enclosing leaf section (prefix + every
                       paragraph/table of the section). Injected into the
                       LLM prompt when this chunk is retrieved, giving the
                       model surrounding context without a second DB lookup.

Algorithm:
  1. Build a heading tree by parsing ``#`` lines.
  2. Walk every leaf section. The section's full body, with the heading
     path prepended, is its ``parent_text``.
  3. Inside a leaf, split the body into paragraph / table blocks separated
     by blank lines:
       • markdown tables → their own chunk (header + separator + rows);
         row-split when oversized.
       • paragraphs are greedily packed up to ``retrieve_target_tokens``.
       • any single paragraph exceeding ``retrieve_max_tokens`` is hard-
         split with a tiktoken sliding window.
  4. Prefix each retrieve piece with the heading path so the embedding
     sees the structural cue.
  5. Document preamble (text before the first heading) is treated as a
     synthetic leaf section when present.

The old "separate parent (generate) rows linked by parent_id" model from
the earlier splitter is replaced by inlining ``parent_text``. ``parent_id``
is no longer populated (legacy column stays NULL).
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
    ):
        """
        Args:
            tokenizer_model:        tiktoken model used to count tokens.
            retrieve_max_tokens:    hard upper bound for embed pieces.
                                    Default 2048 matches
                                    ``gemini-embedding-001`` input limit.
            retrieve_target_tokens: greedy pack target — stop adding more
                                    paragraphs once a chunk crosses this
                                    threshold. Margin under max_tokens
                                    absorbs tokenizer drift between
                                    tiktoken (gpt-4o) and Gemini.
        """
        if retrieve_target_tokens > retrieve_max_tokens:
            raise ValueError("retrieve_target_tokens must be <= retrieve_max_tokens")
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
            prefix = ""
            if section.path:
                # Render with hash levels so the LLM (and embedding) sees
                # the structural cue, not just a flat title.
                prefix = "\n".join(
                    f"{'#' * (i + 1)} {h}"
                    for i, h in enumerate(section.path)
                ) + "\n\n"

            parent_text = (prefix + body).strip()

            for piece in self._iter_retrieve_pieces(body):
                content = (prefix + piece).strip()
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
        """Yield leaf sections (no children).

        The synthetic root is a leaf only when the document has zero
        headings — otherwise document preamble is folded into the first
        heading's siblings (which is fine because preamble is typically
        a title page or empty).
        """
        if not node.children:
            if node.level == 0 and not "".join(node.body_lines).strip():
                return
            yield node
            return
        for child in node.children:
            yield from self._iter_leaf_sections(child)

    # ------------------------------------------------------------------
    # Retrieve-piece extraction
    # ------------------------------------------------------------------

    def _iter_retrieve_pieces(self, body: str) -> List[str]:
        """Return paragraph/table pieces each fitting under
        ``retrieve_max_tokens``.

        Walks the section body block-by-block (blank line separator).
          • Markdown tables become their own piece (preserve structure);
            row-split when oversized.
          • Paragraphs hard-split when a single one exceeds the cap.
          • Otherwise paragraphs are greedily packed up to
            ``retrieve_target_tokens``.
        """
        blocks = self._split_blocks(body)

        out: List[str] = []
        current: List[str] = []
        current_tokens = 0

        def flush_current() -> None:
            nonlocal current, current_tokens
            if current:
                joined = "\n\n".join(current).strip()
                if joined:
                    out.append(joined)
            current = []
            current_tokens = 0

        for block in blocks:
            block_tokens = self._count(block)

            if self._looks_like_table(block):
                flush_current()
                if block_tokens <= self.retrieve_max_tokens:
                    out.append(block)
                else:
                    out.extend(self._split_table_by_rows(block))
                continue

            if block_tokens > self.retrieve_max_tokens:
                flush_current()
                out.extend(self._hard_split(block))
                continue

            if current_tokens + block_tokens > self.retrieve_target_tokens and current:
                flush_current()

            current.append(block)
            current_tokens += block_tokens

        flush_current()
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

    def _split_table_by_rows(self, block: str) -> List[str]:
        """Split a too-big table by data rows, replicating the header +
        separator at the top of each piece."""
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
            if current_tokens + row_tokens > self.retrieve_target_tokens and len(current) > 2:
                out.append("\n".join(current))
                current = [header, sep]
                current_tokens = self._count("\n".join(current))
            current.append(row)
            current_tokens += row_tokens
        if len(current) > 2:
            out.append("\n".join(current))
        return out

    def _hard_split(self, block: str) -> List[str]:
        """Token-level sliding window for a paragraph that overshoots
        ``retrieve_max_tokens`` on its own (rare — long uninterrupted
        prose, usually OCR'd PDFs with no paragraph breaks)."""
        tokens = self.encoding.encode(block)
        out: List[str] = []
        size = self.retrieve_target_tokens
        for start in range(0, len(tokens), size):
            slice_ = tokens[start : start + size]
            out.append(self.encoding.decode(slice_).strip())
        return [s for s in out if s]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count(self, text: str) -> int:
        return len(self.encoding.encode(text))
