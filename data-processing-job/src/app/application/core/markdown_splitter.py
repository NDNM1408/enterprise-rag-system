"""V4 parent-child splitter: section parts cut at every table boundary.

A "part" is the parent unit. Within one leaf section, every table boundary
closes the current part; a trailing pure-text block (if any) becomes its
own part. Each part contains AT MOST one table plus the prose that
preceded it.

Every child chunk under a part references the same ``parent_id`` (UUID per
part) and ``parent_text`` (the rendered part). Retrieval dedupes by
``parent_id``.

Pipeline:
  1. Section tree split by markdown headings (#…######).
  2. Within each leaf section, split into paragraph / table blocks.
  3. Walk blocks → close current part at each table → emit a trailing
     text-only part if prose remains after the last table.
  4. Per part, emit children:
       - text blocks: paragraph-pack up to CHILD_CHUNK_SIZE tokens.
       - small table (≤ TABLE_TOKEN_LIMIT): one whole-table child.
       - large table: row-by-row children; cells with ≥
         MIN_BULLETS_PER_CELL bullets and > CELL_TOKEN_LIMIT tokens are
         further bullet-packed up to BULLET_TOKENS_PER_CHUNK.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import tiktoken

logger = logging.getLogger(__name__)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
# Bullet markers: •, *, -, +, "1." style numbered list.
BULLET_RE = re.compile(r"(?:^|[\s|])(?:[•●▪◦*+\-]|\d+\.)\s+")


@dataclass
class ChunkRow:
    """One child chunk emitted by the splitter."""
    id: str
    content: str
    parent_id: str
    parent_text: str
    chunk_order_index: int
    tokens: int
    heading_path: Optional[str] = None


@dataclass
class _Section:
    level: int
    title: str
    path: List[str] = field(default_factory=list)
    body_lines: List[str] = field(default_factory=list)
    children: List["_Section"] = field(default_factory=list)


@dataclass
class _Part:
    """A section slice: zero-or-more text blocks plus at most one table."""
    text_blocks: List[str] = field(default_factory=list)
    table: Optional[str] = None


class MarkdownSplitter:
    """V4 part-based parent-child splitter."""

    def __init__(
        self,
        tokenizer_model: str = "gpt-4o-mini",
        # Legacy param names kept so the existing container wiring (which
        # passes RETRIEVE_MAX_TOKENS / RETRIEVE_TARGET_TOKENS) still works.
        # Both map to ``child_chunk_size`` — v4 has a single child cap.
        retrieve_max_tokens: int = 256,
        retrieve_target_tokens: int = 256,
        parent_token_limit: int = 256,
        child_chunk_size: Optional[int] = None,
        table_token_limit: int = 256,
        cell_token_limit: int = 80,
        min_bullets_per_cell: int = 2,
        bullet_tokens_per_chunk: int = 400,
    ):
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

        # v4 unifies child cap; pick the smaller of legacy target / explicit override.
        self.child_chunk_size = child_chunk_size or min(
            retrieve_target_tokens, retrieve_max_tokens
        )
        self.parent_token_limit = parent_token_limit
        self.table_token_limit = table_token_limit
        self.cell_token_limit = cell_token_limit
        self.min_bullets_per_cell = min_bullets_per_cell
        self.bullet_tokens_per_chunk = bullet_tokens_per_chunk

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
            prefix = self._render_heading_prefix(section.path)

            for part in self._cut_section_into_parts(body):
                parent_id = str(uuid.uuid4())
                parent_text = self._render_part(prefix, part)

                for content in self._emit_children(prefix, part):
                    if not content.strip():
                        continue
                    rows.append(
                        ChunkRow(
                            id=str(uuid.uuid4()),
                            content=content,
                            parent_id=parent_id,
                            parent_text=parent_text,
                            chunk_order_index=order,
                            tokens=self._count(content),
                            heading_path=heading_path,
                        )
                    )
                    order += 1

        return rows

    # ------------------------------------------------------------------
    # Section tree
    # ------------------------------------------------------------------

    def _build_tree(self, text: str) -> _Section:
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
            node = _Section(level=level, title=title, path=parent.path + [title])
            parent.children.append(node)
            stack.append(node)

        return root

    def _iter_leaf_sections(self, node: _Section):
        if not node.children:
            if node.level == 0 and not "".join(node.body_lines).strip():
                return
            yield node
            return
        for child in node.children:
            yield from self._iter_leaf_sections(child)

    @staticmethod
    def _render_heading_prefix(path: List[str]) -> str:
        if not path:
            return ""
        return "\n".join(f"{'#' * (i + 1)} {h}" for i, h in enumerate(path)) + "\n\n"

    # ------------------------------------------------------------------
    # Section → parts (strict cut at every table boundary)
    # ------------------------------------------------------------------

    def _cut_section_into_parts(self, body: str) -> List[_Part]:
        """Strict-cut: every table closes the current part. Trailing prose
        after the last table becomes its own text-only part."""
        blocks = self._split_blocks(body)
        parts: List[_Part] = []
        pending_text: List[str] = []

        for block in blocks:
            if self._looks_like_table(block):
                parts.append(_Part(text_blocks=pending_text, table=block))
                pending_text = []
            else:
                pending_text.append(block)

        if pending_text:
            parts.append(_Part(text_blocks=pending_text, table=None))

        return parts

    def _render_part(self, prefix: str, part: _Part) -> str:
        """Render a part to the full ``parent_text`` (heading + text + table)."""
        body_parts: List[str] = []
        if part.text_blocks:
            body_parts.append("\n\n".join(part.text_blocks).strip())
        if part.table:
            body_parts.append(part.table)
        body = "\n\n".join(p for p in body_parts if p).strip()
        return (prefix + body).strip()

    # ------------------------------------------------------------------
    # Part → children
    # ------------------------------------------------------------------

    def _emit_children(self, prefix: str, part: _Part) -> List[str]:
        out: List[str] = []

        # 1. Text portion of the part — paragraph-pack to CHILD_CHUNK_SIZE.
        if part.text_blocks:
            text = "\n\n".join(part.text_blocks).strip()
            if text:
                for piece in self._pack_text(text):
                    out.append((prefix + piece).strip())

        # 2. Table portion.
        if part.table is not None:
            table_md = part.table
            if self._count(table_md) <= self.table_token_limit:
                out.append(self._build_small_table_chunk(prefix, part.text_blocks, table_md))
            else:
                out.extend(self._smart_split_table(prefix, table_md))

        return out

    # ------------------------------------------------------------------
    # Text packing
    # ------------------------------------------------------------------

    def _pack_text(self, text: str) -> List[str]:
        """Greedy-pack paragraphs up to ``child_chunk_size``. Oversized
        paragraphs fall back to token-window split."""
        blocks = self._split_blocks(text)
        out: List[str] = []
        current: List[str] = []
        current_tokens = 0

        def flush():
            nonlocal current, current_tokens
            if current:
                joined = "\n\n".join(current).strip()
                if joined:
                    out.append(joined)
            current = []
            current_tokens = 0

        for block in blocks:
            block_tokens = self._count(block)
            if block_tokens > self.child_chunk_size:
                flush()
                out.extend(self._hard_split(block))
                continue
            if current and current_tokens + block_tokens > self.child_chunk_size:
                flush()
            current.append(block)
            current_tokens += block_tokens
        flush()
        return out

    # ------------------------------------------------------------------
    # Table chunking
    # ------------------------------------------------------------------

    def _build_small_table_chunk(
        self, prefix: str, text_blocks: List[str], table_md: str
    ) -> str:
        """Small table fits whole; embed heading + brief description + table."""
        parts: List[str] = []
        # Keep only the last (closest) text paragraph as caption — full text
        # is already preserved in ``parent_text``.
        if text_blocks:
            parts.append(text_blocks[-1])
        parts.append(table_md)
        content = (prefix + "\n\n".join(parts)).strip()
        if self._count(content) <= self.table_token_limit + len(self.encoding.encode(prefix)):
            return content
        # Caption pushed it over — drop caption, keep prefix + table.
        content = (prefix + table_md).strip()
        if self._count(content) <= self.table_token_limit + len(self.encoding.encode(prefix)):
            return content
        return table_md.strip()

    def _smart_split_table(self, prefix: str, table_md: str) -> List[str]:
        """Per-row children; cells with ≥ ``min_bullets_per_cell`` bullets and
        over ``cell_token_limit`` are bullet-packed."""
        lines = table_md.splitlines()
        if len(lines) < 3:
            return [(prefix + table_md).strip()]
        header = lines[0]
        sep = lines[1]
        data_rows = [ln for ln in lines[2:] if ln.strip()]
        if not data_rows:
            return [(prefix + table_md).strip()]

        header_block = f"{header}\n{sep}"
        out: List[str] = []
        for row in data_rows:
            bullet_children = self._maybe_bullet_split_row(prefix, header, sep, row)
            if bullet_children:
                out.extend(bullet_children)
                continue
            # No cell triggered bullet-split → emit the row as a single child.
            chunk = f"{prefix}{header_block}\n{row}".strip()
            if self._count(chunk) <= max(self.child_chunk_size, self.bullet_tokens_per_chunk):
                out.append(chunk)
            else:
                out.extend(self._hard_split(chunk))
        return out

    def _maybe_bullet_split_row(
        self, prefix: str, header: str, sep: str, row: str
    ) -> List[str]:
        """If the row has any cell that is large and contains ≥ ``min_bullets``
        bullets, emit several bullet-packed children. Otherwise return [].

        Each emitted child preserves the row layout (header + sep + row) and
        substitutes the bullet-rich cell with one bucket of bullets at a time.
        """
        cells = self._split_row_cells(row)
        if not cells:
            return []

        target_idx = -1
        bullet_groups: List[str] = []
        for i, cell in enumerate(cells):
            if self._count(cell) <= self.cell_token_limit:
                continue
            bullets = self._extract_bullets(cell)
            if len(bullets) < self.min_bullets_per_cell:
                continue
            target_idx = i
            bullet_groups = self._pack_bullets(bullets)
            break

        if target_idx < 0 or not bullet_groups:
            return []

        header_block = f"{header}\n{sep}"
        out: List[str] = []
        for bucket in bullet_groups:
            substituted = list(cells)
            substituted[target_idx] = bucket
            new_row = "| " + " | ".join(substituted) + " |"
            chunk = f"{prefix}{header_block}\n{new_row}".strip()
            if self._count(chunk) <= max(self.bullet_tokens_per_chunk * 2, self.child_chunk_size):
                out.append(chunk)
            else:
                out.extend(self._hard_split(chunk))
        return out

    @staticmethod
    def _split_row_cells(row: str) -> List[str]:
        s = row.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip() for c in s.split("|")]

    @staticmethod
    def _extract_bullets(cell_text: str) -> List[str]:
        """Detect bullet markers and split the cell into bullets.

        Markdown table cells often squash bullets onto one line — we look for
        ``• * + -`` and "1." markers anywhere in the text and split before
        each marker."""
        # Insert a newline before each bullet marker (skip the first occurrence
        # if it is already at the start).
        normalized = BULLET_RE.sub(lambda m: "\n" + m.group(0).lstrip(), cell_text)
        parts = [p.strip() for p in normalized.split("\n") if p.strip()]
        # Filter to lines that actually start with a bullet marker so we
        # don't treat preceding caption sentences as bullets.
        bullets = [p for p in parts if BULLET_RE.match(" " + p)]
        if len(bullets) < 2:
            # Fallback: try splitting on ``;`` semicolons (also a common
            # squash separator in legal/policy tables).
            alt = [p.strip() for p in re.split(r"\s*;\s*", cell_text) if p.strip()]
            if len(alt) >= 2:
                return alt
        return bullets

    def _pack_bullets(self, bullets: List[str]) -> List[str]:
        """Greedy-pack bullets up to ``bullet_tokens_per_chunk``."""
        out: List[str] = []
        current: List[str] = []
        current_tokens = 0

        def flush():
            nonlocal current, current_tokens
            if current:
                out.append(" ".join(current).strip())
            current = []
            current_tokens = 0

        for b in bullets:
            t = self._count(b)
            if current and current_tokens + t > self.bullet_tokens_per_chunk:
                flush()
            current.append(b)
            current_tokens += t
        flush()
        return out

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _split_blocks(self, body: str) -> List[str]:
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

    def _hard_split(self, block: str) -> List[str]:
        tokens = self.encoding.encode(block)
        out: List[str] = []
        size = self.child_chunk_size
        for start in range(0, len(tokens), size):
            piece = self.encoding.decode(tokens[start : start + size]).strip()
            if piece:
                out.append(piece)
        return out

    def _count(self, text: str) -> int:
        return len(self.encoding.encode(text))
