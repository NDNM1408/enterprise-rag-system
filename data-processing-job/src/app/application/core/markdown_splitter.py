"""Markdown-aware parent-child splitter (denormalized, table-segmented).

Section → parent groups, partitioned at each table boundary. Each parent owns
at most one table plus the surrounding description text (from the previous
table boundary, or section start, up to the table). A parent without a table
covers text-only tail of the section.

Within a parent:
  • Description text → paragraph-packed child chunks (split by blank line,
    greedy-packed up to ``retrieve_target_tokens``).
  • Table → either a single chunk that bundles heading + description + table
    (when the whole table fits), or per-row child chunks (when the table is
    oversize). Per-row chunks include the heading prefix and the table header
    so embeddings still have column context.

Both kinds of children share the same ``parent_text`` (heading + description
+ full table) so retrieval can re-inject full context to the LLM.

Algorithm at a glance::

    section body
        │
        ▼
    split into paragraph / table blocks (blank-line separated)
        │
        ▼
    walk blocks; emit a parent each time a table is consumed,
      and one trailing text-only parent for residual prose
        │
        ▼
    per parent → fan out child chunks (text-pack + table-or-rows)
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
    """One row emitted by the splitter (mirrors the ``chunk`` table columns)."""
    id: str
    content: str
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
class _ParentGroup:
    """One parent slice of a section: a (possibly empty) description text
    block and at most one table. A trailing pure-text group has table=None."""
    text: str
    table: Optional[str]


class MarkdownSplitter:
    """Table-segmented parent-child splitter."""

    def __init__(
        self,
        tokenizer_model: str = "gpt-4o-mini",
        retrieve_max_tokens: int = 2048,
        retrieve_target_tokens: int = 1800,
    ):
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
            prefix = self._render_heading_prefix(section.path)

            for group in self._partition_by_tables(body):
                parent_text = self._compose_parent_text(prefix, group)
                for content in self._iter_children(prefix, group):
                    if not content.strip():
                        continue
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
    # Section → parent groups (split at table boundaries)
    # ------------------------------------------------------------------

    def _partition_by_tables(self, body: str) -> List[_ParentGroup]:
        """Walk blocks; every encountered table closes a parent group with the
        accumulated preceding text as its description. Residual text after the
        final table becomes a trailing text-only parent."""
        blocks = self._split_blocks(body)
        groups: List[_ParentGroup] = []
        pending: List[str] = []

        for block in blocks:
            if self._looks_like_table(block):
                desc = "\n\n".join(pending).strip()
                groups.append(_ParentGroup(text=desc, table=block))
                pending = []
            else:
                pending.append(block)

        if pending:
            desc = "\n\n".join(pending).strip()
            if desc:
                groups.append(_ParentGroup(text=desc, table=None))

        if not groups:
            # Body was entirely whitespace once blocks were filtered; nothing
            # to emit — caller already guards on empty body but keep the
            # invariant explicit.
            return []
        return groups

    def _compose_parent_text(self, prefix: str, group: _ParentGroup) -> str:
        parts = []
        if group.text:
            parts.append(group.text)
        if group.table:
            parts.append(group.table)
        body = "\n\n".join(parts).strip()
        return (prefix + body).strip()

    # ------------------------------------------------------------------
    # Parent group → child chunks
    # ------------------------------------------------------------------

    def _iter_children(self, prefix: str, group: _ParentGroup) -> List[str]:
        out: List[str] = []

        if group.text:
            for piece in self._pack_text_paragraphs(group.text):
                out.append((prefix + piece).strip())

        if group.table is not None:
            table_md = group.table
            table_tokens = self._count(table_md)
            if table_tokens <= self.retrieve_max_tokens:
                out.append(self._build_table_chunk(prefix, group.text, table_md))
            else:
                out.extend(self._split_table_by_rows(prefix, table_md))

        return out

    # ------------------------------------------------------------------
    # Text packing
    # ------------------------------------------------------------------

    def _pack_text_paragraphs(self, text: str) -> List[str]:
        """Greedy-pack paragraph blocks (blank-line separated) up to
        ``retrieve_target_tokens``. Paragraphs that overshoot
        ``retrieve_max_tokens`` on their own are token-window split."""
        blocks = self._split_blocks(text)
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
            if block_tokens > self.retrieve_max_tokens:
                flush()
                out.extend(self._hard_split(block))
                continue
            if current and current_tokens + block_tokens > self.retrieve_target_tokens:
                flush()
            current.append(block)
            current_tokens += block_tokens
        flush()
        return out

    # ------------------------------------------------------------------
    # Table chunking
    # ------------------------------------------------------------------

    def _build_table_chunk(self, prefix: str, description: str, table_md: str) -> str:
        """Whole-table chunk: heading + description (table caption) + table."""
        parts = []
        if description:
            parts.append(description)
        parts.append(table_md)
        content = (prefix + "\n\n".join(parts)).strip()
        if self._count(content) <= self.retrieve_max_tokens:
            return content
        # Table + description + heading still oversize. Drop description first
        # (it is preserved in parent_text), then heading. Table itself is the
        # primary signal.
        content = (prefix + table_md).strip()
        if self._count(content) <= self.retrieve_max_tokens:
            return content
        return table_md.strip()

    def _split_table_by_rows(self, prefix: str, table_md: str) -> List[str]:
        """Per-row chunks for oversize tables. Each chunk = heading + header
        + separator + one data row. Rows that exceed ``retrieve_max_tokens``
        on their own are token-window split as a last resort."""
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
            chunk = f"{prefix}{header_block}\n{row}".strip()
            if self._count(chunk) <= self.retrieve_max_tokens:
                out.append(chunk)
            else:
                out.extend(self._hard_split(chunk))
        return out

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _split_blocks(self, body: str) -> List[str]:
        """Group lines into paragraph / table blocks separated by blank lines."""
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
        """Token-window fallback for content that overshoots the max on its own."""
        tokens = self.encoding.encode(block)
        out: List[str] = []
        size = self.retrieve_target_tokens
        for start in range(0, len(tokens), size):
            slice_ = tokens[start : start + size]
            out.append(self.encoding.decode(slice_).strip())
        return [s for s in out if s]

    def _count(self, text: str) -> int:
        return len(self.encoding.encode(text))
