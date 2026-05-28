"""hier_v2 splitter — block-bounded parent/child for text + LLM-described tables.

Pipeline (one document):

    sections   = split by markdown H1–H6 headings; carry section_path
    for each section:
      blocks   = split into ``text`` vs ``table`` blocks (strict)
      for each block:
        if text:
            parents = paragraph-pack to 2 × CHILD tokens
            for each parent:
              children = paragraph-pack to CHILD with OVERLAP tokens
              emit ``text_child`` rows: text = child, parent_text = parent,
                   embed_text = section_path + child
        if table:
            one LLM call (gemini-2.5-flash, JSON) →
                {"retrieval_text": "...", "generation_text": "..."}
            emit one ``table_summary``: text = generation_text,
                 embed_text = section_path + retrieval_text,
                 table_id = new uuid, parent_id = None
            slice table into row-aligned ≤ CHILD windows (header repeated)
            emit ``table_segment`` per slice: text = raw markdown,
                 embed_text = section_path + raw, parent_text = generation_text,
                 parent_id = table_id (so siblings collapse on dedup)

Every ``ChunkRow`` carries ``chunk_type``, ``embed_text``, ``parent_id``,
``parent_text``, ``table_id``, ``table_dataframe`` so the upsert / query
side knows what role the row plays.

The LLM call is cached on disk by md5(section_path + table_md) — reruns are
instant; first ingest pays once per distinct table.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import pickle
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import tiktoken

from app.configurations.configurations import settings

logger = logging.getLogger(__name__)


# ── Regexes -----------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
# Slightly permissive: 2+ dashes per cell (research_rag uses 3+; allow 2 too
# because some doc-parsing exports emit `|--|--|`).
TABLE_SEP_RE = re.compile(
    r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$"
)


# ── Prompts (verbatim from research_rag/table_split_gate_prompt.md) ---------

SUMMARY_SYSTEM = (
    "You describe a markdown table for a retrieval system. Your description must let "
    "(a) a search query about the table's content find it, and (b) a reader answer "
    "questions from your description alone, without seeing the table. Describe the "
    "SUBSTANCE of the information — what the table is about and what it tells you — "
    "not merely a list of column names. Output STRICT JSON only."
)

SUMMARY_USER_TMPL = (
    "Section path: {section_path}\n"
    "Context before the table:\n"
    "{context}\n\n"
    "Table:\n"
    "{table}\n\n"
    "The two fields play DIFFERENT roles — write them differently:\n\n"
    "- \"retrieval_text\" (this gets embedded for search): describe the table at a\n"
    "  GENERAL level. Do NOT quote specific row values — no individual names, IDs,\n"
    "  dates, or per-row data. 1-2 natural sentences stating what the table is\n"
    "  ABOUT and the kinds of entities / categories / dimensions of information\n"
    "  it captures, and what a reader can learn or look up from it. Do NOT just\n"
    "  enumerate column headers or say \"lists N rows with columns A, B, C\".\n\n"
    "- \"generation_text\" (this is fed to an LLM to ANSWER questions): be COMPLETE,\n"
    "  no word limit. ENUMERATE the table's actual content so the question can be\n"
    "  answered from this text alone without seeing the table: go row by row and\n"
    "  state each item together with its relevant attributes. Preserve the\n"
    "  specific values, names, ids, and numbers. Keep it readable prose or a\n"
    "  clear list, but do not omit rows.\n\n"
    "Output JSON only:\n"
    "{{\"retrieval_text\": \"...\", \"generation_text\": \"...\"}}"
)


# ── Row dataclass -----------------------------------------------------------

@dataclass
class ChunkRow:
    """One emitted chunk. Mirrors the ``chunk`` table columns the upsert
    pipeline writes."""
    id: str
    content: str                 # the verbatim chunk (what generation sees)
    parent_id: Optional[str]     # parent window id (text) or table_id (segment)
    parent_text: str             # parent window verbatim or LLM enumeration
    chunk_order_index: int
    tokens: int
    heading_path: Optional[str] = None

    # hier_v2 additions
    chunk_type: str = "text_child"           # text_child | table_summary | table_segment
    embed_text: str = ""                     # what the embedder reads
    table_id: Optional[str] = None
    table_dataframe: Optional[str] = None    # base64(pickle(df.to_dict())) for tables


@dataclass
class _Section:
    level: int
    title: str
    path: List[str] = field(default_factory=list)
    body_lines: List[str] = field(default_factory=list)
    children: List["_Section"] = field(default_factory=list)


# ── LLM cache (md5-keyed JSON files) ----------------------------------------

def _cache_key(section_path: str, table_md: str) -> str:
    h = hashlib.md5()
    h.update(section_path.encode("utf-8"))
    h.update(b"\x1f")
    h.update(table_md.encode("utf-8"))
    return h.hexdigest()


def _cache_get(cache_dir: Path, key: str) -> Optional[dict]:
    p = cache_dir / f"sum_{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return None


def _cache_put(cache_dir: Path, key: str, value: dict) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"sum_{key}.json").write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("hier_v2 cache write failed: %s", exc)


# ── Helpers: table → DataFrame base64 (for downstream consumers) ------------

def _md_table_to_dataframe_b64(md_table: str) -> Optional[str]:
    """Pandas-dict pickle, base64-encoded. Decoded by callers via
    ``pickle.loads(base64.b64decode(...))`` → ``pd.DataFrame(dict_)``.
    Returns None if the table can't be parsed."""
    try:
        import pandas as pd  # local import — keep splitter usable without pandas
    except ImportError:
        return None

    rows = []
    for line in md_table.splitlines():
        s = line.strip()
        if not s or not s.startswith("|"):
            continue
        if TABLE_SEP_RE.match(s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return None

    header = rows[0]
    width = len(header)
    data = []
    for r in rows[1:]:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        data.append(r)
    try:
        df = pd.DataFrame(data, columns=header)
        return base64.b64encode(pickle.dumps(df.to_dict())).decode("ascii")
    except Exception:
        return None


# ── HierV2Splitter ----------------------------------------------------------

class MarkdownSplitter:
    """hier_v2 splitter — replaces the previous v4 block splitter.

    Kept the original class name so the worker container (which constructs
    ``MarkdownSplitter`` from ``container.py``) doesn't need re-wiring."""

    def __init__(
        self,
        tokenizer_model: str = "gpt-4o-mini",
        # Legacy aliases kept to preserve container.py's call signature.
        retrieve_max_tokens: int = 512,
        retrieve_target_tokens: int = 512,
        child_chunk_size: Optional[int] = None,
        chunk_overlap: int = 50,
        overlap_rows: int = 1,
        llm_chat=None,                     # async (system, user, model=...) → str
        llm_model: Optional[str] = None,
        cache_dir: Optional[str] = None,
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

        # child_chunk_size wins if set; otherwise use the legacy aliases.
        self.child_tokens = child_chunk_size or settings.HIER_V2_CHILD_TOKENS or retrieve_target_tokens
        self.parent_tokens = 2 * self.child_tokens
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.HIER_V2_OVERLAP_TOKENS
        self.overlap_rows = overlap_rows if overlap_rows is not None else settings.HIER_V2_OVERLAP_ROWS
        self.llm_chat = llm_chat
        self.llm_model = llm_model or settings.HIER_V2_TABLE_LLM_MODEL
        self.cache_dir = Path(cache_dir or settings.HIER_V2_CACHE_DIR)

    # ------------------------------------------------------------------
    # Sync entry point (back-compat — container.py calls splitter.split()).
    # Runs the async pipeline on a private event loop. Inside Celery the
    # outer task already does asyncio.run for its own coroutine, so this
    # nested loop is fine.
    # ------------------------------------------------------------------

    def split(self, text: str) -> List[ChunkRow]:
        if not text or not text.strip():
            return []
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Caller is async — schedule the coroutine on a worker loop.
                return asyncio.run_coroutine_threadsafe(
                    self.asplit(text), loop,
                ).result()
        except RuntimeError:
            pass
        return asyncio.run(self.asplit(text))

    # ------------------------------------------------------------------
    # Async entry point — the splitter is fully async because the per-table
    # LLM call lives inside this pipeline.
    # ------------------------------------------------------------------

    async def asplit(self, text: str) -> List[ChunkRow]:
        if not text or not text.strip():
            return []

        rows: List[ChunkRow] = []
        order = 0

        for section in self._iter_leaf_sections(self._build_tree(text)):
            body = "\n".join(section.body_lines).strip()
            if not body:
                continue
            section_path = " > ".join(section.path) if section.path else ""
            heading_path = section_path or None
            prefix_for_embed = (section_path + "\n\n") if section_path else ""

            text_context_carry = ""   # last text block's tail — feeds table prompt context

            for kind, payload in self._split_into_blocks(body):
                if kind == "text":
                    text_context_carry = payload[-1200:]
                    for parent_text, child_text in self._text_parent_children(payload):
                        rows.append(ChunkRow(
                            id=str(uuid.uuid4()),
                            content=child_text,
                            parent_id=self._make_parent_id_for(parent_text, section_path),
                            parent_text=parent_text,
                            chunk_order_index=order,
                            tokens=self._count(child_text),
                            heading_path=heading_path,
                            chunk_type="text_child",
                            embed_text=prefix_for_embed + child_text,
                        ))
                        order += 1
                elif kind == "table":
                    table_md = payload
                    table_id = str(uuid.uuid4())
                    summary = await self._table_summary(
                        section_path=section_path,
                        context=text_context_carry,
                        table_md=table_md,
                    )
                    df_b64 = _md_table_to_dataframe_b64(table_md)

                    # 1) table_summary row
                    rows.append(ChunkRow(
                        id=str(uuid.uuid4()),
                        content=summary["generation_text"],
                        parent_id=None,
                        parent_text="",
                        chunk_order_index=order,
                        tokens=self._count(summary["generation_text"]),
                        heading_path=heading_path,
                        chunk_type="table_summary",
                        embed_text=prefix_for_embed + summary["retrieval_text"],
                        table_id=table_id,
                        table_dataframe=df_b64,
                    ))
                    order += 1

                    # 2) row-aligned segments (header repeated)
                    for segment in self._slice_table_segments(table_md):
                        rows.append(ChunkRow(
                            id=str(uuid.uuid4()),
                            content=segment,
                            parent_id=table_id,
                            parent_text=summary["generation_text"],
                            chunk_order_index=order,
                            tokens=self._count(segment),
                            heading_path=heading_path,
                            chunk_type="table_segment",
                            embed_text=prefix_for_embed + segment,
                            table_id=table_id,
                            table_dataframe=df_b64,
                        ))
                        order += 1

        return rows

    # ------------------------------------------------------------------
    # Section tree (H1–H6 → nested sections with section_path breadcrumb).
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
        # A non-leaf section may itself carry intro body lines (text before
        # the first sub-heading). Emit those too as a synthetic leaf so they
        # aren't dropped.
        intro = "\n".join(node.body_lines).strip()
        if intro and node.level > 0:
            yield _Section(level=node.level, title=node.title, path=node.path,
                           body_lines=node.body_lines)
        for child in node.children:
            yield from self._iter_leaf_sections(child)

    # ------------------------------------------------------------------
    # Block split — text vs table (strict by separator regex).
    # ------------------------------------------------------------------

    def _split_into_blocks(self, body: str) -> List[Tuple[str, str]]:
        lines = body.splitlines()
        blocks: List[Tuple[str, str]] = []
        i = 0
        n = len(lines)
        while i < n:
            # Detect table at i, i+1.
            if (
                i + 1 < n
                and TABLE_LINE_RE.match(lines[i])
                and TABLE_SEP_RE.match(lines[i + 1])
            ):
                j = i + 2
                while j < n and lines[j].strip().startswith("|"):
                    j += 1
                tbl = "\n".join(lines[i:j]).strip()
                if tbl:
                    blocks.append(("table", tbl))
                i = j
                continue
            # Otherwise accumulate text until the next table or EOF.
            j = i
            while j < n:
                if (
                    j + 1 < n
                    and TABLE_LINE_RE.match(lines[j])
                    and TABLE_SEP_RE.match(lines[j + 1])
                ):
                    break
                j += 1
            text_block = "\n".join(lines[i:j]).strip()
            if text_block:
                blocks.append(("text", text_block))
            i = j
        return blocks

    # ------------------------------------------------------------------
    # Text → parent (2×CHILD) → children (CHILD with overlap).
    # ------------------------------------------------------------------

    def _text_parent_children(self, text: str):
        """Yield ``(parent_text, child_text)`` pairs.

        Paragraph-pack into parent windows ≤ ``parent_tokens``; inside each
        parent, paragraph-pack to ``child_tokens`` with ``chunk_overlap``
        token overlap between sibling children. Overlong paragraphs fall
        back to token-window slicing."""
        paragraphs = self._split_paragraphs(text)
        for parent_text in self._pack(paragraphs, self.parent_tokens):
            parent_paragraphs = self._split_paragraphs(parent_text)
            children = list(self._pack(parent_paragraphs, self.child_tokens))
            # Add token-level overlap between consecutive children.
            children = self._apply_overlap(children, self.chunk_overlap)
            for c in children:
                if c.strip():
                    yield parent_text, c

    def _split_paragraphs(self, text: str) -> List[str]:
        blocks: List[str] = []
        buf: List[str] = []
        for line in text.splitlines():
            if line.strip() == "":
                if buf:
                    blocks.append("\n".join(buf).rstrip())
                    buf = []
            else:
                buf.append(line)
        if buf:
            blocks.append("\n".join(buf).rstrip())
        return [b for b in blocks if b]

    def _pack(self, paragraphs: List[str], cap: int) -> List[str]:
        out: List[str] = []
        cur: List[str] = []
        cur_tok = 0
        for p in paragraphs:
            t = self._count(p)
            if t > cap:
                # flush, then hard-split the oversized paragraph
                if cur:
                    out.append("\n\n".join(cur).strip())
                    cur, cur_tok = [], 0
                out.extend(self._hard_split(p, cap))
                continue
            if cur and cur_tok + t > cap:
                out.append("\n\n".join(cur).strip())
                cur, cur_tok = [], 0
            cur.append(p)
            cur_tok += t
        if cur:
            out.append("\n\n".join(cur).strip())
        return [b for b in out if b]

    def _hard_split(self, block: str, cap: int) -> List[str]:
        tokens = self.encoding.encode(block)
        return [
            self.encoding.decode(tokens[s : s + cap]).strip()
            for s in range(0, len(tokens), cap)
        ]

    def _apply_overlap(self, children: List[str], overlap_tokens: int) -> List[str]:
        if overlap_tokens <= 0 or len(children) < 2:
            return children
        out = [children[0]]
        for prev, curr in zip(children, children[1:]):
            prev_tail_tokens = self.encoding.encode(prev)[-overlap_tokens:]
            tail = self.encoding.decode(prev_tail_tokens).strip()
            out.append((tail + "\n" + curr).strip() if tail else curr)
        return out

    # ------------------------------------------------------------------
    # Table summary — one LLM call per table, file-cached.
    # ------------------------------------------------------------------

    async def _table_summary(
        self, *, section_path: str, context: str, table_md: str,
    ) -> dict:
        key = _cache_key(section_path, table_md)
        cached = _cache_get(self.cache_dir, key)
        if cached:
            return cached

        if self.llm_chat is None:
            # No LLM wired → degrade gracefully. Embedding still works on
            # raw segments; only the summary chunk is impoverished.
            out = {
                "retrieval_text": f"Table in {section_path}." if section_path else "Table.",
                "generation_text": table_md,
            }
            _cache_put(self.cache_dir, key, out)
            return out

        user = SUMMARY_USER_TMPL.format(
            section_path=section_path or "(root)",
            context=context or "(none)",
            table=table_md,
        )
        try:
            raw = await self.llm_chat(
                SUMMARY_SYSTEM, user, model=self.llm_model,
            )
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("non-dict")
            out = {
                "retrieval_text": (parsed.get("retrieval_text") or "").strip()
                                  or f"Table in {section_path}.",
                "generation_text": (parsed.get("generation_text") or "").strip()
                                   or table_md,
            }
        except Exception as exc:
            logger.warning("hier_v2 table summary failed (%s); using fallback", exc)
            out = {
                "retrieval_text": f"Table in {section_path}." if section_path else "Table.",
                "generation_text": table_md,
            }
        _cache_put(self.cache_dir, key, out)
        return out

    # ------------------------------------------------------------------
    # Table segments — row-aligned, header repeated at top of every slice.
    # ------------------------------------------------------------------

    def _slice_table_segments(self, table_md: str) -> List[str]:
        lines = [ln for ln in table_md.splitlines() if ln.strip()]
        if len(lines) < 3:
            return [table_md]
        header, sep = lines[0], lines[1]
        data = lines[2:]
        header_block = f"{header}\n{sep}"
        header_tok = self._count(header_block)
        body_budget = max(self.child_tokens - header_tok, 64)

        segments: List[str] = []
        i = 0
        while i < len(data):
            cur_rows: List[str] = []
            cur_tok = 0
            j = i
            while j < len(data):
                row_tok = self._count(data[j])
                if cur_rows and cur_tok + row_tok > body_budget:
                    break
                cur_rows.append(data[j])
                cur_tok += row_tok
                j += 1
            if not cur_rows:
                # one row alone is too big — hard-split it as text (rare)
                segments.extend(self._hard_split(
                    header_block + "\n" + data[i], self.child_tokens,
                ))
                i += 1
                continue
            segments.append(header_block + "\n" + "\n".join(cur_rows))
            # advance with row overlap
            i = max(j - self.overlap_rows, i + 1)
        return segments

    # ------------------------------------------------------------------
    # Misc.
    # ------------------------------------------------------------------

    def _make_parent_id_for(self, parent_text: str, section_path: str) -> str:
        # Deterministic id per (section_path, parent_text) so sibling
        # children retrieved by different sub-queries collapse on dedup.
        h = hashlib.md5()
        h.update(section_path.encode("utf-8"))
        h.update(b"\x1f")
        h.update(parent_text.encode("utf-8"))
        return str(uuid.UUID(h.hexdigest()))

    def _count(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))
