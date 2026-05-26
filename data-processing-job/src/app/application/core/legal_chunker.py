"""Vietnamese legal-corpus markdown chunker.

Splits markdown into chunks whose granularity matches Vietnamese legal
structure (Điều / Chương / Mục). Borrowed from
NDNM1408/llm-wiki-elasticsearch's ``raw_chunker`` with light adjustments to
fit this project's chunk schema (we track ``ordinal`` and ``heading_path``
per chunk so retrieval can surface section context).

Priority cascade (first match wins):
  1. ``Điều \\d+\\.?``     — each article = 1 chunk.
  2. ``^## ``               — H2 sections, if at least 3 present.
  3. ``^# / ## / ###``      — any heading depth, smaller files.
  4. Paragraph sliding window (≈ 1500 char / 200 char overlap).

Post-processing:
  - chunks > ``MAX_CHARS`` are subsplit on paragraph then sentence boundaries
  - chunks < ``MIN_CHARS`` are merged into the previous chunk when they share
    the same parent section_label
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------

TARGET_CHARS = 1500     # ideal chunk size for the fallback sliding window
OVERLAP_CHARS = 200
MAX_CHARS = 2500        # split anything larger
MIN_CHARS = 200         # merge anything smaller (when section_label matches)


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"^(Điều\s+\d+\.?)", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_H2_RE = re.compile(r"^## ", re.MULTILINE)
_PARA_BREAK_RE = re.compile(r"\n\s*\n+")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[\.\?!])\s+(?=[A-ZĐÁÀẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÉÈẺẼẸÊỀẾỂỄỆÍÌỈĨỊÓÒỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÚÙỦŨỤƯỪỨỬỮỰÝỲỶỸỴ])")


@dataclass
class LegalChunk:
    content: str
    section_label: Optional[str]
    heading_path: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    ordinal: int = 0


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def chunk_legal_markdown(text: str) -> List[LegalChunk]:
    """Apply the priority cascade and return ordered chunks."""
    if not text or not text.strip():
        return []

    lines = text.splitlines()

    if len(_ARTICLE_RE.findall(text)) >= 2:
        raw = _split_by_articles(lines)
    elif len(_H2_RE.findall(text)) >= 3:
        raw = _split_by_heading(lines, max_level=2)
    elif _HEADING_RE.search(text):
        raw = _split_by_heading(lines, max_level=3)
    else:
        raw = _split_sliding_window(text)

    refined = _postprocess(raw)

    for i, chunk in enumerate(refined):
        chunk.ordinal = i
    return refined


# ----------------------------------------------------------------------
# Strategy 1 — split by ``Điều N.`` (priority for Vietnamese legal texts)
# ----------------------------------------------------------------------

def _split_by_articles(lines: List[str]) -> List[LegalChunk]:
    """Each ``Điều X.`` line opens a new chunk; lines before the first article
    become a preamble chunk if non-empty."""
    article_starts: List[int] = []
    headings: List[tuple] = []  # (line_no, level, title)

    for idx, raw in enumerate(lines):
        m = _HEADING_RE.match(raw)
        if m:
            headings.append((idx, len(m.group(1)), m.group(2).strip()))
        if _ARTICLE_RE.match(raw):
            article_starts.append(idx)

    if not article_starts:
        return []

    boundaries = article_starts + [len(lines)]
    chunks: List[LegalChunk] = []

    # Preamble (anything before the first article)
    first = boundaries[0]
    if first > 0:
        body = "\n".join(lines[:first]).strip()
        if body:
            chunks.append(LegalChunk(
                content=body,
                section_label=None,
                heading_path=_heading_path_at(headings, 0),
                start_line=1,
                end_line=first,
            ))

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        body = "\n".join(lines[start:end]).rstrip()
        if not body.strip():
            continue
        label_match = _ARTICLE_RE.match(lines[start])
        label = label_match.group(1).rstrip(".") if label_match else None
        chunks.append(LegalChunk(
            content=body,
            section_label=label,
            heading_path=_heading_path_at(headings, start),
            start_line=start + 1,
            end_line=end,
        ))
    return chunks


# ----------------------------------------------------------------------
# Strategy 2 — split by markdown heading depth
# ----------------------------------------------------------------------

def _split_by_heading(lines: List[str], max_level: int) -> List[LegalChunk]:
    """Open a new chunk every time a heading at depth <= ``max_level`` appears."""
    sections: List[tuple] = []  # (start_idx, label, heading_path)
    heading_stack: List[tuple] = []

    for idx, raw in enumerate(lines):
        m = _HEADING_RE.match(raw)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        if level <= max_level:
            sections.append((
                idx,
                title,
                [t for _, t in heading_stack],
            ))

    if not sections:
        return []

    boundaries = [s[0] for s in sections] + [len(lines)]
    chunks: List[LegalChunk] = []

    # Preamble before the first matching heading.
    first = boundaries[0]
    if first > 0:
        body = "\n".join(lines[:first]).strip()
        if body:
            chunks.append(LegalChunk(
                content=body,
                section_label=None,
                heading_path=[],
                start_line=1,
                end_line=first,
            ))

    for i, (start_idx, label, path) in enumerate(sections):
        end = boundaries[i + 1]
        body = "\n".join(lines[start_idx:end]).rstrip()
        if not body.strip():
            continue
        chunks.append(LegalChunk(
            content=body,
            section_label=label,
            heading_path=path,
            start_line=start_idx + 1,
            end_line=end,
        ))
    return chunks


# ----------------------------------------------------------------------
# Strategy 3 — paragraph-aware sliding window
# ----------------------------------------------------------------------

def _split_sliding_window(text: str) -> List[LegalChunk]:
    paragraphs = [p.strip() for p in _PARA_BREAK_RE.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[LegalChunk] = []
    buf: List[str] = []
    size = 0

    for para in paragraphs:
        if size + len(para) > TARGET_CHARS and buf:
            chunks.append(LegalChunk(
                content="\n\n".join(buf),
                section_label=None,
                heading_path=[],
            ))
            # Overlap: keep the tail of the previous chunk
            tail = "\n\n".join(buf)[-OVERLAP_CHARS:]
            buf = [tail, para] if tail else [para]
            size = sum(len(b) for b in buf)
        else:
            buf.append(para)
            size += len(para)

    if buf:
        chunks.append(LegalChunk(
            content="\n\n".join(buf),
            section_label=None,
            heading_path=[],
        ))
    return chunks


# ----------------------------------------------------------------------
# Post-processing — split oversize, merge undersize
# ----------------------------------------------------------------------

def _postprocess(chunks: List[LegalChunk]) -> List[LegalChunk]:
    expanded: List[LegalChunk] = []
    for ch in chunks:
        if len(ch.content) > MAX_CHARS:
            expanded.extend(_subsplit(ch))
        else:
            expanded.append(ch)

    # Merge tiny chunks into the previous one when they share section_label.
    merged: List[LegalChunk] = []
    for ch in expanded:
        if (
            merged
            and len(ch.content) < MIN_CHARS
            and merged[-1].section_label == ch.section_label
            and len(merged[-1].content) + len(ch.content) <= MAX_CHARS
        ):
            merged[-1].content = merged[-1].content.rstrip() + "\n\n" + ch.content.lstrip()
            merged[-1].end_line = ch.end_line or merged[-1].end_line
        else:
            merged.append(ch)
    return merged


def _subsplit(chunk: LegalChunk) -> List[LegalChunk]:
    """Split a too-big chunk on paragraph then sentence boundaries."""
    pieces = [p for p in _PARA_BREAK_RE.split(chunk.content) if p.strip()]
    if len(pieces) <= 1:
        pieces = [s for s in _SENTENCE_BREAK_RE.split(chunk.content) if s.strip()]

    out: List[LegalChunk] = []
    buf: List[str] = []
    size = 0
    for piece in pieces:
        if size + len(piece) > TARGET_CHARS and buf:
            out.append(LegalChunk(
                content="\n\n".join(buf),
                section_label=chunk.section_label,
                heading_path=list(chunk.heading_path),
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            ))
            buf = [piece]
            size = len(piece)
        else:
            buf.append(piece)
            size += len(piece)
    if buf:
        out.append(LegalChunk(
            content="\n\n".join(buf),
            section_label=chunk.section_label,
            heading_path=list(chunk.heading_path),
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        ))
    return out


# ----------------------------------------------------------------------
# Heading path lookup
# ----------------------------------------------------------------------

def _heading_path_at(headings: List[tuple], line_idx: int) -> List[str]:
    """Walk headings declared at or before ``line_idx`` and rebuild the stack."""
    stack: List[tuple] = []
    for h_line, level, title in headings:
        if h_line > line_idx:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return [t for _, t in stack]
