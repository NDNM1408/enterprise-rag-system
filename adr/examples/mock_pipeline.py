"""Mock pipeline minh hoạ section-as-paragraph chunking.

Đây là phiên bản RÚT GỌN của ``data-processing-job/src/app/application/core/
markdown_splitter.py`` — chỉ giữ logic cốt lõi để mô tả trong ADR. Không
import các module của repo, không cần tokenizer thật — đếm token bằng
``len(text.split())`` cho đơn giản. KHÔNG dùng trong production.

Chạy:
    python adr/examples/mock_pipeline.py adr/examples/sample_input.md
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

# Production: 512 với content_prefix_mode="deepest". Mock dùng bigger budget
# để chạy demo dễ thấy "section fits = 1 chunk" trên sample input.
MAX_TOKENS = 200
PREFIX_MODE = "deepest"  # "full" | "deepest" | "none"


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    content: str             # what gets embedded
    parent_text: str         # what the LLM sees on hit
    heading_path: Optional[str]
    tokens: int


@dataclass
class Section:
    level: int                              # 0 = synthetic root for preamble
    title: str
    path: list[str] = field(default_factory=list)
    body_lines: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)


# ── Token counting (mock: word count instead of tiktoken) ──────────────────

def count_tokens(text: str) -> int:
    return len(text.split())


# ── Step 1: build heading tree ─────────────────────────────────────────────

def build_tree(md: str) -> Section:
    root = Section(level=0, title="", path=[])
    stack: list[Section] = [root]
    for line in md.splitlines():
        m = HEADING_RE.match(line)
        if not m:
            stack[-1].body_lines.append(line)
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else root
        node = Section(level=level, title=title, path=parent.path + [title])
        parent.children.append(node)
        stack.append(node)
    return root


# ── Step 2: iterate sections with body (preamble fix included) ─────────────

def iter_sections_with_body(node: Section):
    """Yield every section that owns body text — INCLUDING non-leaf nodes
    whose body precedes their children (= section intros + preamble)."""
    if "".join(node.body_lines).strip():
        yield node
    for child in node.children:
        yield from iter_sections_with_body(child)


# ── Step 3: prefix rendering per mode ──────────────────────────────────────

def render_prefix(path: list[str], mode: str) -> str:
    if mode == "none" or not path:
        return ""
    if mode == "deepest":
        return f"{'#' * len(path)} {path[-1]}\n\n"
    return "\n".join(f"{'#' * (i + 1)} {h}" for i, h in enumerate(path)) + "\n\n"


# ── Step 4: split body into paragraph/table blocks ─────────────────────────

def split_blocks(body: str) -> list[str]:
    blocks: list[str] = []
    buf: list[str] = []
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


def looks_like_table(block: str) -> bool:
    lines = block.splitlines()
    return (
        len(lines) >= 2
        and TABLE_LINE_RE.match(lines[0])
        and TABLE_SEP_RE.match(lines[1])
    ) is True


def split_oversized_section(body: str, prefix: str, max_tokens: int) -> list[str]:
    blocks = split_blocks(body)
    budget = max(1, max_tokens - count_tokens(prefix))
    out, current, current_tok = [], [], 0

    def flush():
        nonlocal current, current_tok
        if current:
            out.append("\n\n".join(current).strip())
        current, current_tok = [], 0

    for block in blocks:
        bt = count_tokens(block)
        if bt > budget:
            flush()
            # Could implement table-row-split or hard-split here.
            # Mock just emits the oversized block as-is to keep code short.
            out.append(block)
            continue
        if current_tok + bt > budget and current:
            flush()
        current.append(block)
        current_tok += bt
    flush()
    return out


# ── Step 5: top-level split ────────────────────────────────────────────────

def split_markdown(md: str, max_tokens: int = MAX_TOKENS, prefix_mode: str = PREFIX_MODE) -> list[Chunk]:
    """Section-as-paragraph: 1 section = 1 chunk if fits; else split into
    children sharing the full section as parent_text."""
    root = build_tree(md)
    rows: list[Chunk] = []

    for section in iter_sections_with_body(root):
        body = "\n".join(section.body_lines).strip()
        if not body:
            continue

        heading_path = " > ".join(section.path) if section.path else None
        parent_prefix = render_prefix(section.path, "full")        # always FULL for parent
        content_prefix = render_prefix(section.path, prefix_mode)  # configurable for content
        parent_text = (parent_prefix + body).strip()

        # Fast path: section fits in budget — one chunk.
        single = (content_prefix + body).strip()
        if count_tokens(single) <= max_tokens:
            rows.append(Chunk(
                content=single,
                parent_text=parent_text,
                heading_path=heading_path,
                tokens=count_tokens(single),
            ))
            continue

        # Overflow path: split into children, all share parent.
        for piece in split_oversized_section(body, content_prefix, max_tokens):
            content = (content_prefix + piece).strip()
            rows.append(Chunk(
                content=content,
                parent_text=parent_text,
                heading_path=heading_path,
                tokens=count_tokens(content),
            ))

    return rows


# ── Demo ────────────────────────────────────────────────────────────────────

def main(path: str) -> None:
    md = Path(path).read_text()
    chunks = split_markdown(md, max_tokens=MAX_TOKENS, prefix_mode=PREFIX_MODE)
    print(f"Input: {path}  ({count_tokens(md)} tokens)")
    print(f"Config: max_tokens={MAX_TOKENS}, prefix_mode={PREFIX_MODE!r}")
    print(f"→ {len(chunks)} chunks emitted\n")
    for i, c in enumerate(chunks):
        same = "(content == parent)" if c.content == c.parent_text else "(differ)"
        print(f"── chunk #{i+1}  heading_path={c.heading_path!r}  tokens={c.tokens}  {same}")
        print("  content:")
        for line in c.content.splitlines()[:6]:
            print(f"    {line}")
        if c.content != c.parent_text:
            print("  parent_text (first 6 lines):")
            for line in c.parent_text.splitlines()[:6]:
                print(f"    {line}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "adr/examples/sample_input.md")
