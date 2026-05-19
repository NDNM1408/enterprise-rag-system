"""HTML → markdown.

Walk the BeautifulSoup tree once, emitting markdown for the structural tags
(``h1``-``h6``, ``p``, ``ul``/``ol``, ``table``, ``pre``/``code``, etc.).
Inspired by ragflow's ``RAGFlowHtmlParser`` but oriented at structure
preservation rather than token-budget chunking.
"""
from __future__ import annotations

from io import BytesIO

import chardet
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from core.base import BaseParser, ParseResult
from core.compat import find_codec

_HEADING = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}
_INLINE = {"a", "span", "b", "strong", "i", "em", "u", "small", "sub", "sup",
           "mark", "abbr", "cite", "q", "kbd", "var", "samp", "time"}


class HtmlParser(BaseParser):
    name = "html"
    extensions = ("html", "htm")

    def parse(self, payload: bytes, filename: str) -> ParseResult:
        encoding = find_codec(payload) if payload else "utf-8"
        text = payload.decode(encoding, errors="ignore")
        soup = BeautifulSoup(text, "html5lib")

        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            c.extract()
        for tag in soup.find_all(True):
            if "style" in tag.attrs:
                del tag.attrs["style"]

        root = soup.body or soup
        out: list[str] = []
        _render(root, out)
        markdown = _normalize("\n\n".join(p for p in out if p and p.strip()))

        title = soup.title.string.strip() if soup.title and soup.title.string else None
        return ParseResult(
            markdown=markdown,
            parser=self.name,
            page_count=0,
            metadata={"extension": filename.rsplit(".", 1)[-1].lower(), "title": title},
        )


def _normalize(s: str) -> str:
    # Collapse runs of >2 blank lines.
    lines = s.splitlines()
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line.rstrip())
    return "\n".join(out).strip()


def _render(node: Tag, out: list[str]) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                out.append(t)
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        if name in _HEADING:
            txt = _inline_text(child)
            if txt:
                out.append(f"{_HEADING[name]} {txt}")
        elif name == "p":
            txt = _inline_text(child)
            if txt:
                out.append(txt)
        elif name == "br":
            out.append("")
        elif name == "hr":
            out.append("---")
        elif name in {"ul", "ol"}:
            out.append(_list_to_md(child))
        elif name == "li":
            txt = _inline_text(child)
            if txt:
                out.append(f"- {txt}")
        elif name == "table":
            md = _table_to_md(child)
            if md:
                out.append(md)
        elif name in {"pre", "code"}:
            block_text = child.get_text("\n", strip=True) if name == "pre" else child.get_text(strip=True)
            if block_text:
                if name == "pre":
                    out.append(f"```\n{block_text}\n```")
                else:
                    out.append(f"`{block_text}`")
        elif name == "blockquote":
            txt = _inline_text(child)
            if txt:
                out.append("\n".join(f"> {ln}" for ln in txt.splitlines()))
        elif name == "img":
            src = child.get("src", "")
            alt = child.get("alt", "")
            if src:
                out.append(f"![{alt}]({src})")
        elif name == "a" and child.find(["table", "ul", "ol", "pre"]):
            # Anchors that wrap block content — recurse so nested blocks render.
            _render(child, out)
        elif name in {"div", "section", "article", "main", "aside", "header", "footer", "figure", "figcaption", "nav"}:
            _render(child, out)
        elif name in _INLINE:
            txt = _inline_text(child)
            if txt:
                out.append(txt)
        else:
            _render(child, out)


def _inline_text(node: Tag) -> str:
    parts: list[str] = []
    for c in node.children:
        if isinstance(c, NavigableString):
            parts.append(str(c))
        elif isinstance(c, Tag):
            n = c.name.lower()
            inner = _inline_text(c)
            if not inner:
                continue
            if n in {"strong", "b"}:
                parts.append(f"**{inner}**")
            elif n in {"em", "i"}:
                parts.append(f"*{inner}*")
            elif n in {"code", "kbd", "samp", "var"}:
                parts.append(f"`{inner}`")
            elif n == "a":
                href = c.get("href", "")
                parts.append(f"[{inner}]({href})" if href else inner)
            elif n == "br":
                parts.append("\n")
            elif n == "img":
                src = c.get("src", "")
                alt = c.get("alt", "")
                if src:
                    parts.append(f"![{alt}]({src})")
            else:
                parts.append(inner)
    return " ".join(t.strip() for t in "".join(parts).split() if t.strip())


def _list_to_md(node: Tag, depth: int = 0) -> str:
    ordered = node.name.lower() == "ol"
    lines: list[str] = []
    idx = 1
    for li in node.find_all("li", recursive=False):
        marker = f"{idx}." if ordered else "-"
        text = _inline_text(li)
        prefix = "  " * depth + f"{marker} "
        if text:
            lines.append(prefix + text)
        for sub in li.find_all(["ul", "ol"], recursive=False):
            lines.append(_list_to_md(sub, depth + 1))
        idx += 1
    return "\n".join(lines)


def _table_to_md(node: Tag) -> str:
    rows = []
    for tr in node.find_all("tr"):
        cells = [c.get_text(" ", strip=True).replace("|", r"\|") for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join(["---"] * n_cols) + " |"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)
