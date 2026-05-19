"""DOCX → markdown.

Walks the document body in true reading order so paragraphs, tables, and
images stay interleaved. Falls back to ``RAGFlowDocxParser`` for table
content when paragraph walking fails.
"""
from __future__ import annotations

import base64
import logging
import re
from io import BytesIO

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from core.base import BaseParser, ExtractedImage, ParseResult

log = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^Heading\s+([1-6])$", re.IGNORECASE)


def _style_to_md(style_name: str | None, text: str) -> str:
    if not text:
        return ""
    name = (style_name or "").strip()
    if not name:
        return text
    if name.lower() == "title":
        return f"# {text}"
    m = _HEADING_RE.match(name)
    if m:
        return f"{'#' * int(m.group(1))} {text}"
    if name.lower().startswith("list bullet"):
        return f"- {text}"
    if name.lower().startswith("list number"):
        return f"1. {text}"
    if name.lower() in {"quote", "intense quote"}:
        return f"> {text}"
    if name.lower() in {"caption"}:
        return f"_{text}_"
    return text


def _table_to_md(table) -> str:
    rows = [[cell.text.replace("\n", " ").strip() for cell in row.cells] for row in table.rows]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    cols = len(header)
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * cols) + " |"]
    for r in body:
        if len(r) < cols:
            r = r + [""] * (cols - len(r))
        elif len(r) > cols:
            r = r[:cols]
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


class DocxParser(BaseParser):
    name = "docx"
    extensions = ("docx",)

    def parse(self, payload: bytes, filename: str) -> ParseResult:
        doc = Document(BytesIO(payload))
        body = doc.element.body
        out: list[str] = []
        images: list[ExtractedImage] = []
        image_idx = 0

        for child in body.iterchildren():
            tag = child.tag
            if tag == qn("w:p"):
                # Paragraph (may contain inline images).
                para = Paragraph(child, doc)
                text = para.text.strip()
                inline_images = self._extract_paragraph_images(doc, para)
                if inline_images:
                    for blob, suffix in inline_images:
                        image_idx += 1
                        name = f"image_{image_idx}.{suffix or 'png'}"
                        mime = f"image/{ 'jpeg' if suffix in ('jpg','jpeg') else (suffix or 'png') }"
                        images.append(ExtractedImage(
                            name=name,
                            bytes_b64=base64.b64encode(blob).decode("ascii"),
                            mime=mime,
                        ))
                        out.append(f"![{name}]({name})")
                if text:
                    rendered = _style_to_md(
                        para.style.name if para.style is not None else None,
                        text,
                    )
                    out.append(rendered)
            elif tag == qn("w:tbl"):
                table = Table(child, doc)
                md = _table_to_md(table)
                if md:
                    out.append(md)
            # Other tags (sectPr, etc.) are skipped.

        markdown = "\n\n".join(s for s in out if s).strip()
        return ParseResult(
            markdown=markdown,
            parser=self.name,
            page_count=0,
            metadata={"extension": "docx", "image_count": len(images)},
            images=images,
        )

    @staticmethod
    def _extract_paragraph_images(doc, paragraph) -> list[tuple[bytes, str]]:
        out: list[tuple[bytes, str]] = []
        for img in paragraph._element.xpath(".//pic:pic"):
            embed = img.xpath(".//a:blip/@r:embed")
            if not embed:
                continue
            try:
                related_part = doc.part.related_parts[embed[0]]
            except KeyError:
                continue
            blob: bytes | None = None
            try:
                if related_part.image is not None:
                    blob = related_part.image.blob
            except Exception:
                blob = None
            if blob is None:
                blob = getattr(related_part, "blob", None)
            if not blob:
                continue
            suffix = "png"
            try:
                ct = getattr(related_part, "content_type", "") or ""
                if "/" in ct:
                    suffix = ct.split("/")[-1].split("+")[0]
            except Exception:
                pass
            out.append((blob, suffix))
        return out
