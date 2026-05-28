"""Fallback PDF parser (pdfplumber). Text-only, no OCR, no layout intelligence.

Used when the parsing pipeline is unavailable or the user passes ``mode=fast``.
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional

import pdfplumber

from core.base import BaseParser, ParseResult, ProgressCallback


class PdfPlainParser(BaseParser):
    name = "pdfplumber-plain"
    extensions = ("pdf",)

    def parse(
        self,
        payload: bytes,
        filename: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> ParseResult:
        sections: list[str] = []
        page_count = 0
        with pdfplumber.open(BytesIO(payload)) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_count += 1
                txt = (page.extract_text() or "").strip()
                if txt:
                    sections.append(f"<!-- page {i + 1} -->\n\n{txt}")
                if progress_cb is not None:
                    progress_cb(i + 1, total)
        markdown = "\n\n".join(sections).strip()
        return ParseResult(
            markdown=markdown,
            parser=self.name,
            page_count=page_count,
            metadata={"extension": "pdf", "mode": "plain"},
        )
