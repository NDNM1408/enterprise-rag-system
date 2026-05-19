"""Excel/CSV → markdown.

Emits one markdown table per sheet using the vendored ragflow workbook
loader (handles raw CSV upload, calamine fallback, illegal-char stripping).
"""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from core.base import BaseParser, ParseResult
from vendored.ragflow.excel_parser import RAGFlowExcelParser

log = logging.getLogger(__name__)


class ExcelParser(BaseParser):
    name = "excel"
    extensions = ("xlsx", "xls", "csv")

    def parse(self, payload: bytes, filename: str) -> ParseResult:
        ext = Path(filename).suffix.lower().lstrip(".")
        wb = RAGFlowExcelParser._load_excel_to_workbook(BytesIO(payload))

        out: list[str] = []
        sheet_count = 0
        for sheetname in wb.sheetnames:
            ws = wb[sheetname]
            try:
                rows = RAGFlowExcelParser._get_rows_limited(ws)
            except Exception as e:
                log.warning("Skip sheet '%s': %s", sheetname, e)
                continue
            if not rows:
                continue
            sheet_count += 1
            md = self._sheet_to_markdown(sheetname, rows)
            if md:
                out.append(md)

        markdown = "\n\n".join(out).strip()
        return ParseResult(
            markdown=markdown,
            parser=self.name,
            page_count=sheet_count,
            metadata={"extension": ext, "sheet_count": sheet_count},
        )

    @staticmethod
    def _sheet_to_markdown(name: str, rows) -> str:
        cells = [[ "" if c.value is None else str(c.value).strip() for c in r] for r in rows]
        cells = [r for r in cells if any(c for c in r)]
        if not cells:
            return ""
        n_cols = max(len(r) for r in cells)
        cells = [r + [""] * (n_cols - len(r)) for r in cells]
        header = cells[0]
        body = cells[1:]
        lines = [f"## {name}", "",
                 "| " + " | ".join(_md_escape(c) for c in header) + " |",
                 "| " + " | ".join(["---"] * n_cols) + " |"]
        for r in body:
            lines.append("| " + " | ".join(_md_escape(c) for c in r) + " |")
        return "\n".join(lines)


def _md_escape(s: str) -> str:
    return s.replace("|", r"\|").replace("\n", " ")
