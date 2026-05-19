"""Markdown passthrough — decode + return as-is."""
from __future__ import annotations

from core.base import BaseParser, ParseResult
from core.compat import find_codec


class MarkdownParser(BaseParser):
    name = "markdown"
    extensions = ("md", "markdown", "mdown", "mkd")

    def parse(self, payload: bytes, filename: str) -> ParseResult:
        encoding = find_codec(payload) if payload else "utf-8"
        text = payload.decode(encoding, errors="ignore")
        return ParseResult(
            markdown=text,
            parser=self.name,
            page_count=0,
            metadata={"extension": filename.rsplit(".", 1)[-1].lower()},
        )
