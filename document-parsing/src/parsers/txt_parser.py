"""Plain text → markdown (wrapped, no transformation beyond decoding)."""
from __future__ import annotations

from core.base import BaseParser, ParseResult
from core.compat import find_codec


class TxtParser(BaseParser):
    name = "txt"
    extensions = ("txt", "log", "rst")

    def parse(self, payload: bytes, filename: str) -> ParseResult:
        encoding = find_codec(payload) if payload else "utf-8"
        text = payload.decode(encoding, errors="ignore")
        return ParseResult(
            markdown=text.strip(),
            parser=self.name,
            page_count=0,
            metadata={"extension": filename.rsplit(".", 1)[-1].lower()},
        )
