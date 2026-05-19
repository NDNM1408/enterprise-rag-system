"""Last-resort image parser when MinerU is unavailable.

Returns a markdown image reference (base64 data URI). No OCR.
"""
from __future__ import annotations

import base64
from pathlib import Path

from core.base import BaseParser, ParseResult

_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}


class ImageFallbackParser(BaseParser):
    name = "image-fallback"
    extensions = ("png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff")

    def parse(self, payload: bytes, filename: str) -> ParseResult:
        ext = Path(filename).suffix.lower().lstrip(".")
        mime = _MIME.get(ext, "application/octet-stream")
        b64 = base64.b64encode(payload).decode("ascii")
        md = f"![{filename}](data:{mime};base64,{b64})"
        return ParseResult(
            markdown=md,
            parser=self.name,
            page_count=1,
            metadata={"extension": ext, "note": "no OCR; MinerU unavailable"},
        )
