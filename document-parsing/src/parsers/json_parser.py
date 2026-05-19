"""JSON / JSONL → markdown.

Pretty-print under a fenced ``json`` block. JSONL gets one block per line.
"""
from __future__ import annotations

import json

from core.base import BaseParser, ParseResult
from core.compat import find_codec


class JsonParser(BaseParser):
    name = "json"
    extensions = ("json", "jsonl", "ndjson")

    def parse(self, payload: bytes, filename: str) -> ParseResult:
        encoding = find_codec(payload) if payload else "utf-8"
        text = payload.decode(encoding, errors="ignore")
        ext = filename.rsplit(".", 1)[-1].lower()

        try:
            obj = json.loads(text)
            pretty = json.dumps(obj, ensure_ascii=False, indent=2)
            md = f"```json\n{pretty}\n```"
            return ParseResult(
                markdown=md, parser=self.name, page_count=1,
                metadata={"extension": ext, "format": "json"},
            )
        except json.JSONDecodeError:
            pass

        # JSONL: one fenced block per valid line.
        blocks: list[str] = []
        valid = 0
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                blocks.append(f"```json\n{json.dumps(obj, ensure_ascii=False, indent=2)}\n```")
                valid += 1
            except json.JSONDecodeError:
                continue
        markdown = "\n\n".join(blocks) if blocks else f"```\n{text}\n```"
        return ParseResult(
            markdown=markdown, parser=self.name, page_count=valid,
            metadata={"extension": ext, "format": "jsonl" if blocks else "raw"},
        )
