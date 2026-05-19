"""Shared parser interface.

Every parser implements ``parse(payload, filename) -> ParseResult``.
``markdown`` is the canonical output; everything else is metadata the caller
can ignore.

Long-running parsers (PDF) accept an optional ``progress_cb`` so the worker
can stream pages_done into the database. The callback signature is
``cb(pages_done: int, pages_total: int) -> None`` and is invoked at most
every ``progress_every`` pages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

ProgressCallback = Callable[[int, int], None]


@dataclass
class ExtractedImage:
    name: str
    bytes_b64: str
    mime: str = "image/png"


@dataclass
class ParseResult:
    markdown: str
    parser: str
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[ExtractedImage] = field(default_factory=list)


class BaseParser:
    """Each parser declares the extensions it owns; registry routes by ext."""

    name: str = "base"
    extensions: tuple[str, ...] = ()

    def parse(
        self,
        payload: bytes,
        filename: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> ParseResult:  # pragma: no cover
        raise NotImplementedError
