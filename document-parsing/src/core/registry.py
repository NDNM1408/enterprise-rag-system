"""Map file extensions to parser instances.

Parsers are imported lazily so a missing optional dep (e.g. MinerU torch stack)
only breaks the parser that needs it, not the whole service.
"""
from __future__ import annotations

import logging
from threading import Lock

from settings import settings

from .base import BaseParser

log = logging.getLogger(__name__)

_REGISTRY: dict[str, BaseParser] = {}
_LOCK = Lock()


def _register(parser: BaseParser) -> None:
    for ext in parser.extensions:
        _REGISTRY[ext.lower().lstrip(".")] = parser


def _build_registry() -> None:
    """Best-effort import of every parser. Failures are logged but non-fatal."""
    from parsers.docx_parser import DocxParser
    from parsers.epub_parser import EpubParser
    from parsers.excel_parser import ExcelParser
    from parsers.html_parser import HtmlParser
    from parsers.json_parser import JsonParser
    from parsers.markdown_passthrough import MarkdownParser
    from parsers.pdf_plain import PdfPlainParser
    from parsers.pptx_parser import PptxParser
    from parsers.txt_parser import TxtParser

    for cls in (
        DocxParser, ExcelParser, PptxParser,
        HtmlParser, EpubParser, MarkdownParser,
        TxtParser, JsonParser,
    ):
        try:
            _register(cls())
        except Exception:
            log.exception("Failed to register %s", cls.__name__)

    # PDF: prefer full MinerU (with OCR) when configured; otherwise use the
    # layout-aware no-OCR pipeline (PP-DocLayoutV2 + table struct + pdfplumber,
    # no torch/VietOCR). Fall back to pure pdfplumber if neither initialises.
    pdf_parser: BaseParser | None = None
    if not settings.pdf_force_plain:
        try:
            from parsers.pdf_mineru import MinerUPdfParser
            pdf_parser = MinerUPdfParser()
        except Exception as e:
            log.warning("MinerU parser unavailable: %s", e)

    if pdf_parser is None:
        try:
            from parsers.pdf_no_ocr import PdfNoOcrParser
            pdf_parser = PdfNoOcrParser()
        except Exception as e:
            log.warning("PdfNoOcrParser unavailable, falling back to pdfplumber: %s", e)

    if pdf_parser is not None:
        _register(pdf_parser)
    else:
        _register(PdfPlainParser())

    # Image fallback: MinerU's parser already covers image extensions; only
    # register the explicit fallback when MinerU isn't in play.
    if pdf_parser is None or pdf_parser.name != "mineru-vn-parser":
        try:
            from parsers.image_fallback import ImageFallbackParser
            _register(ImageFallbackParser())
        except Exception:
            log.exception("Failed to register ImageFallbackParser")


def registry() -> dict[str, BaseParser]:
    if not _REGISTRY:
        with _LOCK:
            if not _REGISTRY:
                _build_registry()
    return _REGISTRY


def for_extension(ext: str) -> BaseParser | None:
    return registry().get(ext.lower().lstrip("."))
