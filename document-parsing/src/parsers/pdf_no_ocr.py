"""PDF parser running the layout-aware no-OCR pipeline.

Identical to the diagnostic ``/parse-pdf-no-ocr`` endpoint:
PP-DocLayoutV2 + table struct (UNet + SLANet+) + pdfplumber text — NO
OCR. Used by ``/parse`` and ``/jobs``.

Builds a slim duck-type ``parser`` namespace exposing only the ONNX
components ``parse_pdf_no_ocr`` reads (``layout``, ``table_cls``,
``table_wired``, ``table_wireless``). ``VNDocParser`` itself is skipped
because its ``__init__`` loads the recognition model (``rec.onnx``),
which the no-OCR path doesn't need.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Optional

from core.base import BaseParser, ParseResult, ProgressCallback
from core.devices import resolve_onnx_providers
from settings import settings


_SINGLETON: Optional[SimpleNamespace] = None
_LOCK = Lock()


def _build_no_ocr_parser():
    """Return the full VNDocParser (shared singleton from pdf_layout).

    Hybrid by design: ``parse_pdf_no_ocr`` uses pdfplumber for born-digital
    text (fast, no OCR) and falls back to the parser's ONNX OCR engine
    (DBNet det + PaddleOCR rec) for scanned/image regions that carry no
    embedded text. That fallback needs ``ocr_engine`` + ``_read_text`` +
    tables wired with the engine — exactly what the full VNDocParser
    provides, so we reuse it instead of a slim OCR-less namespace.
    """
    from parsers.pdf_layout import _get_parser
    return _get_parser()


class PdfNoOcrParser(BaseParser):
    name = "pdf-no-ocr"
    extensions = ("pdf",)

    def parse(
        self,
        payload: bytes,
        filename: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> ParseResult:
        from parsers.pipeline.pipeline_no_ocr import parse_pdf_no_ocr
        from parsers.pdf_layout import _collect_images

        parser = _build_no_ocr_parser()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.pdf"
            input_path.write_bytes(payload)
            image_dir = tmp_path / "out"
            image_dir.mkdir()

            result = parse_pdf_no_ocr(
                parser=parser,
                pdf_path=input_path,
                image_dir=image_dir,
                image_subdir="images",
                progress_cb=progress_cb,
                log_prefix="[parse-pdf-no-ocr]",
            )
            images = _collect_images(image_dir)

        return ParseResult(
            markdown=result["markdown"],
            parser=self.name,
            page_count=result["n_pages"],
            metadata={
                "extension": "pdf",
                "mode": result["mode"],
                "dpi": settings.dpi,
                "tables_total": result["tables_total"],
                "orphans_total": result["orphans_total"],
                "total_ms": result["total_ms"],
                "page_timings": result["page_timings"],
            },
            images=images,
        )
