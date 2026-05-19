"""PDF parser running the layout-aware no-OCR pipeline.

Identical to the diagnostic ``/parse-pdf-no-ocr`` endpoint:
PP-DocLayoutV2 + table struct (UNet + SLANet+) + pdfplumber text — NO
OCR, so no torch / VietOCR deps. Used by ``/parse`` and ``/jobs``.

Builds a slim duck-type ``parser`` namespace exposing only the ONNX
components ``parse_pdf_no_ocr`` reads (``layout``, ``table_cls``,
``table_wired``, ``table_wireless``). ``VNDocParser`` itself is skipped
because its ``__init__`` instantiates ``VietOCRRec``.
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


def _build_no_ocr_parser() -> SimpleNamespace:
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _LOCK:
        if _SINGLETON is not None:
            return _SINGLETON

        from vn_parser.layout import LayoutDetector
        from vn_parser.table_cls import TableClassifier
        from vn_parser.table_slanet import PaddleTableModel
        from vn_parser.table_unet import UnetWiredTable

        models = Path(settings.mineru_models_dir)
        layout_providers = resolve_onnx_providers(settings.device_layout)
        table_cls_providers = resolve_onnx_providers(settings.device_table_cls)

        layout = LayoutDetector(
            models / "layout.onnx",
            conf=settings.mineru_layout_conf,
            providers=layout_providers,
        )

        table_cls = None
        if (models / "table_cls.onnx").exists():
            table_cls = TableClassifier(
                models / "table_cls.onnx", providers=table_cls_providers,
            )

        table_wired = None
        if (models / "table_unet.onnx").exists():
            table_wired = UnetWiredTable(
                str(models / "table_unet.onnx"), ocr_engine=None,
            )

        table_wireless = None
        if (models / "table_slanet.onnx").exists():
            table_wireless = PaddleTableModel(
                ocr_engine=None, model_path=str(models / "table_slanet.onnx"),
            )

        _SINGLETON = SimpleNamespace(
            layout=layout,
            table_cls=table_cls,
            table_wired=table_wired,
            table_wireless=table_wireless,
            ocr_det=None,
            ocr_rec=None,
            ocr_engine=None,
            orient=None,
        )
        return _SINGLETON


class PdfNoOcrParser(BaseParser):
    name = "pdf-no-ocr"
    extensions = ("pdf",)

    def parse(
        self,
        payload: bytes,
        filename: str,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> ParseResult:
        from parsers.mineru.pipeline_no_ocr import parse_pdf_no_ocr
        from parsers.pdf_mineru import _collect_images

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
                "dpi": settings.mineru_dpi,
                "tables_total": result["tables_total"],
                "orphans_total": result["orphans_total"],
                "total_ms": result["total_ms"],
                "page_timings": result["page_timings"],
            },
            images=images,
        )
