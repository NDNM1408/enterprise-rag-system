"""One-shot warmup for the batched CPU pipeline.

Each ONNX session compiles per-input-shape kernels the first time it
sees a new shape. In the production batched pipeline that first hit
lands on the first real chunk of pages, adding 60–120 s to the first
parse the worker handles. ``warm_batched_inference`` runs a single
dummy batch through Layout, DBNet, VietOCR, TableClassifier and the
table-struct models at the configured batch sizes so subsequent real
parses skip the cold start.

Wired into ``worker_process_init`` so it runs once per Celery fork
process — alongside the existing parser-singleton preload.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _make_dummy_page(h: int, w: int) -> np.ndarray:
    """White A4-ish page with a couple of horizontal strokes so DBNet
    actually finds something to detect (otherwise the postprocessor
    short-circuits and the kernel doesn't compile for the boxes path)."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in (h // 4, h // 2, 3 * h // 4):
        img[y - 4:y + 4, 100:w - 100] = 30
    return img


def _make_dummy_line(h: int = 36, w: int = 240) -> np.ndarray:
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    for x in range(20, w - 20, 30):
        img[8:h - 8, x:x + 12] = 30
    return img


def _make_dummy_table(h: int = 320, w: int = 480) -> np.ndarray:
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in (40, 120, 200, 280):
        cv2.line(img, (10, y), (w - 10, y), (0, 0, 0), 2)
    for x in (60, 180, 320, 440):
        cv2.line(img, (x, 20), (x, h - 20), (0, 0, 0), 2)
    return img


def warm_batched_inference(parser: Any) -> None:
    """Run one dummy batch through every batched stage of the pipeline.

    Quietly skips stages whose model isn't loaded (e.g. ``table_wired``
    when ``table_unet.onnx`` is absent).
    """
    from settings import settings

    from .batched_inference import (
        classify_tables_batched,
        run_dbnet_batched,
        run_layout_batched,
    )

    bs = max(1, settings.layout_batch_size)
    # A4 at the configured DPI (200 by default ≈ 1654 × 2339)
    dpi = settings.dpi
    h = int(round(11.69 * dpi))  # 11.69 in × dpi
    w = int(round(8.27 * dpi))   # 8.27 in × dpi
    dummy_page = _make_dummy_page(h, w)
    items = [(i, dummy_page) for i in range(bs)]

    t0 = time.perf_counter()
    try:
        run_layout_batched(parser, items)
    except Exception:
        log.exception("[warmup] layout batched failed")
    t_layout = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        run_dbnet_batched(parser, items)
    except Exception:
        log.exception("[warmup] dbnet batched failed")
    t_dbnet = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        line = _make_dummy_line()
        parser.ocr_rec.recognize_batch([line] * 4)
    except Exception:
        log.exception("[warmup] vietocr batched failed")
    t_ocr = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        if parser.table_cls is not None:
            tbl = _make_dummy_table()
            classify_tables_batched(parser, [tbl, tbl])
            if parser.table_wired is not None:
                rgb = cv2.cvtColor(tbl, cv2.COLOR_BGR2RGB)
                parser.table_wired.wired(rgb, [])
            if parser.table_wireless is not None:
                rgb = cv2.cvtColor(tbl, cv2.COLOR_BGR2RGB)
                parser.table_wireless.predict(rgb, [])
    except Exception:
        log.exception("[warmup] tables batched failed")
    t_tab = time.perf_counter() - t0

    log.info(
        "[warmup] B=%d page=%dx%d  layout=%.1fs  dbnet=%.1fs  vietocr=%.1fs  tables=%.1fs  total=%.1fs",
        bs, h, w, t_layout, t_dbnet, t_ocr, t_tab,
        t_layout + t_dbnet + t_ocr + t_tab,
    )
