"""Batched wrappers around the vendored vn_parser ONNX models.

The vendored model classes (``LayoutDetector``, ``OCRDet``,
``TableClassifier``, ``UnetWiredTable``, ``PaddleTableModel``) all run a
single image per ``session.run`` call.  For CPU throughput the dominant
cost is Python overhead per call, not the matmul itself, so even
``batch_size=4`` already gives a sizeable speedup.

Each wrapper has the same interface:

    run_layout_batched(parser, items)   -> dict[idx -> list[block]]
    run_dbnet_batched(parser, items)    -> dict[idx -> list[quad]]
    classify_tables_batched(parser, crops) -> list[str]
    run_tables_batched(parser, tables_queue) -> dict[ref -> html]

``items`` is a list of ``(idx, np.ndarray HxWx3 RGB)`` tuples; pages of
identical shape are batched together via ``bucket_by_size``.

If ONNX export was made with a static batch=1 axis, the layout/DBNet
calls fall back to a serial loop transparently — same output, no
speedup but no crash.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from settings import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def bucket_by_size(items: Sequence[Tuple[int, np.ndarray]]) -> Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]]:
    """Group page images by (H, W); pages of equal shape share one ONNX run."""
    buckets: Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]] = {}
    for idx, img in items:
        key = (img.shape[0], img.shape[1])
        buckets.setdefault(key, []).append((idx, img))
    return buckets


def _chunks(seq: Sequence, size: int):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


# ---------------------------------------------------------------------------
#  Layout — fixed input_size (800, 800), straight batching always safe
# ---------------------------------------------------------------------------

def run_layout_batched(
    parser: Any,
    items: Sequence[Tuple[int, np.ndarray]],
    batch_size: int | None = None,
) -> Dict[int, List[dict]]:
    """Run the layout detector over many pages with batched ONNX calls.

    Layout preprocess resizes every image to a fixed (W, H) regardless
    of source shape, so no size-bucketing is required here — we can pass
    a heterogeneous list straight to ``LayoutDetector.batch_predict``.
    """
    if not items:
        return {}
    bs = batch_size or settings.layout_batch_size
    indices = [i for i, _ in items]
    images = [img for _, img in items]
    try:
        results = parser.layout.batch_predict(images, batch_size=bs)
    except Exception as e:
        log.warning("layout.batch_predict(batch=%d) failed (%s); falling back to serial", bs, e)
        results = [parser.layout.predict(img) for img in images]
    return {idx: blocks for idx, blocks in zip(indices, results)}


# ---------------------------------------------------------------------------
#  DBNet — input shape varies with source shape; bucket then batch
# ---------------------------------------------------------------------------

def _dbnet_preprocess_batch(ocr_det, bgr_images: Sequence[np.ndarray]):
    """Resize + normalize a list of identically-shaped BGR images.

    Returns (batched_tensor (B,3,H',W'), resized_shape (H',W'),
    src_shapes [(h,w)...]).
    """
    src_shapes: List[Tuple[int, int]] = []
    resized_list: List[np.ndarray] = []
    target_shape: Tuple[int, int] | None = None
    for img in bgr_images:
        resized, (src_h, src_w) = ocr_det._resize(img)
        if target_shape is None:
            target_shape = resized.shape[:2]
        elif resized.shape[:2] != target_shape:
            # Should not happen if caller bucketed by size; fall back to single.
            raise ValueError(
                f"DBNet batch shape mismatch: {resized.shape[:2]} vs {target_shape}"
            )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - ocr_det.MEAN) / ocr_det.STD
        resized_list.append(rgb.transpose(2, 0, 1))
        src_shapes.append((src_h, src_w))
    batch = np.stack(resized_list, axis=0).astype(np.float32)
    return batch, target_shape, src_shapes


def run_dbnet_batched(
    parser: Any,
    items: Sequence[Tuple[int, np.ndarray]],
    batch_size: int | None = None,
) -> Dict[int, List[np.ndarray]]:
    """Run DBNet (``ocr_det.onnx``) over many pages.

    ``items`` carries RGB numpy arrays — we convert to BGR per page since
    that is the format the vendored ``OCRDet`` was written for.
    """
    if not items:
        return {}
    bs = batch_size or settings.dbnet_batch_size
    ocr_det = parser.ocr_det
    out: Dict[int, List[np.ndarray]] = {}

    # Bucket so each ``session.run`` sees one shape.
    buckets = bucket_by_size(items)
    for shape_key, bucket in buckets.items():
        for chunk in _chunks(bucket, bs):
            indices = [idx for idx, _ in chunk]
            bgr_list = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for _, img in chunk]
            try:
                batch, _, src_shapes = _dbnet_preprocess_batch(ocr_det, bgr_list)
                prob_maps = ocr_det.session.run(None, {ocr_det.input_name: batch})[0]
            except Exception as e:
                log.warning(
                    "dbnet batched run failed at shape=%s bs=%d (%s); serial fallback",
                    shape_key, len(chunk), e,
                )
                for idx, img_bgr in zip(indices, bgr_list):
                    try:
                        boxes, _ = ocr_det.detect(img_bgr)
                    except Exception:
                        log.exception("dbnet single-page failed on idx=%d", idx)
                        boxes = []
                    out[idx] = boxes or []
                continue

            for j, idx in enumerate(indices):
                src_h, src_w = src_shapes[j]
                try:
                    boxes, _ = ocr_det.post(prob_maps[j:j + 1], src_h, src_w)
                except Exception:
                    log.exception("dbnet postprocess failed on idx=%d", idx)
                    boxes = []
                out[idx] = boxes or []

    return out


# ---------------------------------------------------------------------------
#  Table classifier — fixed (224, 224); always batch-safe
# ---------------------------------------------------------------------------

def classify_tables_batched(
    parser: Any,
    crops_bgr: Sequence[np.ndarray],
    batch_size: int = 32,
) -> List[str]:
    """Run ``table_cls`` over many table crops; returns list[label]."""
    from vn_parser.table_cls import TABLE_TYPES

    if not crops_bgr:
        return []
    if parser.table_cls is None:
        return ["wired"] * len(crops_bgr)

    tc = parser.table_cls
    labels: List[str] = []
    for chunk in _chunks(crops_bgr, batch_size):
        try:
            tensors = [tc._preprocess(c) for c in chunk]
            batch = np.concatenate(tensors, axis=0).astype(np.float32)
            out = tc.session.run(None, {tc.input_name: batch})[0]
            idx_arr = np.argmax(out, axis=1)
            labels.extend(TABLE_TYPES[int(i)] for i in idx_arr)
        except Exception as e:
            log.warning("table_cls batched run failed (%s); serial fallback", e)
            for c in chunk:
                try:
                    labels.append(tc.classify(c))
                except Exception:
                    log.exception("table_cls single-crop failed")
                    labels.append("wired")
    return labels


# ---------------------------------------------------------------------------
#  Table struct — model wrappers expose only per-image .predict(); we loop
# ---------------------------------------------------------------------------

def _struct_wired(parser, crop_bgr) -> str:
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        wt = parser.table_wired.wired(rgb, [])
        return wt.pred_html or ""
    except Exception:
        log.exception("table_wired failed")
        return ""


def _struct_wireless(parser, crop_bgr) -> str:
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        html_code, _, _, _ = parser.table_wireless.predict(rgb, [])
        return html_code or ""
    except Exception:
        log.exception("table_wireless failed")
        return ""


def run_tables_pipeline(
    parser: Any,
    tables_queue: Sequence[Tuple[Any, np.ndarray]],
) -> Dict[Any, Tuple[str, str]]:
    """Classify + struct every table crop. Returns {ref: (kind, html)}.

    ``tables_queue`` is ``[(ref, crop_bgr), ...]`` produced by the
    orchestrator (ref typically ``(page_idx, block_index)``).

    Tables of either kind whose model isn't loaded fall back to the
    other; tables with neither available are emitted as ``("fallback", "")``.
    """
    if not tables_queue:
        return {}

    refs = [r for r, _ in tables_queue]
    crops = [c for _, c in tables_queue]

    labels = classify_tables_batched(parser, crops)

    results: Dict[Any, Tuple[str, str]] = {}
    wired_available = parser.table_wired is not None
    wireless_available = parser.table_wireless is not None

    for ref, crop, label in zip(refs, crops, labels):
        kind = label
        if kind == "wired" and not wired_available:
            kind = "wireless"
        elif kind == "wireless" and not wireless_available:
            kind = "wired"
        if kind == "wired" and wired_available:
            html = _struct_wired(parser, crop)
        elif kind == "wireless" and wireless_available:
            html = _struct_wireless(parser, crop)
        else:
            kind, html = "fallback", ""
        results[ref] = (kind, html)
    return results
