"""Mega-batched VietOCR over all OCR line crops in a document.

The vendored ``VietOCRRec.recognize_batch`` already buckets by image
width internally before issuing a single torch forward per bucket — what
this module adds is whole-document gathering.  Calling ``recognize_batch``
once with every line crop from every page lets the underlying predictor
maximise its width buckets, and removes the per-page Python overhead of
many small calls.

Inputs come in as ``[(crop_np, ref), ...]`` with ``ref`` chosen by the
orchestrator (``(page_idx, block_id, quad_id)``); the result is a dict
mapping ``ref`` to recognised text so the assembler can stitch back.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Hashable, List, Sequence, Tuple

import numpy as np

from settings import settings

log = logging.getLogger(__name__)


def _crop_width(crop: np.ndarray) -> int:
    if crop is None or crop.size == 0:
        return 0
    return int(crop.shape[1])


def _bucket_step(width: int, step: int) -> int:
    step = max(step, 1)
    return ((width + step - 1) // step) * step


def _chunks(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def run_ocr_megabatch(
    parser: Any,
    ocr_queue: Sequence[Tuple[np.ndarray, Hashable]],
    batch_size: int | None = None,
    bucket_width_step: int | None = None,
) -> Dict[Hashable, str]:
    """Recognize every crop in ``ocr_queue`` and return ``{ref: text}``.

    Sort + group by width bucket so each ``recognize_batch`` call sees
    similar-width crops (less padding waste); within each bucket split
    into ``batch_size`` chunks.
    """
    if not ocr_queue:
        return {}

    bs = batch_size or settings.ocr_batch_size
    step = bucket_width_step or settings.ocr_bucket_width_step

    sized: List[Tuple[int, np.ndarray, Hashable]] = []
    empty_refs: List[Hashable] = []
    for crop, ref in ocr_queue:
        w = _crop_width(crop)
        if w <= 0 or crop is None or getattr(crop, "size", 0) == 0:
            empty_refs.append(ref)
            continue
        sized.append((_bucket_step(w, step), crop, ref))

    sized.sort(key=lambda t: t[0])

    buckets: Dict[int, List[Tuple[np.ndarray, Hashable]]] = {}
    for bucket_w, crop, ref in sized:
        buckets.setdefault(bucket_w, []).append((crop, ref))

    results: Dict[Hashable, str] = {ref: "" for ref in empty_refs}
    rec = parser.ocr_rec
    for bucket_w in sorted(buckets):
        bucket = buckets[bucket_w]
        for chunk in _chunks(bucket, bs):
            crops = [c for c, _ in chunk]
            refs = [r for _, r in chunk]
            try:
                texts = rec.recognize_batch(crops)
            except Exception as e:
                log.warning(
                    "recognize_batch failed (bucket_w=%d size=%d): %s; per-item fallback",
                    bucket_w, len(chunk), e,
                )
                texts = []
                for c in crops:
                    try:
                        texts.append(rec.recognize(c))
                    except Exception:
                        log.exception("recognize() failed; emitting empty text")
                        texts.append("")
            for ref, text in zip(refs, texts):
                results[ref] = text or ""
    return results
