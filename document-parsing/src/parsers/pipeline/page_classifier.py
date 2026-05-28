"""Per-page classification + document-mode gate for the CPU-batched pipeline.

Each page is labelled TEXT / IMAGE / SPARSE based on pdfplumber output:

  TEXT    — born-digital with real text layer; skip VietOCR
  IMAGE   — scanned / image-only; OCR every detected line
  SPARSE  — neither: blank page, watermark, mixed form. Default: skip OCR.

The document-level gate decides whether the batched TEXT-mode pipeline is
worth running at all: if too many pages would need OCR, the orchestrator
falls back to the original per-page hybrid path which is already
well-tuned for scan-heavy input.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import List, Sequence

from settings import settings

log = logging.getLogger(__name__)

TEXT = "TEXT"
IMAGE = "IMAGE"
SPARSE = "SPARSE"

DOC_MODE_TEXT = "TEXT"
DOC_MODE_SCAN = "SCAN"


def is_valid_text_char(c: str) -> bool:
    """Latin/Vietnamese-friendly: accept letters, digits, punctuation, ws."""
    if c in " \t\n\r":
        return True
    cat = unicodedata.category(c)
    return cat.startswith(("L", "N", "P"))


def compute_junk_char_ratio(words: Sequence[dict]) -> float:
    """Fraction of characters that look like CID/garbled output.

    PDFs with broken ToUnicode maps return text that pdfplumber happily
    extracts but which decodes to private-use codepoints. Above ~30% junk
    we treat the text layer as unreliable and force IMAGE handling.
    """
    text = "".join(w.get("text", "") for w in words)
    if not text:
        return 0.0
    valid = sum(1 for c in text if is_valid_text_char(c))
    return 1.0 - valid / len(text)


def _unique_word_ratio(words: Sequence[dict]) -> float:
    """Diagonal-watermark detector. A page reading "CONFIDENTIAL" tiled 30×
    looks word-rich but is informationally empty."""
    if not words:
        return 0.0
    tokens = [w.get("text", "").strip().lower() for w in words]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _bbox_area(item: dict, keys=("x0", "top", "x1", "bottom")) -> float:
    try:
        x0, y0, x1, y1 = (float(item[k]) for k in keys)
    except (KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _classify_page(page) -> str:
    """Inspect a single pdfplumber.Page and return TEXT/IMAGE/SPARSE."""
    try:
        words = page.extract_words(
            x_tolerance=2, y_tolerance=2, keep_blank_chars=False
        ) or []
    except Exception:
        log.exception("extract_words failed on page %r — defaulting SPARSE", page)
        words = []

    word_count = len(words)
    page_area = float(page.width) * float(page.height) if page.width and page.height else 0.0

    # Adjust for watermark-like repetition (do NOT touch real text).
    if word_count > 0:
        uniq_ratio = _unique_word_ratio(words)
        if uniq_ratio < 0.30:
            word_count = 0  # Treat as informationally empty.

    if word_count > 0:
        junk = compute_junk_char_ratio(words)
        if junk > settings.page_junk_char_threshold:
            return IMAGE

    text_area = sum(_bbox_area(w) for w in words)
    text_coverage = text_area / page_area if page_area > 0 else 0.0

    image_area = sum(_bbox_area(im) for im in getattr(page, "images", []) or [])
    image_coverage = image_area / page_area if page_area > 0 else 0.0

    if (
        word_count >= settings.page_text_min_words
        and text_coverage >= settings.page_text_coverage_min
    ):
        return TEXT

    if (
        word_count < settings.page_image_max_words
        and image_coverage >= settings.page_image_area_ratio
    ):
        return IMAGE

    # Optional: rescue form-like pages (preprinted text + handwritten regions).
    if (
        settings.ocr_sparse_with_image
        and image_coverage >= 0.20
    ):
        return IMAGE

    return SPARSE


def classify_pages(pdfplumber_doc) -> List[str]:
    """Return one label per page in order, length == len(pdf.pages)."""
    labels: List[str] = []
    for i, page in enumerate(pdfplumber_doc.pages):
        try:
            labels.append(_classify_page(page))
        except Exception:
            log.exception("classify_page raised on page %d — defaulting SPARSE", i)
            labels.append(SPARSE)
    return labels


def decide_document_mode(labels: Sequence[str]) -> str:
    """Choose between TEXT-mode (batched) and SCAN-mode (legacy hybrid)."""
    n = len(labels)
    if n == 0:
        return DOC_MODE_SCAN
    text_ratio = sum(1 for l in labels if l == TEXT) / n
    return DOC_MODE_TEXT if text_ratio >= settings.doc_text_mode_threshold else DOC_MODE_SCAN
