"""TEXT-mode orchestrator for the CPU-batched vn_parser pipeline.

Driven by ``LayoutPdfParser.parse()`` when ``decide_document_mode`` picks
TEXT.  Stages (see SPEC §3):

  1. Iterate over pages in chunks of ``settings.page_chunk_size``:
     • render the chunk to RGB numpy arrays
     • run Layout (batched ONNX) → blocks/page
     • run DBNet  (batched ONNX) → quads/page
     • per page: build Block records, save image crops, distribute
       quads to text blocks, enqueue IMAGE-page crops into ``ocr_queue``
       and every table crop into ``tables_queue``.
  2. After the loop, run **VietOCR mega-batch** and **table struct**
     pipeline in parallel via ``ThreadPoolExecutor``.
  3. Stitch recognised text and HTML back into Block records, then
     serialise via ``VNDocParser.to_markdown``.

Image-like and figure crops are saved to disk inside the same temp
directory used by the legacy hybrid path so the caller can collect them
via ``_collect_images``.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium

from settings import settings

from .batched_inference import (
    run_dbnet_batched,
    run_layout_batched,
    run_tables_pipeline,
)
from .ocr_megabatch import run_ocr_megabatch
from .page_classifier import IMAGE, SPARSE, TEXT

log = logging.getLogger(__name__)


def _render_chunk(
    pdfium_doc, page_indices: Sequence[int], dpi: int
) -> List[Tuple[int, np.ndarray]]:
    """Render pages to RGB numpy arrays at the configured DPI."""
    scale = dpi / 72.0
    out: List[Tuple[int, np.ndarray]] = []
    for idx in page_indices:
        page = pdfium_doc[idx]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil().convert("RGB")
        bitmap.close()
        page.close()
        out.append((idx, np.asarray(pil)))
    return out


def _save_block_crop(
    rgb_image: np.ndarray, bbox: Tuple[int, int, int, int],
    page_index: int, label: str, index: int,
    sub_dir: Path, image_subdir: str,
) -> str:
    """Crop + save a block region; returns the markdown-relative image path."""
    h_img, w_img = rgb_image.shape[:2]
    pad = 4
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w_img, x1 + pad); y1 = min(h_img, y1 + pad)
    crop = rgb_image[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    fname = f"page_{page_index + 1:03d}_{label}_{index:02d}.jpg"
    cv2.imwrite(str(sub_dir / fname), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return f"{image_subdir}/{fname}"


def _quad_center(quad: np.ndarray) -> Tuple[float, float]:
    xs = quad[:, 0].astype(np.float32)
    ys = quad[:, 1].astype(np.float32)
    return float(xs.mean()), float(ys.mean())


def _quad_bbox(quad: np.ndarray) -> Tuple[float, float, float, float]:
    xs = quad[:, 0].astype(np.float32)
    ys = quad[:, 1].astype(np.float32)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _quad_y(quad: np.ndarray) -> float:
    return float(quad[:, 1].astype(np.float32).mean())


def _quad_x(quad: np.ndarray) -> float:
    return float(quad[:, 0].astype(np.float32).min())


def _point_in_bbox(cx: float, cy: float, bbox: Sequence[float]) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _pdfplumber_text_in_quad(words: Sequence[dict], quad: np.ndarray, scale: float) -> str:
    """Return concatenated pdfplumber word text whose centers lie inside the quad bbox.

    ``words`` use PDF-units (origin top-left, points); ``quad`` is in
    rendered image pixels.  Divide by ``scale`` (= dpi/72) to compare.
    """
    qx0, qy0, qx1, qy1 = _quad_bbox(quad)
    qx0 /= scale; qy0 /= scale; qx1 /= scale; qy1 /= scale
    pad_y = 1.0  # PDF points; absorbs minor mismatch between quads and char baselines
    pad_x = 0.5
    matches: List[Tuple[float, str]] = []
    for w in words:
        try:
            cx = (float(w["x0"]) + float(w["x1"])) / 2
            cy = (float(w["top"]) + float(w["bottom"])) / 2
        except (KeyError, TypeError, ValueError):
            continue
        if (qx0 - pad_x) <= cx <= (qx1 + pad_x) and (qy0 - pad_y) <= cy <= (qy1 + pad_y):
            matches.append((float(w.get("x0", 0)), str(w.get("text", ""))))
    if not matches:
        return ""
    matches.sort(key=lambda t: t[0])
    return " ".join(t for _, t in matches if t).strip()


def _group_lines_by_y(items: Sequence[Tuple[float, float, str]]) -> str:
    """Re-flow individual line strings into reading-order text.

    Each item is ``(y_mean, x_min, text)``; lines whose y-centers are
    within ``LINE_THRESH`` merge horizontally (sometimes DBNet splits a
    single visual line into a few quads).
    """
    LINE_THRESH = 12.0
    rows: List[List[str]] = []
    cur_y: Optional[float] = None
    sorted_items = sorted(items, key=lambda t: (round(t[0] / 8.0) * 8.0, t[1]))
    for y_mean, _x_min, text in sorted_items:
        text = (text or "").strip()
        if not text:
            continue
        if cur_y is None or abs(y_mean - cur_y) > LINE_THRESH:
            rows.append([text])
            cur_y = y_mean
        else:
            rows[-1].append(text)
    return "\n".join(" ".join(parts) for parts in rows).strip()


def parse_text_mode_batched(
    parser: Any,
    pdf_path: Path,
    image_dir: Path,
    image_subdir: str,
    labels: Sequence[str],
    plumber_pages: Sequence[Any],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, int, str]:
    """Run the batched TEXT-mode pipeline and return ``(markdown, n_pages, mode)``.

    ``plumber_pages`` is the already-open list of ``pdfplumber.Page``;
    keeping it open across this call avoids re-opening the file.
    """
    from vn_parser import VNDocParser
    from vn_parser.ocr_det import OCRDet
    from vn_parser.pipeline import (
        Block,
        FORMULA_LABELS,
        IMAGE_LIKE_LABELS,
        PageResult,
        TABLE_LABELS,
        TEXT_LIKE_LABELS,
    )

    dpi = settings.dpi
    scale = dpi / 72.0
    sub_dir = image_dir / image_subdir
    sub_dir.mkdir(parents=True, exist_ok=True)

    pdfium_doc = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdfium_doc)
    if n_pages == 0:
        pdfium_doc.close()
        return "", 0, "text-mode:empty"

    pages_state: Dict[int, PageResult] = {}
    quad_meta: Dict[int, Dict[int, List[Tuple[Any, np.ndarray]]]] = {}
    # quad_meta[page_idx][block_idx] -> list of (ref, quad_array)
    ocr_queue: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
    tables_queue: List[Tuple[Tuple[int, int], np.ndarray]] = []

    n_image_pages = sum(1 for l in labels if l == IMAGE)
    n_text_pages = sum(1 for l in labels if l == TEXT)
    n_sparse_pages = sum(1 for l in labels if l == SPARSE)

    if progress_cb is not None:
        try:
            progress_cb(0, n_pages)
        except Exception:
            log.exception("progress_cb (initial) raised")

    try:
        for start in range(0, n_pages, settings.page_chunk_size):
            chunk_indices = list(range(start, min(start + settings.page_chunk_size, n_pages)))
            chunk_t0 = time.perf_counter()

            rendered = _render_chunk(pdfium_doc, chunk_indices, dpi)
            # Layout and DBNet are independent (both consume ``rendered``,
            # neither feeds the other), so run them concurrently. ONNX
            # session.run releases the GIL → real CPU parallelism. The
            # same pattern is used in api/routes.py::_parse_image_inline.
            with ThreadPoolExecutor(max_workers=2) as chunk_pool:
                fut_layout = chunk_pool.submit(run_layout_batched, parser, rendered)
                fut_dbnet = chunk_pool.submit(run_dbnet_batched, parser, rendered)
                layout_per_page = fut_layout.result()
                dbnet_per_page = fut_dbnet.result()

            for idx, rgb in rendered:
                label = labels[idx] if idx < len(labels) else SPARSE
                plumber_page = plumber_pages[idx]
                try:
                    plumber_words = plumber_page.extract_words(
                        x_tolerance=2, y_tolerance=2, keep_blank_chars=False
                    ) or []
                except Exception:
                    log.exception("extract_words failed on page %d", idx)
                    plumber_words = []

                layout_blocks_dicts = layout_per_page.get(idx, []) or []
                quads = dbnet_per_page.get(idx, []) or []

                h, w = rgb.shape[:2]
                page = PageResult(page_index=idx, width=w, height=h, angle=0)
                pages_state[idx] = page
                quad_meta[idx] = {}

                # ── Build Block records and route by label ────────────────
                bgr_lazy: Optional[np.ndarray] = None  # lazy BGR copy for table crops
                page_text_block_idx: List[int] = []
                for lb in layout_blocks_dicts:
                    cls_id = lb.get("cls_id", -1)
                    block = Block(
                        cls_id=cls_id,
                        label=lb["label"],
                        score=lb.get("score", 0.0),
                        bbox=tuple(int(v) for v in lb["bbox"]),
                        index=lb.get("index", len(page.blocks) + 1),
                    )
                    block.extra["page_label"] = label

                    if block.label in IMAGE_LIKE_LABELS:
                        path = _save_block_crop(
                            rgb, block.bbox, idx, block.label,
                            block.index, sub_dir, image_subdir,
                        )
                        if path:
                            block.image_path = path
                        page.blocks.append(block)
                        continue

                    if block.label in TABLE_LABELS:
                        path = _save_block_crop(
                            rgb, block.bbox, idx, block.label,
                            block.index, sub_dir, image_subdir,
                        )
                        if path:
                            block.image_path = path
                        if settings.parse_table_struct and parser.table_cls is not None:
                            x0, y0, x1, y1 = block.bbox
                            if bgr_lazy is None:
                                bgr_lazy = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            crop = bgr_lazy[y0:y1, x0:x1]
                            if crop.size > 0:
                                tables_queue.append(
                                    ((idx, len(page.blocks)), crop)
                                )
                        page.blocks.append(block)
                        continue

                    if block.label in TEXT_LIKE_LABELS or block.label in FORMULA_LABELS:
                        page_text_block_idx.append(len(page.blocks))
                        quad_meta[idx][len(page.blocks)] = []
                    page.blocks.append(block)

                # ── Distribute quads to text blocks via center-containment ─
                # Skip quads whose center is inside a table block — table struct
                # owns OCR for those regions.
                table_bboxes = [
                    page.blocks[bi].bbox
                    for bi in range(len(page.blocks))
                    if page.blocks[bi].label in TABLE_LABELS
                ]
                text_block_bboxes = [
                    (bi, page.blocks[bi].bbox) for bi in page_text_block_idx
                ]

                if quads and text_block_bboxes:
                    for q in quads:
                        cx, cy = _quad_center(q)
                        if any(_point_in_bbox(cx, cy, tb) for tb in table_bboxes):
                            continue
                        for bi, bbox in text_block_bboxes:
                            if _point_in_bbox(cx, cy, bbox):
                                quad_meta[idx][bi].append(("__pending__", q))
                                break

                # ── Generate per-quad text or OCR queue entries ───────────
                for bi in page_text_block_idx:
                    quad_list = quad_meta[idx].get(bi, [])
                    new_list: List[Tuple[Any, np.ndarray]] = []
                    for _, q in quad_list:
                        if label == IMAGE:
                            if bgr_lazy is None:
                                bgr_lazy = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            crop = OCRDet.crop_quad(bgr_lazy, q)
                            if crop is None or crop.size == 0:
                                new_list.append((None, q))
                                continue
                            ref = (idx, bi, len(ocr_queue))
                            ocr_queue.append((crop, ref))
                            new_list.append((ref, q))
                        else:
                            text = _pdfplumber_text_in_quad(plumber_words, q, scale)
                            new_list.append((text, q))
                    quad_meta[idx][bi] = new_list

                # ── TEXT/SPARSE fallback: if a text block has no quad-text at
                # all but pdfplumber found ≥ pdf_block_min_words inside its
                # bbox, fall back to block-level pdfplumber so we don't drop
                # paragraphs that DBNet missed.
                if label in (TEXT, SPARSE):
                    for bi in page_text_block_idx:
                        if any(
                            isinstance(t, str) and t.strip()
                            for t, _ in quad_meta[idx][bi]
                        ):
                            continue
                        x0, y0, x1, y1 = page.blocks[bi].bbox
                        x0p = x0 / scale; y0p = y0 / scale
                        x1p = x1 / scale; y1p = y1 / scale
                        words_in: List[Tuple[float, float, str]] = []
                        for w in plumber_words:
                            try:
                                cx_ = (float(w["x0"]) + float(w["x1"])) / 2
                                cy_ = (float(w["top"]) + float(w["bottom"])) / 2
                            except (KeyError, TypeError, ValueError):
                                continue
                            if x0p <= cx_ <= x1p and y0p <= cy_ <= y1p:
                                words_in.append((cy_, float(w.get("x0", 0)),
                                                 str(w.get("text", ""))))
                        if len(words_in) >= settings.pdf_block_min_words:
                            page.blocks[bi].text = _group_lines_by_y(words_in)
                            page.blocks[bi].extra["text_source"] = "pdfplumber_block"

                # Tag block sources for observability.
                for bi in page_text_block_idx:
                    page.blocks[bi].extra.setdefault(
                        "text_source",
                        "vietocr_quad" if label == IMAGE else "pdfplumber_quad",
                    )

            # Periodic progress: pages so far.
            done = min(start + len(chunk_indices), n_pages)
            print(
                f"[text-mode chunk] pages {chunk_indices[0] + 1}..{chunk_indices[-1] + 1}/{n_pages} "
                f"render+layout+dbnet+map = {(time.perf_counter() - chunk_t0):.2f}s",
                flush=True,
            )
            if progress_cb is not None:
                try:
                    progress_cb(done, n_pages)
                except Exception:
                    log.exception("progress_cb raised")

        # ── Parallel: VietOCR mega-batch || table struct ──────────────
        ocr_t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_ocr = pool.submit(run_ocr_megabatch, parser, ocr_queue)
            fut_tab = pool.submit(run_tables_pipeline, parser, tables_queue)
            ocr_results = fut_ocr.result()
            table_results = fut_tab.result()
        print(
            f"[text-mode] OCR(crops={len(ocr_queue)}) || tables(crops={len(tables_queue)}) "
            f"= {(time.perf_counter() - ocr_t0):.2f}s",
            flush=True,
        )

        # ── Stitch results back into Block records ────────────────────
        for idx, page in pages_state.items():
            for bi, quad_list in quad_meta[idx].items():
                items: List[Tuple[float, float, str]] = []
                for ref_or_text, q in quad_list:
                    text = ""
                    if isinstance(ref_or_text, str):
                        text = ref_or_text
                    elif isinstance(ref_or_text, tuple):
                        text = ocr_results.get(ref_or_text, "")
                    if not text:
                        continue
                    items.append((_quad_y(q), _quad_x(q), text))
                if items:
                    # Preserve block-level fallback if quad path returned nothing.
                    existing = page.blocks[bi].text
                    quad_text = _group_lines_by_y(items)
                    if quad_text:
                        page.blocks[bi].text = quad_text
                    elif existing:
                        pass

            for ti in range(len(page.blocks)):
                if page.blocks[ti].label in TABLE_LABELS:
                    ref = (idx, ti)
                    kind, html = table_results.get(ref, ("fallback", ""))
                    if html:
                        page.blocks[ti].text = html
                    if kind:
                        page.blocks[ti].extra["table_kind"] = kind

        # ── Markdown ──────────────────────────────────────────────────
        ordered = [pages_state[i] for i in range(n_pages) if i in pages_state]
        markdown = VNDocParser.to_markdown(ordered)
        mode = (
            f"text-mode:text={n_text_pages},image={n_image_pages},sparse={n_sparse_pages},"
            f"ocr_crops={len(ocr_queue)},tables={len(tables_queue)}"
        )

        if progress_cb is not None:
            try:
                progress_cb(n_pages, n_pages)
            except Exception:
                log.exception("progress_cb (final) raised")
        return markdown, n_pages, mode

    finally:
        pdfium_doc.close()
