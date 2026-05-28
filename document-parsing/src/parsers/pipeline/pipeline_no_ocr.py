"""No-OCR PDF pipeline: layout + table_struct + pdfplumber.

Per page:
  • render to image (pypdfium2)
  • pdfplumber → words (rotated glyphs filtered)
  • PP-DocLayoutV2 → block bboxes
  • per text block: gather pdfplumber words by bbox → format as lines
  • per table block: crop + run ``extract_table_routed`` with precomputed_ocr
    built from the pdfplumber words inside the table (fast-path, no OCR)
  • image/figure block: crop saved to ``image_dir``
  • orphan words (not in any layout block): emitted as plain text

This module is the single source of truth for the diagnostic
``/parse-pdf-no-ocr`` endpoint *and* the Celery worker's PDF path. Both
get the same per-page progress logging via ``print(flush=True)``.

Returns a dict so callers (route handler / ``LayoutPdfParser._parse_pdf``)
can pull whichever fields they need.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import pdfplumber
import pypdfium2 as pdfium

from settings import settings
from vn_parser.ocr_det import OCRDet

log = logging.getLogger(__name__)


# ---- words → markdown helpers ------------------------------------------------

TITLE_LABELS = {
    "doc_title", "paragraph_title", "figure_title",
    "header", "table_caption", "image_caption",
}


def drop_rotated_words(words: list) -> list:
    """Filter out pdfplumber words whose glyphs aren't upright.

    Rotated text (vertical watermarks, page-margin labels) is read
    character-by-character by pdfplumber and explodes a single visual
    phrase into many one-glyph "words" stacked vertically — corrupts
    line clustering. Each word dict carries ``upright`` (bool); ``False``
    ⇒ skip.
    """
    return [w for w in words if w.get("upright", True)]


def format_pdfplumber_words(words: list) -> str:
    """Group pdfplumber words into lines by vertical overlap, join in reading order."""
    if not words:
        return ""

    def _top(w):    return float(w.get("top", 0))
    def _bot(w):    return float(w.get("bottom", 0))
    def _x0(w):     return float(w.get("x0", 0))
    def _text(w):   return str(w.get("text", ""))
    def _h(w):      return max(_bot(w) - _top(w), 0.0)

    heights = sorted(_h(w) for w in words if _h(w) > 0)
    median_h = heights[len(heights) // 2] if heights else 10.0
    min_overlap = max(median_h * 0.3, 1.5)

    ordered = sorted(words, key=lambda w: (_top(w), _x0(w)))

    lines: list[list] = []
    cur_top: float | None = None
    cur_bot: float | None = None
    for w in ordered:
        t, b = _top(w), _bot(w)
        if cur_top is None:
            lines.append([w])
            cur_top, cur_bot = t, b
            continue
        overlap = min(cur_bot, b) - max(cur_top, t)
        if overlap >= min_overlap:
            lines[-1].append(w)
            cur_top = min(cur_top, t)
            cur_bot = max(cur_bot, b)
        else:
            lines.append([w])
            cur_top, cur_bot = t, b

    out: list[str] = []
    for line in lines:
        line.sort(key=_x0)
        out.append(" ".join(_text(w) for w in line if _text(w)))
    return "\n".join(ln for ln in out if ln.strip()).strip()


def _build_precomputed_ocr_for_table(
    words: list,
    table_bbox_img: tuple,
    scale: float,
) -> list:
    """Convert pdfplumber words inside ``table_bbox_img`` (in image pixels) to
    ``extract_table_routed``'s ``precomputed_ocr`` shape:
    ``[[quad_4x2_local, (text, score)], ...]``.

    Each word's PDF-points bbox → image pixels via ``scale`` → translate to
    table-crop-local coords by subtracting the table top-left.
    """
    x0, y0, x1, y1 = table_bbox_img
    out = []
    for w in words:
        try:
            wx0 = float(w["x0"]) * scale
            wy0 = float(w["top"]) * scale
            wx1 = float(w["x1"]) * scale
            wy1 = float(w["bottom"]) * scale
        except (KeyError, TypeError, ValueError):
            continue
        cx = (wx0 + wx1) / 2
        cy = (wy0 + wy1) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        lx0, ly0 = wx0 - x0, wy0 - y0
        lx1, ly1 = wx1 - x0, wy1 - y0
        quad = np.array(
            [[lx0, ly0], [lx1, ly0], [lx1, ly1], [lx0, ly1]],
            dtype=np.float32,
        )
        text = str(w.get("text", ""))
        out.append([quad, (text, 1.0)])
    return out


# ---- main pipeline -----------------------------------------------------------

def parse_pdf_no_ocr(
    parser: Any,
    pdf_path: Path,
    image_dir: Path,
    *,
    image_subdir: str = "images",
    progress_cb: Optional[Callable[[int, int], None]] = None,
    log_prefix: str = "[parse-pdf-no-ocr]",
) -> Dict[str, Any]:
    """Run the no-OCR pipeline on a PDF; emit per-page progress to stdout.

    Returns a dict with ``markdown``, ``n_pages``, ``mode``, ``page_timings``,
    ``tables_total``, ``orphans_total``.

    ``image_dir`` receives cropped figure/table images under
    ``image_dir/image_subdir/``; the caller (worker / route handler) is
    responsible for cleanup or upload.
    """
    from parsers.pdf_layout import extract_table_both_parallel
    from vn_parser import VNDocParser
    from vn_parser.pipeline import (
        Block,
        FORMULA_LABELS,
        IMAGE_LIKE_LABELS,
        PageResult,
        TABLE_LABELS,
        TEXT_LIKE_LABELS,
    )

    overall_t0 = time.perf_counter()
    dpi = settings.dpi
    scale = dpi / 72.0
    sub_dir = image_dir / image_subdir
    sub_dir.mkdir(parents=True, exist_ok=True)

    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf_doc)
    print(f"{log_prefix} opened {n_pages} pages dpi={dpi}", flush=True)
    if n_pages == 0:
        pdf_doc.close()
        return {
            "markdown": "", "n_pages": 0,
            "mode": "no-ocr:empty",
            "page_timings": [], "tables_total": 0, "orphans_total": 0,
        }

    if progress_cb is not None:
        try:
            progress_cb(0, n_pages)
        except Exception:
            log.exception("progress_cb (initial) raised")

    results: List[Any] = []
    page_timings: List[dict] = []
    total_tables = 0
    total_orphans = 0
    total_rotated_dropped = 0
    total_ocr_blocks = 0

    # OCR fallback is available only when the caller passed a full VNDocParser
    # (job/PDF path) with a recognition engine. The slim ``/parse-pdf-no-ocr``
    # diagnostic parser has no OCR — it stays pure pdfplumber.
    ocr_available = (
        getattr(parser, "ocr_engine", None) is not None
        and hasattr(parser, "_read_text")
    )

    try:
        with pdfplumber.open(str(pdf_path)) as plumber:
            for i in range(n_pages):
                page_t0 = time.perf_counter()

                t0 = time.perf_counter()
                pdfium_page = pdf_doc[i]
                bitmap = pdfium_page.render(scale=scale)
                pil = bitmap.to_pil().convert("RGB")
                bitmap.close()
                pdfium_page.close()
                rgb = np.asarray(pil)
                t_render = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                try:
                    raw_words = plumber.pages[i].extract_words(
                        x_tolerance=2, y_tolerance=2, keep_blank_chars=False,
                    ) or []
                    words = drop_rotated_words(raw_words)
                    dropped_rotated = len(raw_words) - len(words)
                except Exception:
                    log.exception("pdfplumber failed on page %d", i)
                    words = []
                    dropped_rotated = 0
                t_pdfp = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                try:
                    layout_blocks = parser.layout.predict(pil)
                except Exception:
                    log.exception("layout failed on page %d", i)
                    layout_blocks = []
                t_layout = (time.perf_counter() - t0) * 1000

                page_h, page_w = rgb.shape[:2]
                page = PageResult(page_index=i, width=page_w, height=page_h, angle=0)

                bgr_lazy: np.ndarray | None = None
                consumed: set[int] = set()
                page_tables_count = 0
                # Per-page OCR batch: collect line crops from every scan-only
                # text block on this page so one mega ``recognize_batch`` call
                # handles them all (instead of one batched-rec per block).
                # Each entry: (block_ref, [y_means], [x_mins], [line_crops]).
                pending_ocr_blocks: list = []
                t_tables_total = 0.0

                for lb in layout_blocks:
                    block = Block(
                        cls_id=lb.get("cls_id", -1),
                        label=lb["label"],
                        score=lb.get("score", 0.0),
                        bbox=tuple(int(v) for v in lb["bbox"]),
                        index=lb.get("index", len(page.blocks) + 1),
                    )
                    x0, y0, x1, y1 = block.bbox
                    pad = 4
                    bx0 = max(0, x0 - pad); by0 = max(0, y0 - pad)
                    bx1 = min(page_w, x1 + pad); by1 = min(page_h, y1 + pad)

                    if block.label in IMAGE_LIKE_LABELS:
                        crop_rgb = rgb[by0:by1, bx0:bx1]
                        if crop_rgb.size > 0:
                            fname = (
                                f"page_{i + 1:03d}_{block.label}_{block.index:02d}.jpg"
                            )
                            cv2.imwrite(
                                str(sub_dir / fname),
                                cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                            )
                            block.image_path = f"{image_subdir}/{fname}"
                        page.blocks.append(block)
                        continue

                    if block.label in TABLE_LABELS:
                        crop_rgb = rgb[by0:by1, bx0:bx1]
                        if crop_rgb.size > 0:
                            fname = (
                                f"page_{i + 1:03d}_{block.label}_{block.index:02d}.jpg"
                            )
                            cv2.imwrite(
                                str(sub_dir / fname),
                                cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                            )
                            block.image_path = f"{image_subdir}/{fname}"
                        if settings.parse_table_struct and parser.table_cls is not None:
                            if bgr_lazy is None:
                                bgr_lazy = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            tbl_crop = bgr_lazy[y0:y1, x0:x1]
                            if tbl_crop.size > 0:
                                pre_ocr = _build_precomputed_ocr_for_table(
                                    words, (x0, y0, x1, y1), scale,
                                )
                                # Scanned table (no embedded words): drop the
                                # empty precomputed_ocr so the extractor runs
                                # ONNX OCR on the cells instead of emitting a
                                # blank table.
                                table_pre = pre_ocr
                                if not pre_ocr and ocr_available:
                                    table_pre = None
                                t_t0 = time.perf_counter()
                                try:
                                    _, html = extract_table_both_parallel(
                                        parser, tbl_crop, precomputed_ocr=table_pre,
                                    )
                                except Exception:
                                    log.exception(
                                        "table_struct failed on page %d block %d",
                                        i, block.index,
                                    )
                                    html = ""
                                t_table_ms = (time.perf_counter() - t_t0) * 1000
                                t_tables_total += t_table_ms
                                if html:
                                    block.text = html
                                page_tables_count += 1
                                print(
                                    f"{log_prefix}   table page={i + 1} "
                                    f"block_idx={block.index} "
                                    f"ms={t_table_ms:.0f} html_chars={len(html)} "
                                    f"precomp_words={len(pre_ocr)}",
                                    flush=True,
                                )
                                # Mark words in table region as consumed so
                                # they don't also fall into "orphan".
                                for wi, w in enumerate(words):
                                    try:
                                        cx = (float(w["x0"]) + float(w["x1"])) / 2 * scale
                                        cy = (float(w["top"]) + float(w["bottom"])) / 2 * scale
                                    except (KeyError, TypeError, ValueError):
                                        continue
                                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                                        consumed.add(wi)
                        page.blocks.append(block)
                        continue

                    if block.label in TEXT_LIKE_LABELS or block.label in FORMULA_LABELS:
                        inside: list = []
                        for wi, w in enumerate(words):
                            if wi in consumed:
                                continue
                            try:
                                cx = (float(w["x0"]) + float(w["x1"])) / 2 * scale
                                cy = (float(w["top"]) + float(w["bottom"])) / 2 * scale
                            except (KeyError, TypeError, ValueError):
                                continue
                            if x0 <= cx <= x1 and y0 <= cy <= y1:
                                inside.append(w)
                                consumed.add(wi)
                        if inside:
                            block.text = format_pdfplumber_words(inside)
                            if block.label in TITLE_LABELS:
                                # Heading markdown (`#`, `##`, `**...**`) wraps
                                # only the first line — flatten wrapped headings.
                                block.text = " ".join(
                                    ln.strip() for ln in block.text.splitlines()
                                    if ln.strip()
                                )
                            block.extra["text_source"] = "pdfplumber"
                        elif ocr_available:
                            # No embedded text under this block → it's a
                            # scanned/image region. Run det now (per-block,
                            # variable crop size); defer rec — every line
                            # from every scan block on this page is recognised
                            # in a single mega ``recognize_batch`` call after
                            # the layout loop.
                            if bgr_lazy is None:
                                bgr_lazy = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            ocr_crop = bgr_lazy[by0:by1, bx0:bx1]
                            if ocr_crop.size > 0:
                                try:
                                    boxes, _ = parser.ocr_det.detect(ocr_crop)
                                except Exception:
                                    boxes = []
                                ys: list[float] = []
                                xs: list[float] = []
                                line_crops: list[np.ndarray] = []
                                for q in boxes:
                                    lc = OCRDet.crop_quad(ocr_crop, q)
                                    if lc.size == 0:
                                        continue
                                    ys.append(float(q[:, 1].mean()))
                                    xs.append(float(q[:, 0].min()))
                                    line_crops.append(lc)
                                if line_crops:
                                    pending_ocr_blocks.append(
                                        (block, ys, xs, line_crops)
                                    )
                                    block.extra["text_source"] = "ocr"
                        page.blocks.append(block)
                        continue

                    # Unknown / unsupported label — keep but leave empty.
                    page.blocks.append(block)

                # ── Mega-batched rec for every scan text block on this page ──
                # Stack every line crop from every pending block into ONE
                # ``recognize_batch`` call. PaddleOCRRec pads each line to the
                # model width and runs them as a single ONNX batch — orders
                # of magnitude fewer calls than per-block batched-rec, the
                # full benefit landing on GPU.
                if pending_ocr_blocks:
                    all_line_crops: list[np.ndarray] = []
                    spans: list[tuple] = []   # (block, ys, xs, start, end)
                    for blk, ys, xs, crops in pending_ocr_blocks:
                        spans.append((blk, ys, xs, len(all_line_crops),
                                      len(all_line_crops) + len(crops)))
                        all_line_crops.extend(crops)
                    if all_line_crops:
                        t_ocr0 = time.perf_counter()
                        try:
                            all_texts = parser.ocr_rec.recognize_batch(all_line_crops)
                        except Exception:
                            log.exception("page %d OCR mega-batch failed; falling back per-crop", i)
                            all_texts = [parser.ocr_rec.recognize(c) for c in all_line_crops]
                        t_ocr_ms = (time.perf_counter() - t_ocr0) * 1000
                        print(
                            f"{log_prefix}   ocr page={i + 1} blocks={len(spans)} "
                            f"lines={len(all_line_crops)} ms={t_ocr_ms:.0f}",
                            flush=True,
                        )

                    line_height_thresh = 12.0
                    for blk, ys, xs, s, e in spans:
                        triples = list(zip(ys, xs, all_texts[s:e]))
                        # Sort top-to-bottom (quantised) then left-to-right.
                        triples.sort(key=lambda t: (round(t[0] / 8.0) * 8.0, t[1]))
                        lines: list[str] = []
                        cur_line: list[str] = []
                        cur_y: Optional[float] = None
                        for y_mean, _x_min, raw in triples:
                            text = (raw or "").strip()
                            if not text:
                                continue
                            if cur_y is None or abs(y_mean - cur_y) <= line_height_thresh:
                                cur_line.append(text)
                                cur_y = y_mean if cur_y is None else cur_y
                            else:
                                lines.append(" ".join(cur_line))
                                cur_line = [text]
                                cur_y = y_mean
                        if cur_line:
                            lines.append(" ".join(cur_line))
                        if not lines:
                            continue
                        joined = "\n".join(lines)
                        if blk.label in TITLE_LABELS:
                            joined = " ".join(
                                ln.strip() for ln in joined.splitlines() if ln.strip()
                            )
                        blk.text = joined
                        total_ocr_blocks += 1

                orphans = [w for wi, w in enumerate(words) if wi not in consumed]
                if orphans:
                    orphan_block = Block(
                        cls_id=-1, label="text", score=0.0,
                        bbox=(0, 0, page_w, page_h),
                        index=len(page.blocks) + 1,
                    )
                    orphan_block.text = format_pdfplumber_words(orphans)
                    orphan_block.extra["orphan"] = True
                    orphan_block.extra["text_source"] = "pdfplumber_orphan"
                    page.blocks.append(orphan_block)

                results.append(page)
                total_tables += page_tables_count
                total_orphans += len(orphans)
                total_rotated_dropped += dropped_rotated

                t_total = (time.perf_counter() - page_t0) * 1000
                page_timings.append({
                    "page": i + 1,
                    "ms": round(t_total, 1),
                    "render_ms": round(t_render, 1),
                    "pdfplumber_ms": round(t_pdfp, 1),
                    "layout_ms": round(t_layout, 1),
                    "tables_ms": round(t_tables_total, 1),
                    "tables": page_tables_count,
                    "layout_blocks": len(layout_blocks),
                    "orphan_words": len(orphans),
                    "rotated_dropped": dropped_rotated,
                })
                print(
                    f"{log_prefix} page {i + 1}/{n_pages}  {t_total:.0f}ms  "
                    f"render={t_render:.0f} pdfp={t_pdfp:.0f} "
                    f"layout={t_layout:.0f} tables={t_tables_total:.0f}  "
                    f"blocks={len(layout_blocks)} tables_n={page_tables_count} "
                    f"orphan_words={len(orphans)} rotated_dropped={dropped_rotated}",
                    flush=True,
                )
                if progress_cb is not None:
                    try:
                        progress_cb(i + 1, n_pages)
                    except Exception:
                        log.exception("progress_cb raised")

    finally:
        pdf_doc.close()

    markdown = VNDocParser.to_markdown(results)
    overall_ms = (time.perf_counter() - overall_t0) * 1000
    # ``mode`` lives in a ``VARCHAR(64)`` DB column → keep it terse.
    # ``ocr=N`` = text blocks recognised by ONNX OCR (scanned/image regions);
    # 0 means the whole doc was born-digital (pure pdfplumber).
    mode = (
        f"hybrid:p={n_pages},t={total_tables},o={total_orphans},"
        f"rd={total_rotated_dropped},ocr={total_ocr_blocks}"
    )
    print(
        f"{log_prefix} DONE total={overall_ms:.0f}ms pages={n_pages} "
        f"md_chars={len(markdown)} tables_total={total_tables} "
        f"orphans_total={total_orphans}",
        flush=True,
    )
    return {
        "markdown": markdown,
        "n_pages": n_pages,
        "mode": mode,
        "page_timings": page_timings,
        "tables_total": total_tables,
        "orphans_total": total_orphans,
        "rotated_dropped_total": total_rotated_dropped,
        "total_ms": round(overall_ms, 1),
    }
