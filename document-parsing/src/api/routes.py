from __future__ import annotations

import base64
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from celery_app import celery_app
from core.registry import for_extension, registry
from infrastructure import s3
from infrastructure.db import ParsingJob
from infrastructure.repositories.job_repo import JobRepo
from settings import settings

from .schemas import (
    HealthResponse,
    JobByReferenceRequest,
    JobListResponse,
    JobProgress,
    JobResponse,
    JobResultLinks,
    JobSubmitResponse,
    ParseImage,
    ParseResponse,
)

log = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
#  Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    reg = registry()
    return HealthResponse(
        status="ok",
        parsers={ext: parser.name for ext, parser in reg.items()},
    )


# ---------------------------------------------------------------------------
#  Sync /parse — kept for small files / direct callers.
# ---------------------------------------------------------------------------

@router.post("/parse", response_model=ParseResponse)
async def parse(
    file: UploadFile = File(...),
    return_images: bool = Form(False),
    return_metadata: bool = Form(True),
) -> ParseResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if not ext:
        raise HTTPException(status_code=400, detail="file has no extension")

    parser = for_extension(ext)
    if parser is None:
        raise HTTPException(status_code=415, detail=f"unsupported extension: .{ext}")

    payload = await file.read()
    _enforce_size(payload)

    started = time.perf_counter()
    try:
        result = parser.parse(payload, file.filename)
    except Exception as e:
        log.exception("Parser %s failed on %s", parser.name, file.filename)
        raise HTTPException(status_code=500, detail=f"parser error: {e}") from e
    duration_ms = int((time.perf_counter() - started) * 1000)

    return ParseResponse(
        markdown=result.markdown,
        parser=result.parser,
        page_count=result.page_count,
        duration_ms=duration_ms,
        filename=file.filename,
        metadata=result.metadata if return_metadata else {},
        images=[
            ParseImage(name=img.name, mime=img.mime, data_base64=img.bytes_b64)
            for img in (result.images if return_images else [])
        ],
    )


# ---------------------------------------------------------------------------
#  Async /jobs
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {"png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp", "gif"}


@router.post("/parse-image")
async def parse_image_sync(file: UploadFile = File(...)) -> dict:
    """Sync image OCR — runs the full layout + OCR pipeline inline.

    No S3, no Celery. The image bytes are decoded, fed to the layout
    detector, and each text-bearing block goes through VietOCR. The
    response includes the rendered markdown, per-block details, and a
    per-stage timing breakdown. Each stage is also written to the API
    container's log.

    Best for: callers that want the result back in the same HTTP request
    (small images, single-page OCR). For multi-page PDFs use ``/jobs``.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in _IMAGE_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"image extensions only ({sorted(_IMAGE_EXTS)}); got .{ext}",
        )

    payload = await file.read()
    _enforce_size(payload)

    # Off-thread so we don't block the event loop while OCR runs (CPU-heavy).
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _parse_image_inline, payload, file.filename)


def _parse_image_inline(payload: bytes, filename: str) -> dict:
    """Synchronous image OCR pipeline executed in a worker thread.

    Phases (each timed and logged):
      decode_image     — bytes → PIL.Image
      orient_cls       — page rotation correction (optional)
      layout           — PP-DocLayoutV2 ONNX
      per-block        — for each layout block: crop+save / OCR / table struct
      markdown_assemble
    """
    import io
    import time
    import cv2
    import numpy as np
    from PIL import Image

    from core.timing import StageTimer, timed
    from parsers.pdf_layout import _get_parser

    print(f"[parse-image] start file={filename} bytes={len(payload)}", flush=True)
    overall_start = time.perf_counter()
    timer = StageTimer()

    with timer.stage("decode_image"):
        pil = Image.open(io.BytesIO(payload)).convert("RGB")
    print(f"[parse-image] decode_image = {timer.totals_s['decode_image']:.3f}s "
          f"({pil.size[0]}x{pil.size[1]})", flush=True)

    with timer.stage("warmup_singleton"):
        parser = _get_parser()
    if timer.totals_s["warmup_singleton"] > 0.5:
        print(f"[parse-image] warmup_singleton = {timer.totals_s['warmup_singleton']:.3f}s",
              flush=True)

    from vn_parser.ocr_det import OCRDet
    from vn_parser.pipeline import (
        FORMULA_LABELS,
        IMAGE_LIKE_LABELS,
        TABLE_LABELS,
        TEXT_LIKE_LABELS,
    )
    from concurrent.futures import ThreadPoolExecutor
    from parsers.pdf_layout import _detect_lines, _group_ocr_lines, extract_table_routed

    angle = 0
    with timer.stage("orient_cls"):
        if parser.orient is not None:
            pil, angle = parser.orient.correct(pil)
    print(f"[parse-image] orient_cls = {timer.totals_s.get('orient_cls', 0.0):.3f}s "
          f"(angle={angle})", flush=True)

    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── Layout + page-level DBNet in parallel (both ONNX, GIL-released) ─
    with timer.stage("layout_and_det_parallel"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_layout = pool.submit(parser.layout.predict, pil)
            fut_det = pool.submit(parser.ocr_det.detect, bgr)
            layout_blocks = fut_layout.result()
            all_quads, _scores = fut_det.result()
    print(f"[parse-image] layout || ocr_det_page = "
          f"{timer.totals_s.get('layout_and_det_parallel', 0):.3f}s "
          f"({len(layout_blocks)} layout blocks, "
          f"{len(all_quads) if all_quads is not None else 0} text lines)",
          flush=True)

    # Build line items (y_mean, x_min, line_crop_bgr) for ALL detected lines.
    line_items_all: list = []
    if all_quads is not None and len(all_quads) > 0:
        for q in all_quads:
            ys = q[:, 1].astype("float32")
            xs = q[:, 0].astype("float32")
            line_crop = OCRDet.crop_quad(bgr, q)
            if line_crop.size > 0:
                line_items_all.append((float(ys.mean()), float(xs.min()), line_crop, q))

    # Assign each line to a layout block by center containment. Lines that
    # don't fall in any block become "fallback" text blocks.
    block_records: list[dict] = []
    n_text = n_table = n_image = n_formula = 0
    block_lines: dict[int, list] = {}  # record_idx → [(y, x, crop, quad), ...]

    # First, build block records (in layout order).
    for lb in layout_blocks:
        x0, y0, x1, y1 = lb["bbox"]
        label = lb["label"]
        if label in IMAGE_LIKE_LABELS:
            kind = "image"; n_image += 1
        elif label in TABLE_LABELS:
            kind = "table"; n_table += 1
        elif label in FORMULA_LABELS:
            kind = "formula"; n_formula += 1
        elif label in TEXT_LIKE_LABELS:
            kind = "text"; n_text += 1
        else:
            kind = "other"
        block_records.append({
            "index": lb.get("index"),
            "label": label,
            "kind": kind,
            "bbox": lb["bbox"],
            "score": lb.get("score"),
            "text": None,
        })
        block_lines[len(block_records) - 1] = []

    # Assign lines. Skip lines inside tables — table_struct does its own
    # batched OCR on those, and running them again here serializes the
    # whole VietOCR pass into a giant single bucket which is slower on
    # CPU than 3 small concurrent batches.
    with timer.stage("assign_lines"):
        for line in line_items_all:
            y_mean, x_mean, crop, quad = line
            cx = float((quad[:, 0].mean()))
            cy = float((quad[:, 1].mean()))
            assigned_to = None
            for ri, lb in enumerate(layout_blocks):
                x0, y0, x1, y1 = lb["bbox"]
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    assigned_to = ri
                    break
            if assigned_to is None:
                continue
            if (
                settings.parse_table_struct
                and block_records[assigned_to]["kind"] == "table"
            ):
                continue
            block_lines[assigned_to].append(line)

    # Sort each block's lines top-to-bottom, left-to-right.
    for ri, items in block_lines.items():
        items.sort(key=lambda t: (round(t[0] / 8.0) * 8.0, t[1]))

    # ── Run heavy stages in parallel ─────────────────────────────────────
    all_crops: list = []
    offsets: list = []
    for ri in range(len(block_records)):
        items = block_lines[ri]
        offsets.append((len(all_crops), len(items)))
        all_crops.extend(it[2] for it in items)

    table_jobs: list = []  # (record_idx, crop_bgr)
    if settings.parse_table_struct and n_table:
        for ri, rec in enumerate(block_records):
            if rec["kind"] != "table":
                continue
            x0, y0, x1, y1 = rec["bbox"]
            crop = bgr[y0:y1, x0:x1]
            if crop.size > 0:
                table_jobs.append((ri, crop))

    # Parallel: rec_batch || N table_structs (each does its own
    # batched OCR via OCREngine — keeps work split across threads).
    if table_jobs:
        with timer.stage("rec_and_tables_parallel"):
            with ThreadPoolExecutor(max_workers=1 + len(table_jobs)) as pool:
                text_future = pool.submit(
                    parser.ocr_rec.recognize_batch, all_crops
                ) if all_crops else None
                table_futures = {
                    ri: pool.submit(extract_table_routed, parser, crop)
                    for ri, crop in table_jobs
                }
                texts = text_future.result() if text_future else []
                for ri, fut in table_futures.items():
                    kind, html = fut.result()
                    block_records[ri]["_html_table"] = html
                    block_records[ri]["_table_kind"] = kind
        print(f"[parse-image] rec || tables = "
              f"{timer.totals_s.get('rec_and_tables_parallel', 0):.3f}s "
              f"text_crops={len(all_crops)} tables={len(table_jobs)}",
              flush=True)
    elif all_crops:
        with timer.stage("ocr_rec_batch"):
            texts = parser.ocr_rec.recognize_batch(all_crops)
        print(f"[parse-image] ocr_rec_batch = "
              f"{timer.totals_s.get('ocr_rec_batch', 0):.3f}s "
              f"({len(all_crops)} crops)", flush=True)
    else:
        texts = []

    # Distribute recognized text back per block.
    for ri, rec in enumerate(block_records):
        start, count = offsets[ri]
        if count == 0:
            if rec.get("_html_table"):
                rec["_pending_md"] = rec["_html_table"]
                rec["text"] = rec["_html_table"]
            elif rec["kind"] == "image":
                rec["_pending_md"] = f"_[{rec['label']} bbox={rec['bbox']}]_"
            else:
                rec["_pending_md"] = ""
            continue

        items = block_lines[ri]
        text_out = _group_ocr_lines(
            [((y, x, c), t) for (y, x, c, _q), t in zip(items, texts[start:start + count])]
        )
        rec["text"] = text_out

        label = rec["label"]
        if rec.get("_html_table"):
            rec["_pending_md"] = rec["_html_table"]
        elif label in FORMULA_LABELS:
            rec["_pending_md"] = f"$$\n{text_out}\n$$" if text_out else ""
        elif label == "doc_title" and text_out:
            rec["_pending_md"] = f"# {text_out}"
        elif label == "paragraph_title" and text_out:
            rec["_pending_md"] = f"## {text_out}"
        elif label == "figure_title" and text_out:
            rec["_pending_md"] = f"**{text_out}**"
        else:
            rec["_pending_md"] = text_out

    md_pieces: list[str] = [r.pop("_pending_md", "") for r in block_records]
    for r in block_records:
        r.pop("_html_table", None)

    with timer.stage("markdown_assemble"):
        markdown = "\n\n".join(p for p in md_pieces if p)

    overall = time.perf_counter() - overall_start
    print(
        f"[parse-image] DONE file={filename} total={overall:.3f}s "
        f"(decode={timer.totals_s.get('decode_image', 0):.3f} "
        f"orient={timer.totals_s.get('orient_cls', 0):.3f} "
        f"layout||det={timer.totals_s.get('layout_and_det_parallel', 0):.3f} "
        f"assign={timer.totals_s.get('assign_lines', 0):.3f} "
        f"rec||tables={timer.totals_s.get('rec_and_tables_parallel', 0):.3f}) "
        f"blocks: text={n_text} table={n_table} image={n_image} formula={n_formula}",
        flush=True,
    )

    return {
        "filename": filename,
        "image_size": list(pil.size),
        "angle": angle,
        "markdown": markdown,
        "blocks": block_records,
        "summary": {
            "text_blocks": n_text,
            "table_blocks": n_table,
            "image_blocks": n_image,
            "formula_blocks": n_formula,
        },
        "timing_seconds": {
            **timer.as_dict(),
            "total": round(overall, 3),
        },
    }


class _DummyBlock:
    """vn_parser._extract_table mutates a Block.extra dict — give it a stub."""
    def __init__(self, label: str) -> None:
        self.label = label
        self.extra: dict = {}


# ---------------------------------------------------------------------------
#  Diagnostic: pdfplumber-only page-by-page parse
# ---------------------------------------------------------------------------

@router.post("/parse-pdf-text-only")
async def parse_pdf_text_only(file: UploadFile = File(...)) -> dict:
    """Iterate a PDF page-by-page using **only pdfplumber**.

    No layout detector, no DBNet, no VietOCR — establishes the floor for
    text-extraction throughput on a born-digital PDF.  Per-page progress
    + timing is printed to stdout so you can watch real-time progress via
    ``docker logs -f document-parsing-service`` while the request is open.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext != "pdf":
        raise HTTPException(status_code=415, detail="PDF only")

    payload = await file.read()
    _enforce_size(payload)

    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _parse_pdf_text_only_inline, payload, file.filename
    )


def _parse_pdf_text_only_inline(payload: bytes, filename: str) -> dict:
    import io
    import time as _time
    import pdfplumber

    overall_t0 = _time.perf_counter()
    print(f"[parse-pdf-text] start file={filename} bytes={len(payload)}", flush=True)

    page_md_parts: list[str] = []
    page_timings: list[dict] = []
    total_words = 0

    with pdfplumber.open(io.BytesIO(payload)) as pdf:
        n_pages = len(pdf.pages)
        print(f"[parse-pdf-text] opened {n_pages} pages", flush=True)

        for i, page in enumerate(pdf.pages):
            t0 = _time.perf_counter()
            try:
                words = page.extract_words(
                    x_tolerance=2, y_tolerance=2, keep_blank_chars=False,
                ) or []
                words = _drop_rotated_words(words)
            except Exception as e:
                elapsed_ms = (_time.perf_counter() - t0) * 1000
                print(
                    f"[parse-pdf-text] page {i + 1}/{n_pages}  ERROR after "
                    f"{elapsed_ms:.1f}ms: {e!r}",
                    flush=True,
                )
                page_md_parts.append(f"<!-- page {i + 1} (error: {e!r}) -->")
                page_timings.append({
                    "page": i + 1, "ms": round(elapsed_ms, 1),
                    "words": 0, "error": repr(e),
                })
                continue

            md = _format_pdfplumber_words(words)
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            wc = len(words)
            total_words += wc
            page_md_parts.append(f"<!-- page {i + 1} -->\n\n{md}")
            page_timings.append({
                "page": i + 1, "ms": round(elapsed_ms, 1), "words": wc,
            })
            print(
                f"[parse-pdf-text] page {i + 1}/{n_pages}  {elapsed_ms:.1f}ms  "
                f"{wc} words  ({len(md)} chars)",
                flush=True,
            )

    markdown = "\n\n".join(page_md_parts)
    overall_ms = (_time.perf_counter() - overall_t0) * 1000
    avg_ms = overall_ms / max(n_pages, 1)
    print(
        f"[parse-pdf-text] DONE  total={overall_ms:.0f}ms  pages={n_pages}  "
        f"words={total_words}  avg={avg_ms:.1f}ms/page  md_chars={len(markdown)}",
        flush=True,
    )

    return {
        "filename": filename,
        "pages": n_pages,
        "total_words": total_words,
        "total_ms": round(overall_ms, 1),
        "avg_ms_per_page": round(avg_ms, 1),
        "page_timings": page_timings,
        "markdown": markdown,
    }


def _drop_rotated_words(words: list) -> list:
    """Filter out pdfplumber words whose glyphs aren't upright.

    Rotated text (vertical ArXiv watermarks, page-margin labels printed at
    90°) is read character-by-character by pdfplumber and explodes a single
    visual phrase into many one-glyph "words" stacked vertically — corrupts
    the line-clustering output for the surrounding block. Each pdfplumber
    word dict carries ``upright`` (bool); ``False`` ⇒ skip.
    """
    return [w for w in words if w.get("upright", True)]


def _format_pdfplumber_words(words: list) -> str:
    """Group pdfplumber words into lines by vertical overlap, join in reading order.

    Uses an adaptive line threshold derived from the median word height in
    this block — a tight fixed value works for body text but splits large
    headings (Mem0 paper title etc.) when mixed-font baselines differ by
    several points. With overlap-based clustering, a word joins the
    current line whenever its bbox vertically overlaps the line's running
    bbox by at least ~30% of the typical word height.
    """
    if not words:
        return ""

    def _top(w):    return float(w.get("top", 0))
    def _bot(w):    return float(w.get("bottom", 0))
    def _x0(w):     return float(w.get("x0", 0))
    def _text(w):   return str(w.get("text", ""))
    def _h(w):      return max(_bot(w) - _top(w), 0.0)

    heights = sorted(_h(w) for w in words if _h(w) > 0)
    if heights:
        median_h = heights[len(heights) // 2]
    else:
        median_h = 10.0
    # Two words belong to the same line if their y-ranges overlap by at
    # least ``min_overlap`` PDF points. 30% of median height covers
    # superscripts, italic baselines, and mixed-font headings while
    # still separating distinct lines.
    min_overlap = max(median_h * 0.3, 1.5)

    # Sort by top (then x as tiebreaker) so we walk the page in reading order.
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
        # Vertical overlap between this word and the current line's bbox.
        overlap = min(cur_bot, b) - max(cur_top, t)
        if overlap >= min_overlap:
            lines[-1].append(w)
            cur_top = min(cur_top, t)
            cur_bot = max(cur_bot, b)
        else:
            lines.append([w])
            cur_top, cur_bot = t, b

    out_lines: list[str] = []
    for line in lines:
        line.sort(key=_x0)
        out_lines.append(" ".join(_text(w) for w in line if _text(w)))
    return "\n".join(ln for ln in out_lines if ln.strip()).strip()


# ---------------------------------------------------------------------------
#  Diagnostic: layout + tables + pdfplumber (no OCR)
# ---------------------------------------------------------------------------

@router.post("/parse-pdf-no-ocr")
async def parse_pdf_no_ocr(
    file: UploadFile = File(...),
    return_images: bool = Form(False),
) -> dict:
    """Parse a born-digital PDF with layout + table struct, but **no OCR**.

    For every page:
      • PP-DocLayoutV2 ONNX → block bboxes (title / paragraph / table / image / formula / ...)
      • pdfplumber → word list with bboxes
      • text-bearing block ← pdfplumber words whose center is inside the bbox
      • table block        ← table_cls + UNet/SLANet+, fed precomputed OCR
                              from the pdfplumber words inside the table bbox
                              (no DBNet, no VietOCR — saves ~5-15 s per table)
      • image/figure block ← cropped to disk
      • orphan words (not in any layout block) → emitted as plain text
        in reading order at the end of the page

    Per-page progress + stage timing is printed to stdout — follow with
    ``docker logs -f document-parsing-service``.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext != "pdf":
        raise HTTPException(status_code=415, detail="PDF only")

    payload = await file.read()
    _enforce_size(payload)

    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _parse_pdf_no_ocr_inline, payload, file.filename, return_images,
    )


def _build_precomputed_ocr_for_table(
    words: list,
    table_bbox_img: tuple[int, int, int, int],
    scale: float,
) -> list:
    """Convert pdfplumber words that fall inside ``table_bbox_img`` into the
    ``precomputed_ocr`` shape ``extract_table_routed`` expects:

        [[quad_4x2_local, (text, score)], ...]

    ``table_bbox_img`` is in render-image pixel coordinates; ``words`` are in
    PDF points → multiply by ``scale`` to get image pixels, then subtract the
    table top-left to express in local (table-crop) coordinates.
    """
    import numpy as np
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


def _parse_pdf_no_ocr_inline(payload: bytes, filename: str, return_images: bool) -> dict:
    import tempfile
    from pathlib import Path

    from parsers.pipeline.pipeline_no_ocr import parse_pdf_no_ocr
    from parsers.pdf_layout import _collect_images, _get_parser

    print(f"[parse-pdf-no-ocr] start file={filename} bytes={len(payload)}", flush=True)
    parser = _get_parser()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.pdf"
        input_path.write_bytes(payload)
        image_dir = tmp_path / "out"
        image_dir.mkdir(parents=True, exist_ok=True)

        result = parse_pdf_no_ocr(
            parser=parser,
            pdf_path=input_path,
            image_dir=image_dir,
            image_subdir="images",
            log_prefix="[parse-pdf-no-ocr]",
        )
        extracted_images = _collect_images(image_dir) if return_images else []

    return {
        "filename": filename,
        "pages": result["n_pages"],
        "total_ms": result["total_ms"],
        "avg_ms_per_page": round(
            result["total_ms"] / max(result["n_pages"], 1), 1,
        ),
        "tables_total": result["tables_total"],
        "orphans_total": result["orphans_total"],
        "page_timings": result["page_timings"],
        "markdown": result["markdown"],
        "images": [
            {"name": img.name, "mime": img.mime, "data_base64": img.bytes_b64}
            for img in extracted_images
        ],
    }


# ---------------------------------------------------------------------------
#  Async /jobs
# ---------------------------------------------------------------------------

@router.post("/jobs", response_model=JobSubmitResponse, status_code=202)
async def submit_job(file: UploadFile = File(...)) -> JobSubmitResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if not ext:
        raise HTTPException(status_code=400, detail="file has no extension")
    return await _submit_for_extension(file, ext)


async def _submit_for_extension(file: UploadFile, ext: str) -> JobSubmitResponse:
    parser = for_extension(ext)
    if parser is None:
        raise HTTPException(status_code=415, detail=f"unsupported extension: .{ext}")

    payload = await file.read()
    _enforce_size(payload)

    s3_key_template = "{job_id}/input." + ext
    # Insert DB row first so we know the id, then upload to S3 under that id.
    job = JobRepo.create(filename=file.filename, s3_input_key="", parser=parser.name)
    s3_key = s3_key_template.format(job_id=str(job.id))
    s3.put_bytes(s3_key, payload, _content_type_for(ext))
    JobRepo.update_input_key(job.id, s3_key)

    celery_app.send_task(
        "celery_app.tasks.parse_tasks.parse_document_task",
        args=[str(job.id)],
        queue=settings.celery_queue,
        routing_key=settings.celery_routing_key,
    )
    return JobSubmitResponse(id=job.id, state=job.state, filename=job.filename)


@router.post("/jobs/by-reference", response_model=JobSubmitResponse, status_code=202)
def submit_job_by_reference(req: JobByReferenceRequest) -> JobSubmitResponse:
    """Submit a parse job for a file already in S3 (no upload).

    The orchestrator uploads the file once into its own bucket and asks
    document-parsing to read it back from there. ``callback_url``, if
    supplied, receives running/done/failed updates from the worker.
    """
    ext = Path(req.filename).suffix.lstrip(".").lower()
    if not ext:
        raise HTTPException(status_code=400, detail="filename has no extension")

    parser = for_extension(ext)
    if parser is None:
        raise HTTPException(status_code=415, detail=f"unsupported extension: .{ext}")

    if not req.source_url.startswith("s3://"):
        raise HTTPException(
            status_code=400,
            detail="source_url must be an s3:// URL",
        )

    metadata = {
        "callback_url": req.callback_url,
        "external_document_id": req.external_document_id,
        "source_url": req.source_url,
    }
    job = JobRepo.create(
        filename=req.filename,
        s3_input_key=req.source_url,  # full URL — worker uses download_url_to_file
        parser=parser.name,
        metadata=metadata,
    )

    celery_app.send_task(
        "celery_app.tasks.parse_tasks.parse_document_task",
        args=[str(job.id)],
        queue=settings.celery_queue,
        routing_key=settings.celery_routing_key,
    )
    return JobSubmitResponse(id=job.id, state=job.state, filename=job.filename)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: uuid.UUID) -> JobResponse:
    job = JobRepo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _to_response(job)


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(limit: int = 50) -> JobListResponse:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be in [1, 200]")
    jobs = JobRepo.list_recent(limit=limit)
    return JobListResponse(
        jobs=[_to_response(j) for j in jobs],
        total=len(jobs),
    )


@router.delete("/jobs/{job_id}")
def delete_job(job_id: uuid.UUID) -> dict:
    job = JobRepo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    deleted = s3.delete_prefix(f"{job_id}/")
    JobRepo.delete(job_id)
    return {"id": str(job_id), "deleted_objects": deleted}


@router.get("/jobs/{job_id}/markdown")
def get_job_markdown(job_id: uuid.UUID, presign: bool = False) -> dict | str:
    job = JobRepo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state != "done" or not job.s3_markdown_key:
        raise HTTPException(status_code=409, detail=f"markdown not ready (state={job.state})")
    if presign:
        return {"url": s3.presign_get(job.s3_markdown_key)}
    return s3.get_bytes(job.s3_markdown_key).decode("utf-8")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _enforce_size(payload: bytes) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {settings.max_upload_mb} MB limit",
        )


_CT = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "html": "text/html", "htm": "text/html",
    "md": "text/markdown", "txt": "text/plain", "json": "application/json",
    "csv": "text/csv", "epub": "application/epub+zip",
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "webp": "image/webp", "gif": "image/gif",
    "tif": "image/tiff", "tiff": "image/tiff", "bmp": "image/bmp",
}


def _content_type_for(ext: str) -> str:
    return _CT.get(ext.lower(), "application/octet-stream")


def _to_response(job: ParsingJob) -> JobResponse:
    pct = None
    if job.pages_total:
        pct = round(100.0 * (job.pages_done or 0) / job.pages_total, 1)

    result = None
    if job.state == "done" and job.s3_markdown_key:
        result = JobResultLinks(
            markdown_key=job.s3_markdown_key,
            markdown_url=s3.presign_get(job.s3_markdown_key),
            image_count=job.image_count,
            image_prefix=job.s3_image_prefix,
        )

    return JobResponse(
        id=job.id,
        filename=job.filename,
        state=job.state,
        parser=job.parser,
        mode=job.mode,
        progress=JobProgress(
            pages_done=job.pages_done or 0,
            pages_total=job.pages_total,
            pct=pct,
        ),
        result=result,
        error=job.error,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        duration_ms=job.duration_ms,
    )
