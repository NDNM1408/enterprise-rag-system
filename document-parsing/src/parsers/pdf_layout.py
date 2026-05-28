"""PDF / image parser backed by the ``vn_parser`` pipeline.

The ``vn_parser`` package is vendored under ``src/vendored/vn_parser`` and
imported via ``PYTHONPATH=/app/src/vendored``; only the model weights (3.8 GB
of ONNX + torch checkpoints) are mounted at runtime via ``PARSER_MODELS_DIR``.

Two execution modes:

* ``hybrid`` (default, ``PDF_HYBRID_MODE=true``) — for PDFs with a text
  layer, use pdfplumber to harvest text and run the layout model to
  crop figures/tables/charts. Skips VietOCR entirely (~50–100× faster).
  Pages with no text layer fall back to full OCR.

* ``full`` (``PDF_HYBRID_MODE=false``) — original pipeline: render →
  layout → OCR-det → VietOCR per text block → table struct.

If the vn_parser pipeline isn't installed or its models aren't on disk,
importing this module raises and the registry falls back to ``pdf_plain``.
"""
from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

import pdfplumber
import pypdfium2 as pdfium

from core.base import BaseParser, ExtractedImage, ParseResult, ProgressCallback
from core.timing import StageTimer
from settings import settings

log = logging.getLogger(__name__)

# pdfminer logs a "CropBox missing" warning per page on PDFs without
# explicit CropBox metadata. Harmless — silence to keep parse logs readable.
logging.getLogger("pdfminer").setLevel(logging.ERROR)

_MODELS_DIR = Path(settings.models_dir)


def _ensure_parser_importable() -> None:
    """Verify required model weights exist and the vendored vn_parser imports.

    The package itself lives in ``src/vendored/vn_parser`` (added to
    PYTHONPATH at container build); only the ONNX + torch weights need
    a runtime path.
    """
    if not _MODELS_DIR.exists():
        raise RuntimeError(
            f"Parser models dir not found: {_MODELS_DIR}. "
            "Mount weights via PARSER_MODELS_DIR."
        )
    required = ("layout.onnx", "ocr_det.onnx")
    missing = [n for n in required if not (_MODELS_DIR / n).exists()]
    if missing:
        raise RuntimeError(
            f"Parser models incomplete; missing: {missing} in {_MODELS_DIR}"
        )
    # Trigger import early so missing python deps surface at startup.
    import vn_parser  # noqa: F401


_ensure_parser_importable()

_INSTANCE: Any | None = None
_INSTANCE_LOCK = Lock()


def _build_parser():
    """Build a VNDocParser, then re-instantiate each component with the
    per-stage device the operator configured. The vendored ``VNDocParser``
    constructor takes a single ``providers=`` for all ONNX models and a
    single ``device=`` for the torch ones, so per-stage control happens
    via post-init swap.
    """
    from vn_parser import VNDocParser
    from vn_parser.layout import LayoutDetector
    from vn_parser.ocr_adapter import OCREngine
    from vn_parser.ocr_det import OCRDet
    from vn_parser.orient_cls import OrientationClassifier
    from vn_parser.table_cls import TableClassifier

    from core.devices import (
        describe_provider,
        resolve_onnx_providers,
    )

    log.info(
        "Initializing VNDocParser (models=%s, hybrid=%s)",
        _MODELS_DIR, settings.pdf_hybrid_mode,
    )

    # Per-stage ONNX providers. Every stage — including text recognition —
    # is ONNX now, so each gets a providers list straight from the device
    # spec (no torch device resolution).
    layout_providers = resolve_onnx_providers(settings.device_layout)
    ocr_det_providers = resolve_onnx_providers(settings.device_ocr_det)
    orient_providers = resolve_onnx_providers(settings.device_orient)
    table_cls_providers = resolve_onnx_providers(settings.device_table_cls)
    ocr_rec_providers = resolve_onnx_providers(settings.device_ocr_rec)

    # Build with the resolved providers up front — no stage-2 swap needed
    # now that recognition is ONNX too.
    parser = VNDocParser(
        models_dir=str(_MODELS_DIR),
        layout_conf=settings.layout_conf,
        rec_model=settings.rec_model,
        rec_char_dict=settings.rec_char_dict,
        rec_img_shape=(3, settings.rec_img_h, settings.rec_img_w),
        rec_use_space=settings.rec_use_space,
        rec_providers=ocr_rec_providers,
        providers=["CPUExecutionProvider"],
    )

    # Swap the remaining ONNX stages onto their per-stage providers.
    parser.layout = LayoutDetector(
        _MODELS_DIR / "layout.onnx",
        conf=settings.layout_conf,
        providers=layout_providers,
    )
    parser.ocr_det = OCRDet(
        _MODELS_DIR / "ocr_det.onnx",
        providers=ocr_det_providers,
    )
    if parser.orient is not None:
        parser.orient = OrientationClassifier(
            _MODELS_DIR / "orient_cls.onnx",
            providers=orient_providers,
        )
    if parser.table_cls is not None:
        parser.table_cls = TableClassifier(
            _MODELS_DIR / "table_cls.onnx",
            providers=table_cls_providers,
        )

    # Rebuild the OCREngine so it points at the swapped det + the ONNX rec.
    parser.ocr_engine = OCREngine(parser.ocr_det, parser.ocr_rec)

    if not settings.parse_table_wireless and parser.table_wireless is not None:
        log.info("[devices] dropping table_wireless (PARSE_TABLE_WIRELESS=false)")
        parser.table_wireless = None

    log.info(
        "[devices] layout=%s ocr_det=%s ocr_rec=%s orient=%s table_cls=%s",
        describe_provider(layout_providers),
        describe_provider(ocr_det_providers),
        describe_provider(ocr_rec_providers),
        describe_provider(orient_providers) if parser.orient else "off",
        describe_provider(table_cls_providers) if parser.table_cls else "off",
    )
    return parser


def _get_parser():
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = _build_parser()
    return _INSTANCE


class LayoutPdfParser(BaseParser):
    name = "layout-vn-parser"
    extensions = ("pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp")

    def parse(
        self,
        payload: bytes,
        filename: str,
        progress_cb: ProgressCallback | None = None,
    ) -> ParseResult:
        ext = Path(filename).suffix.lower() or ".pdf"
        is_pdf = ext == ".pdf"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / f"input{ext}"
            input_path.write_bytes(payload)
            image_dir = tmp_path / "out"
            image_dir.mkdir()

            if is_pdf and settings.pdf_hybrid_mode:
                md, page_count, mode = self._parse_pdf(
                    input_path, image_dir, progress_cb=progress_cb,
                )
            else:
                md, page_count = self._parse_full(input_path, image_dir)
                mode = "full"

            images = _collect_images(image_dir)

        return ParseResult(
            markdown=md,
            parser=self.name,
            page_count=page_count,
            metadata={
                "extension": ext.lstrip("."),
                "mode": mode,
                "dpi": settings.dpi,
            },
            images=images,
        )

    def _parse_pdf(
        self,
        input_path: Path,
        image_dir: Path,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> tuple[str, int, str]:
        """Default PDF path: layout + table_struct + pdfplumber (no OCR).

        Runs the shared ``parse_pdf_no_ocr`` pipeline so the diagnostic
        ``/parse-pdf-no-ocr`` endpoint and the async ``/jobs`` worker
        produce identical output and per-page progress logs.

        ``ENABLE_CPU_BATCHED_PIPELINE=false`` falls back to the legacy
        hybrid path (full DBNet + VietOCR per page) for parity with the
        original behaviour.
        """
        if not settings.enable_cpu_batched_pipeline:
            return self._parse_hybrid(input_path, image_dir, progress_cb=progress_cb)

        from parsers.pipeline.pipeline_no_ocr import parse_pdf_no_ocr

        parser = _get_parser()
        result = parse_pdf_no_ocr(
            parser=parser,
            pdf_path=input_path,
            image_dir=image_dir,
            image_subdir="images",
            progress_cb=progress_cb,
            log_prefix="[parse-pdf-no-ocr]",
        )
        return result["markdown"], result["n_pages"], result["mode"]

    # ------------------------------------------------------------------
    #  full-OCR mode — original VNDocParser.parse()
    # ------------------------------------------------------------------
    def _parse_full(self, input_path: Path, image_dir: Path) -> tuple[str, int]:
        parser = _get_parser()
        results = parser.parse(
            str(input_path),
            dpi=settings.dpi,
            image_dir=str(image_dir),
        )
        return parser.to_markdown(results), len(results)

    # ------------------------------------------------------------------
    #  hybrid mode — pdfplumber text + vn_parser layout-only
    # ------------------------------------------------------------------
    def _parse_hybrid(
        self,
        input_path: Path,
        image_dir: Path,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> tuple[str, int, str]:
        """Per-block mixed pipeline.

        For every page:
          1. Render at DPI → PIL image.
          2. Run layout detector → blocks (text, table, image, formula, ...).
          3. For each text-bearing block, count pdfplumber words inside its
             bbox. If ≥ ``pdf_block_min_words`` → use the embedded text.
             Else → OCR just that crop with VietOCR.
          4. Image / table blocks are cropped to disk as figures (no OCR).

        This handles mixed pages cleanly: the body paragraphs of a born-
        digital PDF take the fast pdfplumber path, while text baked into
        a diagram (caption inside a figure raster) gets per-region OCR.
        """
        import cv2
        import numpy as np

        from vn_parser.pipeline import (
            Block,
            FORMULA_LABELS,
            IMAGE_LIKE_LABELS,
            PageResult,
            TABLE_LABELS,
            TEXT_LIKE_LABELS,
        )

        parser = _get_parser()
        dpi = settings.dpi
        scale = dpi / 72.0
        min_words = settings.pdf_block_min_words

        results: list[PageResult] = []
        total_ocr_blocks = 0
        total_text_blocks = 0
        timer = StageTimer()

        pdf_doc = pdfium.PdfDocument(str(input_path))
        try:
            with pdfplumber.open(str(input_path)) as plumber:
                n_pages = len(pdf_doc)
                image_subdir = "images"

                # Initial progress tick — sets pages_total in the DB so
                # clients see the denominator before page 10 finishes.
                if progress_cb is not None:
                    try:
                        progress_cb(0, n_pages)
                    except Exception:
                        log.exception("progress callback (initial) raised")

                import time as _time
                for i in range(n_pages):
                    page_t0 = _time.perf_counter()
                    page_timer = StageTimer()  # per-page breakdown for logging

                    plumber_page = plumber.pages[i]
                    with timer.stage("pdfplumber_words"), page_timer.stage("pdfplumber_words"):
                        words = plumber_page.extract_words(
                            x_tolerance=2, y_tolerance=2, keep_blank_chars=False,
                        )

                    with timer.stage("pdfium_render"), page_timer.stage("pdfium_render"):
                        pdfium_page = pdf_doc[i]
                        bitmap = pdfium_page.render(scale=scale)
                        pil_image = bitmap.to_pil().convert("RGB")
                        bitmap.close()
                        pdfium_page.close()

                    page, page_ocr_blocks, page_text_blocks = self._parse_page_hybrid(
                        parser, words, pil_image, i, scale,
                        image_dir, image_subdir, min_words,
                        cv2, np, timer, page_timer,
                        Block, PageResult, IMAGE_LIKE_LABELS,
                        TABLE_LABELS, FORMULA_LABELS, TEXT_LIKE_LABELS,
                    )
                    results.append(page)
                    total_ocr_blocks += page_ocr_blocks
                    total_text_blocks += page_text_blocks

                    # Per-page log line — use print(flush=True) so Celery's
                    # stdout capture delivers it in real time instead of
                    # batching through the logger.
                    page_total = _time.perf_counter() - page_t0
                    print(
                        f"[page {i + 1}/{n_pages}] total={page_total:.2f}s "
                        f"{page_timer.summary()} "
                        f"(text={page_text_blocks} ocr={page_ocr_blocks})",
                        flush=True,
                    )

                    if (i + 1) % 5 == 0 or (i + 1) == n_pages:
                        print(
                            f"[progress] {i + 1}/{n_pages} pages "
                            f"(text-blocks={total_text_blocks}, ocr-fallback-blocks={total_ocr_blocks}) | "
                            f"cumulative: {timer.summary()}",
                            flush=True,
                        )
                        if progress_cb is not None:
                            try:
                                progress_cb(i + 1, n_pages)
                            except Exception:
                                log.exception("progress callback raised")
                if progress_cb is not None and n_pages % 10 != 0:
                    try:
                        progress_cb(n_pages, n_pages)
                    except Exception:
                        log.exception("progress callback (final) raised")
        finally:
            pdf_doc.close()

        from vn_parser import VNDocParser
        with timer.stage("markdown_serialize"):
            md = VNDocParser.to_markdown(results)
        mode = f"hybrid:text={total_text_blocks},ocr={total_ocr_blocks}"
        log.info("[timing] %s", timer.summary())
        return md, len(results), mode

    @staticmethod
    def _parse_page_hybrid(
        parser, words, pil_image, page_index, scale,
        image_dir, image_subdir, min_words,
        cv2, np, timer, page_timer,
        Block, PageResult, IMAGE_LIKE_LABELS,
        TABLE_LABELS, FORMULA_LABELS, TEXT_LIKE_LABELS,
    ):
        """Hybrid per-region pipeline (text-first DBNet).

        Phases:
          1. ``layout.predict`` — get region bboxes.
          2. Classify each block:
               - image / table → crop saved as JPEG
               - text / formula:
                   pdfplumber words inside bbox ≥ ``min_words`` → use them
                                                 < min_words → fallback to OCR
          3. **Full-page DBNet**: if any fallback blocks exist, run DBNet
             ONCE on the full page rendered image and assign quads to
             fallback blocks by center containment (skips quads inside
             table blocks — table_struct handles those).
          4. Parallel: page-level batched VietOCR over fallback line
             crops || per-table struct (UNet/SLANet+ via
             ``extract_table_routed``). Skips the thread pool when
             there's no parallel work.
          5. Group OCR'd lines back to each fallback block; attach HTML
             table results to table blocks.

        Returns ``(PageResult, n_ocr_blocks, n_text_blocks)``.
        """
        from concurrent.futures import ThreadPoolExecutor

        from vn_parser.ocr_det import OCRDet

        with timer.stage("layout"), page_timer.stage("layout"):
            layout_blocks = parser.layout.predict(pil_image)

        sub_dir = image_dir / image_subdir
        sub_dir.mkdir(parents=True, exist_ok=True)

        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        blocks: list = []
        n_ocr_blocks = 0
        n_text_blocks = 0

        fallback_indices: list[int] = []
        table_indices: list[int] = []

        # ── Phase 2: classify ────────────────────────────────────────────
        for lb in layout_blocks:
            label = lb["label"]
            x0_img, y0_img, x1_img, y1_img = lb["bbox"]
            score = lb["score"]
            index = lb["index"]
            cls_id = lb["cls_id"]

            block = Block(
                cls_id=cls_id, label=label, score=score,
                bbox=(x0_img, y0_img, x1_img, y1_img), index=index,
            )

            if label in IMAGE_LIKE_LABELS:
                with timer.stage("crop_save"), page_timer.stage("crop_save"):
                    _save_block_crop(pil_image, x0_img, y0_img, x1_img, y1_img,
                                     page_index, label, index, sub_dir,
                                     image_subdir, block)
            elif label in TABLE_LABELS:
                with timer.stage("crop_save"), page_timer.stage("crop_save"):
                    _save_block_crop(pil_image, x0_img, y0_img, x1_img, y1_img,
                                     page_index, label, index, sub_dir,
                                     image_subdir, block)
                if settings.parse_table_struct:
                    table_indices.append(len(blocks))

            elif label in TEXT_LIKE_LABELS or label in FORMULA_LABELS:
                with timer.stage("words_match"), page_timer.stage("words_match"):
                    contained = _words_inside(words, x0_img, y0_img, x1_img, y1_img, scale)

                if len(contained) >= min_words:
                    with timer.stage("text_pdfplumber"), page_timer.stage("text_pdfplumber"):
                        block.text = _format_lines(contained)
                    block.extra["text_source"] = "pdfplumber"
                    n_text_blocks += 1
                else:
                    block.extra["text_source"] = "vietocr"
                    n_ocr_blocks += 1
                    fallback_indices.append(len(blocks))

            blocks.append(block)

        # ── Phase 3: full-page DBNet — fallback blocks only (tables skipped) ─
        # Table-region lines are NOT included here. table_struct runs its
        # own batched OCR on table cells via OCREngine — splitting that
        # work into parallel sessions is faster on CPU than a single
        # giant batch over all 100+ line crops.
        block_lines: dict[int, list] = {bi: [] for bi in fallback_indices}
        if fallback_indices:
            with timer.stage("ocr_det_page"), page_timer.stage("ocr_det_page"):
                all_quads, _scores = parser.ocr_det.detect(bgr)
            if all_quads is not None and len(all_quads) > 0:
                fallback_bboxes = [
                    (bi, blocks[bi].bbox) for bi in fallback_indices
                ]
                table_bboxes = (
                    [blocks[ti].bbox for ti in table_indices]
                    if settings.parse_table_struct else []
                )
                with timer.stage("assign_lines"), page_timer.stage("assign_lines"):
                    for q in all_quads:
                        cx = float(q[:, 0].mean())
                        cy = float(q[:, 1].mean())
                        # Skip lines inside a table — table_struct handles them.
                        if any(tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]
                               for tb in table_bboxes):
                            continue
                        for bi, (x0, y0, x1, y1) in fallback_bboxes:
                            if x0 <= cx <= x1 and y0 <= cy <= y1:
                                ys = q[:, 1].astype("float32")
                                xs = q[:, 0].astype("float32")
                                line_crop = OCRDet.crop_quad(bgr, q)
                                if line_crop.size > 0:
                                    block_lines[bi].append((
                                        float(ys.mean()),
                                        float(xs.min()),
                                        line_crop,
                                    ))
                                break

            for bi in fallback_indices:
                block_lines[bi].sort(key=lambda t: (round(t[0] / 8.0) * 8.0, t[1]))

        # ── Phase 4: rec_batch || N table_structs in parallel ───────────
        # Each table_struct runs its own batched OCR via OCREngine —
        # parallel sessions are faster on CPU than one giant batch.
        all_crops: list = []
        offsets: list = []
        for bi in fallback_indices:
            items = block_lines[bi]
            offsets.append((bi, len(all_crops), len(items)))
            all_crops.extend(it[2] for it in items)

        table_crops: list = []
        if settings.parse_table_struct:
            for ti in table_indices:
                x0, y0, x1, y1 = blocks[ti].bbox
                crop = bgr[y0:y1, x0:x1]
                if crop.size > 0:
                    table_crops.append((ti, crop))

        def _do_rec_safe() -> list[str]:
            try:
                return parser.ocr_rec.recognize_batch(all_crops)
            except Exception as e:
                log.warning("batched OCR failed on page %d: %s", page_index + 1, e)
                return ["" for _ in all_crops]

        if table_crops and all_crops:
            with timer.stage("rec_and_tables_parallel"), page_timer.stage("rec_and_tables_parallel"):
                with ThreadPoolExecutor(max_workers=1 + len(table_crops)) as pool:
                    rec_future = pool.submit(_do_rec_safe)
                    table_futures = {
                        ti: pool.submit(extract_table_routed, parser, crop)
                        for ti, crop in table_crops
                    }
                    texts = rec_future.result()
                    for ti, fut in table_futures.items():
                        try:
                            kind, html = fut.result()
                        except Exception as e:
                            log.warning("table_struct failed on page %d block %d: %s",
                                        page_index + 1, blocks[ti].index, e)
                            kind, html = "fallback", ""
                        if html:
                            blocks[ti].text = html
                            blocks[ti].extra["table_kind"] = kind
        elif table_crops:
            texts = []
            stage_name = "tables_parallel" if len(table_crops) > 1 else "table_struct"
            with timer.stage(stage_name), page_timer.stage(stage_name):
                if len(table_crops) > 1:
                    with ThreadPoolExecutor(max_workers=len(table_crops)) as pool:
                        futures = {ti: pool.submit(extract_table_routed, parser, crop)
                                   for ti, crop in table_crops}
                        for ti, fut in futures.items():
                            try:
                                kind, html = fut.result()
                            except Exception as e:
                                log.warning("table_struct failed on page %d: %s",
                                            page_index + 1, e)
                                continue
                            if html:
                                blocks[ti].text = html
                                blocks[ti].extra["table_kind"] = kind
                else:
                    ti, crop = table_crops[0]
                    try:
                        kind, html = extract_table_routed(parser, crop)
                        if html:
                            blocks[ti].text = html
                            blocks[ti].extra["table_kind"] = kind
                    except Exception as e:
                        log.warning("table_struct failed on page %d: %s",
                                    page_index + 1, e)
        elif all_crops:
            with timer.stage("ocr_rec_batch"), page_timer.stage("ocr_rec_batch"):
                texts = _do_rec_safe()
        else:
            texts = []

        # ── Phase 5: distribute OCR'd texts back to fallback blocks ─────
        for bi, start, count in offsets:
            items = block_lines[bi]
            block_lines_pairs = list(zip(items, texts[start:start + count]))
            blocks[bi].text = _group_ocr_lines(block_lines_pairs)

        page = PageResult(
            page_index=page_index,
            width=pil_image.width,
            height=pil_image.height,
            angle=0,
            blocks=blocks,
        )
        return page, n_ocr_blocks, n_text_blocks


def _save_block_crop(pil_image, x0, y0, x1, y1, page_index, label,
                     index, sub_dir, image_subdir, block) -> None:
    """Crop an image / table region from the page and save as JPEG."""
    pad = 4
    w_img, h_img = pil_image.size
    box = (
        max(0, x0 - pad), max(0, y0 - pad),
        min(w_img, x1 + pad), min(h_img, y1 + pad),
    )
    crop = pil_image.crop(box)
    fname = f"page_{page_index + 1:03d}_{label}_{index:02d}.jpg"
    crop.save(sub_dir / fname, format="JPEG", quality=85)
    block.image_path = f"{image_subdir}/{fname}"


def extract_table_routed(parser, crop_bgr, precomputed_ocr=None) -> tuple[str, str]:
    """Table struct with table_cls dispatch — runs ONLY the matching
    structure model (UNet for wired, SLANet+ for wireless) instead of
    both. Roughly halves table_struct time vs the dual-run + heuristic
    pick used by the original ``_extract_table``.

    ``precomputed_ocr`` is the recommended fast path: pass
    ``[[quad_local, (text, score)], ...]`` already detected + recognized
    on the page (with quads translated to table-crop-local coords). The
    structure models accept it directly, saving a DBNet + VietOCR pass
    per table — ~3-5 s on CPU.

    Returns ``(kind, html)``. ``kind`` is ``"wired"``, ``"wireless"``, or
    ``"fallback"`` if neither model is available.
    """
    import cv2

    if parser.table_wired is None and parser.table_wireless is None:
        return ("fallback", "")

    from vn_parser.table_unet.main import escape_html

    if precomputed_ocr is not None:
        ocr_result = [
            [item[0], escape_html(item[1][0]), item[1][1]]
            for item in precomputed_ocr
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], tuple)
        ]
    else:
        # Fallback: detect + recognize inside the table crop.
        try:
            ocr_raw = parser.ocr_engine.ocr(crop_bgr)[0]
        except Exception:
            ocr_raw = []
        ocr_result = [
            [item[0], escape_html(item[1][0]), item[1][1]]
            for item in (ocr_raw or [])
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], tuple)
        ]

    # Dispatch via the lightweight table_cls (small CNN, runs in <50 ms).
    kind = "wired"  # safe default if classifier is missing
    if parser.table_cls is not None and settings.parse_table_wireless:
        try:
            kind = parser.table_cls.classify(crop_bgr)
        except Exception:
            pass

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    html = ""
    if kind == "wired" and parser.table_wired is not None:
        try:
            wt = parser.table_wired.wired(rgb, ocr_result)
            html = wt.pred_html or ""
        except Exception:
            html = ""
    elif kind == "wireless" and parser.table_wireless is not None:
        try:
            html_code, _, _, _ = parser.table_wireless.predict(rgb, ocr_result)
            html = html_code or ""
        except Exception:
            html = ""
    else:
        # Classifier returned a kind whose model isn't loaded — fall back
        # to whichever IS loaded.
        if parser.table_wired is not None:
            try:
                wt = parser.table_wired.wired(rgb, ocr_result)
                html = wt.pred_html or ""
                kind = "wired"
            except Exception:
                html = ""
        elif parser.table_wireless is not None:
            try:
                html_code, _, _, _ = parser.table_wireless.predict(rgb, ocr_result)
                html = html_code or ""
                kind = "wireless"
            except Exception:
                html = ""

    return (kind, html)


def extract_table_both_parallel(
    parser, crop_bgr, precomputed_ocr=None,
) -> tuple[str, str]:
    """Run wired (UNet) + wireless (SLANet+) in parallel, pick the better HTML.

    Skips ``TableClassifier`` entirely — empirically the classifier confuses
    table types often enough that running both and picking via the
    ``_select_table_html`` heuristic (cell count vs OCR text coverage vs
    blank-cell rate) is more reliable.

    Both struct models accept the same ``precomputed_ocr`` shape, so we
    convert ``precomputed_ocr`` once and share the list. Both ONNX
    sessions release the GIL during ``session.run`` → ``ThreadPoolExecutor``
    gives true parallelism on CPU.

    Returns ``(kind, html)`` where ``kind`` ∈ ``{"wired", "wireless", "fallback"}``.
    """
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    from vn_parser.pipeline import _select_table_html
    from vn_parser.table_unet.main import escape_html

    if parser.table_wired is None and parser.table_wireless is None:
        return ("fallback", "")

    # Fast path: wireless disabled (PARSE_TABLE_WIRELESS=false). Skip the
    # thread pool and the _select_table_html heuristic — just run wired
    # synchronously.
    wireless_enabled = settings.parse_table_wireless and parser.table_wireless is not None

    # Normalise precomputed OCR to ``[[quad, escaped_text, score], ...]``.
    if precomputed_ocr is not None:
        ocr_result = [
            [item[0], escape_html(item[1][0]), item[1][1]]
            for item in precomputed_ocr
            if isinstance(item, (list, tuple)) and len(item) == 2
            and isinstance(item[1], tuple)
        ]
    else:
        try:
            ocr_raw = parser.ocr_engine.ocr(crop_bgr)[0]
        except Exception:
            ocr_raw = []
        ocr_result = [
            [item[0], escape_html(item[1][0]), item[1][1]]
            for item in (ocr_raw or [])
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], tuple)
        ]

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    def _run_wired() -> str:
        if parser.table_wired is None:
            return ""
        try:
            wt = parser.table_wired.wired(rgb, ocr_result)
            return wt.pred_html or ""
        except Exception:
            log.exception("table_wired failed")
            return ""

    def _run_wireless() -> str:
        if parser.table_wireless is None:
            return ""
        try:
            html_code, _, _, _ = parser.table_wireless.predict(rgb, ocr_result)
            return html_code or ""
        except Exception:
            log.exception("table_wireless failed")
            return ""

    if not wireless_enabled:
        wired_html = _run_wired()
        return ("wired" if wired_html else "fallback", wired_html)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_wired = pool.submit(_run_wired)
        f_wireless = pool.submit(_run_wireless)
        wired_html = f_wired.result()
        wireless_html = f_wireless.result()

    return _select_table_html(wired_html, wireless_html, ocr_result)


def _detect_lines(parser, crop_bgr, OCRDet) -> list:
    """DBNet text-line detection inside a layout block.

    Returns a list of ``(y_mean, x_min, line_crop_bgr)`` sorted in
    reading order (top-to-bottom, then left-to-right). Recognition
    happens later in a single page-level batch.
    """
    boxes, _ = parser.ocr_det.detect(crop_bgr)
    if not boxes:
        return []
    items = []
    for q in boxes:
        ys = q[:, 1].astype("float32")
        xs = q[:, 0].astype("float32")
        items.append((float(ys.mean()), float(xs.min()), q))
    items.sort(key=lambda t: (round(t[0] / 8.0) * 8.0, t[1]))

    out = []
    for y_mean, x_min, q in items:
        line_crop = OCRDet.crop_quad(crop_bgr, q)
        if line_crop.size > 0:
            out.append((y_mean, x_min, line_crop))
    return out


def _group_ocr_lines(line_text_pairs) -> str:
    """Group recognized line texts back into reading-order lines.

    DBNet returns one box per text line — we sorted those by y/x in
    ``_detect_lines``. Boxes whose y-centers are within ``LINE_THRESH``
    are merged horizontally (same physical line, multiple word boxes).
    """
    LINE_THRESH = 12.0
    lines: list[list[str]] = []
    cur_y: float | None = None
    for (y_mean, _x_min, _crop), text in line_text_pairs:
        text = (text or "").strip()
        if not text:
            continue
        if cur_y is None or abs(y_mean - cur_y) > LINE_THRESH:
            lines.append([text])
            cur_y = y_mean
        else:
            lines[-1].append(text)
    return "\n".join(" ".join(parts) for parts in lines).strip()


def _words_inside(words, x0_img, y0_img, x1_img, y1_img, scale) -> list[dict]:
    x0_pdf = x0_img / scale
    y0_pdf = y0_img / scale
    x1_pdf = x1_img / scale
    y1_pdf = y1_img / scale
    out = []
    for w in words:
        cx = (w["x0"] + w["x1"]) / 2
        cy = (w["top"] + w["bottom"]) / 2
        if x0_pdf <= cx <= x1_pdf and y0_pdf <= cy <= y1_pdf:
            out.append(w)
    return out


def _format_lines(contained: list[dict]) -> str:
    """Group pdfplumber words into lines by y-center, then join."""
    contained.sort(key=lambda w: (round((w["top"] + w["bottom"]) / 2 / 6), w["x0"]))
    lines: list[list[str]] = []
    cur_y: float | None = None
    for w in contained:
        y_mid = (w["top"] + w["bottom"]) / 2
        if cur_y is None or abs(y_mid - cur_y) > 6:
            lines.append([w["text"]])
            cur_y = y_mid
        else:
            lines[-1].append(w["text"])
    return "\n".join(" ".join(ln) for ln in lines).strip()


def _collect_images(image_dir: Path) -> list[ExtractedImage]:
    """Walk ``image_dir`` recursively and base64-encode every image.

    Used to ship cropped figures back through the ParseResult so the worker
    can upload them to S3 without re-walking the temp dir.
    """
    out: list[ExtractedImage] = []
    if not image_dir.exists():
        return out
    for p in sorted(image_dir.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower().lstrip(".")
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif",
        }.get(suffix)
        if not mime:
            continue
        out.append(ExtractedImage(
            name=str(p.relative_to(image_dir)),
            bytes_b64=base64.b64encode(p.read_bytes()).decode("ascii"),
            mime=mime,
        ))
    return out
