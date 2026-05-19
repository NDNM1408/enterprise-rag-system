"""End-to-end Vietnamese document parsing pipeline.

PDF / image -> per-page rendering -> orientation correction (optional)
            -> layout detection
            -> for each text-bearing block: OCR-det -> VietOCR rec
            -> assemble into a structured result (markdown + JSON).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from vn_parser.layout import LayoutDetector, visualize as visualize_layout
from vn_parser.ocr_det import OCRDet
from vn_parser.ocr_rec import VietOCRRec
from vn_parser.ocr_adapter import OCREngine
from vn_parser.orient_cls import OrientationClassifier
from vn_parser.table_cls import TableClassifier
from vn_parser.table_slanet import PaddleTableModel
from vn_parser.table_unet import UnetWiredTable
from vn_parser.vlm import VLMExtractor, NOT_EXTRACT_LIST

# Layout label sets — used by both PP-DocLayoutV2 (pipeline) and VLM backends.
# Pipeline labels (PP-DocLayoutV2 25-class scheme):
TEXT_LIKE_LABELS = {
    "abstract", "aside_text", "content", "doc_title", "figure_title",
    "footer", "footnote", "header", "number", "paragraph_title",
    "reference", "reference_content", "text", "vertical_text",
    "vision_footnote",
    # VLM-side text-bearing labels also need OCR fallback when VLM skipped them.
    "title", "page_number", "page_footnote", "ref_text",
    "table_caption", "image_caption", "table_footnote", "image_footnote",
    "code_caption", "list_item", "phonetic",
}
IMAGE_LIKE_LABELS = {
    # pipeline:
    "image", "chart", "header_image", "footer_image", "seal",
    # VLM:
    "image_block",
}
SKIP_LABELS = IMAGE_LIKE_LABELS  # backward-compat alias
TABLE_LABELS = {"table"}
FORMULA_LABELS = {
    "display_formula", "inline_formula", "formula_number",  # pipeline
    "equation", "equation_block",                           # VLM
}
CODE_LIKE_LABELS = {"code", "algorithm"}

DEFAULT_RENDER_DPI = 200

SEAL_CONTAINMENT_TOLERANCE_PX = 4


def _is_contained(inner: Sequence[int], outer: Sequence[int],
                  tol: int = SEAL_CONTAINMENT_TOLERANCE_PX) -> bool:
    """Return True if `inner` bbox sits entirely inside `outer` bbox (with slack)."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return (ix0 >= ox0 - tol and iy0 >= oy0 - tol
            and ix1 <= ox1 + tol and iy1 <= oy1 + tol)


def _count_table_cells(html_code: str) -> int:
    if not html_code:
        return 0
    low = html_code.lower()
    return low.count("<td") + low.count("<th")


def _select_table_html(
    wired_html: str, wireless_html: str, ocr_result: list
) -> Tuple[str, str]:
    """Port of MinerU's UnetTableModel.predict switching heuristic.

    Returns (kind, html). Compares both models on cell count, OCR text
    coverage, and blank-cell rate to pick the better result.
    """
    if not wired_html and not wireless_html:
        return "fallback", ""
    if not wired_html:
        return "wireless", wireless_html
    if not wireless_html:
        return "wired", wired_html

    wired_len = _count_table_cells(wired_html)
    wireless_len = _count_table_cells(wireless_html)
    gap = wireless_len - wired_len

    # Count OCR text presence in each rendered HTML
    wired_text = sum(1 for r in ocr_result if r[1] and r[1] in wired_html)
    wireless_text = sum(1 for r in ocr_result if r[1] and r[1] in wireless_html)

    try:
        from bs4 import BeautifulSoup
        wired_blank = sum(
            1 for c in BeautifulSoup(wired_html, "html.parser").find_all(["td", "th"])
            if not c.text.strip()
        )
        wireless_blank = sum(
            1 for c in BeautifulSoup(wireless_html, "html.parser").find_all(["td", "th"])
            if not c.text.strip()
        )
    except Exception:
        wired_blank, wireless_blank = 0, 0

    wireless_non_blank = wireless_len - wireless_blank
    wired_non_blank = wired_len - wired_blank

    switch = False
    if wireless_non_blank > wired_non_blank:
        scale = round(wired_non_blank ** 0.5)
        a = wired_non_blank + scale * 2
        b = scale * (scale + 2)
        if (wireless_non_blank + 3) >= max(a, b):
            switch = True

    pick_wireless = (
        switch
        or (0 <= gap <= 5 and wired_len <= round(wireless_len * 0.75))
        or (gap == 0 and wired_len <= 4)
        or (wired_text <= wireless_text * 0.6 and wireless_text >= 10)
    )
    if pick_wireless:
        return "wireless", wireless_html
    return "wired", wired_html


def _drop_inside_seal(blocks: List[Dict]) -> List[Dict]:
    """Remove non-seal blocks whose bbox is fully contained in a seal block."""
    seals = [b["bbox"] for b in blocks if b.get("label") == "seal"]
    if not seals:
        return blocks
    kept = []
    for b in blocks:
        if b.get("label") != "seal" and any(
            _is_contained(b["bbox"], s) for s in seals
        ):
            continue
        kept.append(b)
    # Re-index reading order so it stays contiguous (1..N) after dropping.
    for new_idx, b in enumerate(kept, start=1):
        b["index"] = new_idx
    return kept


@dataclass
class Block:
    cls_id: int
    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # xyxy
    index: int
    text: str = ""
    image_path: Optional[str] = None  # relative path to saved crop, if image-like
    extra: dict = field(default_factory=dict)


@dataclass
class PageResult:
    page_index: int          # 0-based
    width: int
    height: int
    angle: int = 0
    blocks: List[Block] = field(default_factory=list)


class VNDocParser:
    def __init__(
        self,
        models_dir: Union[str, Path] = "models_onnx",
        device: str = "cpu",
        enable_orientation: bool = True,
        layout_conf: float = 0.5,
        det_box_thresh: float = 0.5,
        det_unclip_ratio: float = 1.6,
        vietocr_config: str = "vgg_transformer",
        vietocr_weights: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
        drop_inside_seal: bool = True,
        enable_vlm: bool = False,
        vlm_model_path: Optional[Union[str, Path]] = None,
        vlm_dtype: str = "float32",
    ):
        self.drop_inside_seal = drop_inside_seal
        self.enable_vlm = enable_vlm
        self._vlm_model_path = vlm_model_path
        self._vlm_dtype = vlm_dtype
        models = Path(models_dir)
        self.layout = LayoutDetector(
            models / "layout.onnx", conf=layout_conf, providers=providers
        )
        self.ocr_det = OCRDet(
            models / "ocr_det.onnx",
            box_thresh=det_box_thresh,
            unclip_ratio=det_unclip_ratio,
            providers=providers,
        )
        self.ocr_rec = VietOCRRec(
            config_name=vietocr_config, device=device, weights=vietocr_weights
        )
        self.orient: Optional[OrientationClassifier] = None
        if enable_orientation and (models / "orient_cls.onnx").exists():
            self.orient = OrientationClassifier(
                models / "orient_cls.onnx", providers=providers
            )

        # Combined OCR engine (DBNet det ONNX + VietOCR rec) — used by tables.
        self.ocr_engine = OCREngine(self.ocr_det, self.ocr_rec)

        # Table pipeline: TabCls + UNet (wired) + SLANet+ (wireless).
        # Same dual-model + heuristic-switch design as MinerU's UnetTableModel.
        self.table_cls: Optional[TableClassifier] = None
        self.table_wired: Optional[UnetWiredTable] = None
        self.table_wireless: Optional[PaddleTableModel] = None
        if (models / "table_cls.onnx").exists():
            self.table_cls = TableClassifier(models / "table_cls.onnx", providers=providers)
        if (models / "table_unet.onnx").exists():
            self.table_wired = UnetWiredTable(
                str(models / "table_unet.onnx"), self.ocr_engine
            )
        if (models / "table_slanet.onnx").exists():
            self.table_wireless = PaddleTableModel(
                self.ocr_engine, str(models / "table_slanet.onnx")
            )

        # Optional VLM (MinerU2.5-Pro-2604-1.2B). Heavy: ~1.2B params CPU.
        self.vlm: Optional[VLMExtractor] = None
        if self.enable_vlm:
            mp = Path(self._vlm_model_path) if self._vlm_model_path else (Path(__file__).resolve().parent.parent / "models" / "vlm")
            if mp.exists():
                self.vlm = VLMExtractor(model_path=mp, dtype=self._vlm_dtype, use_tqdm=False)
            else:
                raise FileNotFoundError(
                    f"VLM enabled but model dir not found: {mp}. "
                    "Either point vlm_model_path at the MinerU2.5-Pro-2604-1.2B folder, "
                    "or set enable_vlm=False."
                )

    # ---- Image / PDF loading -----------------------------------------------
    @staticmethod
    def _render_pdf(pdf_path: Path, dpi: int) -> List[Image.Image]:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            pages = []
            for page in doc:
                bitmap = page.render(scale=dpi / 72.0)
                pages.append(bitmap.to_pil().convert("RGB"))
                bitmap.close()
                page.close()
            return pages
        finally:
            doc.close()

    @classmethod
    def load_pages(cls, src: Union[str, Path], dpi: int = DEFAULT_RENDER_DPI) -> List[Image.Image]:
        p = Path(src)
        ext = p.suffix.lower()
        if ext == ".pdf":
            return cls._render_pdf(p, dpi)
        return [Image.open(p).convert("RGB")]

    # ---- Per-page parsing --------------------------------------------------
    def parse_page(
        self,
        image: Image.Image,
        page_index: int = 0,
        image_dir: Optional[Path] = None,
        image_subdir: str = "images",
    ) -> PageResult:
        """Parse a single rendered page.

        If `image_dir` is given, image-like blocks (image/chart/header_image/...
        and seal) are cropped and saved under `image_dir/<image_subdir>/`,
        and `Block.image_path` is set to a markdown-friendly relative path.
        """
        angle = 0
        if self.orient is not None:
            image, angle = self.orient.correct(image)
        w, h = image.size
        page = PageResult(page_index=page_index, width=w, height=h, angle=angle)
        if self.vlm is not None:
            vlm_blocks = self.vlm.extract_page(image, not_extract_list=NOT_EXTRACT_LIST)
            # Adapt VLM blocks to LayoutDetector schema for the rest of the loop.
            layout_blocks = []
            for i, b in enumerate(vlm_blocks, start=1):
                layout_blocks.append({
                    "cls_id": -1,
                    "label": b["type"],
                    "score": 1.0,
                    "bbox": b["bbox"],
                    "index": i,
                    "_vlm_content": b.get("content"),
                    "_vlm_angle": b.get("angle"),
                    "_vlm_merge_prev": b.get("merge_prev", False),
                })
        else:
            layout_blocks = self.layout.predict(image)
        if self.drop_inside_seal:
            layout_blocks = _drop_inside_seal(layout_blocks)

        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        save_dir: Optional[Path] = None
        if image_dir is not None:
            save_dir = Path(image_dir) / image_subdir
            save_dir.mkdir(parents=True, exist_ok=True)

        for r in layout_blocks:
            block = Block(
                cls_id=r["cls_id"],
                label=r["label"],
                score=r["score"],
                bbox=tuple(r["bbox"]),
                index=r["index"],
            )
            vlm_content = r.get("_vlm_content")
            if vlm_content:
                block.extra["vlm"] = True
            if block.label in IMAGE_LIKE_LABELS:
                if save_dir is not None:
                    x0, y0, x1, y1 = block.bbox
                    crop = bgr[y0:y1, x0:x1]
                    if crop.size > 0:
                        fname = (
                            f"page{page_index + 1:04d}_block{block.index:03d}"
                            f"_{block.label}.png"
                        )
                        cv2.imwrite(str(save_dir / fname), crop)
                        block.image_path = f"{image_subdir}/{fname}"
                page.blocks.append(block)
                continue
            x0, y0, x1, y1 = block.bbox
            crop = bgr[y0:y1, x0:x1]
            if crop.size == 0:
                page.blocks.append(block)
                continue

            if block.label in TABLE_LABELS:
                # Prefer VLM HTML if VLM produced one; otherwise fall back to
                # the wired/wireless ONNX pipeline.
                if vlm_content and "<table" in vlm_content.lower():
                    block.text = vlm_content
                    block.extra["table_kind"] = "vlm"
                else:
                    block.text = self._extract_table(crop, block)
            elif block.label in FORMULA_LABELS:
                # VLM yields LaTeX directly; pipeline backend OCR-concats.
                block.text = vlm_content if vlm_content else self._read_text(crop)
            elif block.label in CODE_LIKE_LABELS:
                block.text = vlm_content if vlm_content else self._read_text(crop)
            elif block.label in TEXT_LIKE_LABELS:
                # text-bearing block — prefer VietOCR for Vietnamese accuracy
                # even if VLM happened to produce content (NOT_EXTRACT_LIST means
                # VLM should have skipped extraction here, but be defensive).
                block.text = self._read_text(crop)
            else:
                block.text = vlm_content if vlm_content else self._read_text(crop)
            page.blocks.append(block)
        return page

    def _extract_table(self, crop_bgr: np.ndarray, block: "Block") -> str:
        """Run BOTH wired (UNet) and wireless (SLANet+) and pick the better
        result with the same heuristic MinerU uses (UnetTableModel.predict).
        """
        if self.table_wired is None and self.table_wireless is None:
            block.extra["table_kind"] = "fallback"
            return self._read_text(crop_bgr)

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # Compute OCR result once (det+rec) — both models reuse it.
        try:
            ocr_raw = self.ocr_engine.ocr(crop_bgr)[0]
        except Exception as e:
            block.extra["table_ocr_error"] = str(e)
            ocr_raw = []
        from vn_parser.table_unet.main import escape_html
        ocr_result = [
            [item[0], escape_html(item[1][0]), item[1][1]]
            for item in (ocr_raw or [])
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], tuple)
        ]

        wired_html = ""
        if self.table_wired is not None:
            try:
                # Run wired with the shared ocr_result.
                wt = self.table_wired.wired(rgb, ocr_result)
                wired_html = wt.pred_html or ""
            except Exception as e:
                block.extra["table_wired_error"] = str(e)

        wireless_html = ""
        if self.table_wireless is not None:
            try:
                html_code, _, _, _ = self.table_wireless.predict(rgb, ocr_result)
                wireless_html = html_code or ""
            except Exception as e:
                block.extra["table_wireless_error"] = str(e)

        chosen = _select_table_html(wired_html, wireless_html, ocr_result)
        block.extra["table_kind"] = chosen[0]
        return chosen[1] or self._read_text(crop_bgr)

    def _read_text(self, crop_bgr: np.ndarray) -> str:
        boxes, _ = self.ocr_det.detect(crop_bgr)
        if not boxes:
            return ""
        # Sort top-to-bottom, then left-to-right
        items: List[Tuple[float, float, np.ndarray]] = []
        for q in boxes:
            ys = q[:, 1].astype(np.float32)
            xs = q[:, 0].astype(np.float32)
            items.append((ys.mean(), xs.min(), q))
        items.sort(key=lambda x: (round(x[0] / 8.0) * 8.0, x[1]))

        lines: List[str] = []
        cur_line: List[str] = []
        cur_y: Optional[float] = None
        line_height_thresh = 12.0
        for y_mean, _x_min, q in items:
            crop = OCRDet.crop_quad(crop_bgr, q)
            if crop.size == 0:
                continue
            text = self.ocr_rec.recognize(crop).strip()
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
        return "\n".join(lines)

    # ---- High-level helpers -----------------------------------------------
    def parse(
        self,
        src: Union[str, Path],
        dpi: int = DEFAULT_RENDER_DPI,
        image_dir: Optional[Union[str, Path]] = None,
        image_subdir: str = "images",
    ) -> List[PageResult]:
        pages = self.load_pages(src, dpi=dpi)
        out_dir = Path(image_dir) if image_dir else None
        return [
            self.parse_page(p, page_index=i,
                            image_dir=out_dir, image_subdir=image_subdir)
            for i, p in enumerate(pages)
        ]

    @staticmethod
    def to_markdown(results: Sequence[PageResult]) -> str:
        out: List[str] = []
        for page in results:
            out.append(f"<!-- page {page.page_index + 1} (rotated {page.angle}°) -->")
            for b in page.blocks:
                if b.label == "doc_title":
                    if b.text:
                        out.append(f"# {b.text}")
                elif b.label == "paragraph_title":
                    if b.text:
                        out.append(f"## {b.text}")
                elif b.label == "figure_title":
                    if b.text:
                        out.append(f"**{b.text}**")
                elif b.label in IMAGE_LIKE_LABELS:
                    if b.image_path:
                        out.append(f"![{b.label}]({b.image_path})")
                    else:
                        out.append(f"_[{b.label} {b.bbox}]_")
                elif b.label in TABLE_LABELS:
                    text_is_html = bool(b.text) and "<table" in b.text.lower()
                    if text_is_html:
                        out.append(b.text)
                    else:
                        if b.image_path:
                            out.append(f"![table]({b.image_path})")
                        if b.text:
                            out.append(b.text)
                elif b.label in FORMULA_LABELS:
                    if b.text:
                        out.append(f"$$\n{b.text}\n$$")
                else:
                    if b.text:
                        out.append(b.text)
            out.append("")
        return "\n\n".join(out)

    @staticmethod
    def to_json(results: Sequence[PageResult]) -> dict:
        return {
            "pages": [
                {
                    "page_index": p.page_index,
                    "width": p.width,
                    "height": p.height,
                    "angle": p.angle,
                    "blocks": [
                        {
                            "cls_id": b.cls_id,
                            "label": b.label,
                            "score": b.score,
                            "bbox": list(b.bbox),
                            "index": b.index,
                            "text": b.text,
                            **({"image_path": b.image_path} if b.image_path else {}),
                            **({"extra": b.extra} if b.extra else {}),
                        }
                        for b in p.blocks
                    ],
                }
                for p in results
            ]
        }
