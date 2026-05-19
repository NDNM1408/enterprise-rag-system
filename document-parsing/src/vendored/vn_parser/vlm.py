"""VLM-based hybrid layout + content extractor (MinerU2.5-Pro-2604-1.2B).

Runs `mineru_vl_utils.MinerUClient.batch_two_step_extract` end-to-end:
  step 1: layout detection (bbox + label per block)
  step 2: content extraction for non-text-like blocks (table HTML, image
          caption, formula LaTeX, code, etc.)

Text-bearing blocks listed in `NOT_EXTRACT_LIST` (mirroring MinerU's
NotExtractType) are skipped at step 2 — the caller fills text via VietOCR.

Output blocks normalized to:
  {
    "type":   str,            # VLM BlockType (e.g. "text", "table", "image")
    "bbox":   [x0,y0,x1,y1],  # absolute pixel coords (xyxy)
    "content": str | None,    # VLM-extracted content (None for skipped types)
    "angle":   None | int,    # 0/90/180/270
  }
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# Mirrors mineru-source/.../enum_class.py::NotExtractType — text blocks the
# VLM should NOT extract (we OCR them with VietOCR for Vietnamese accuracy).
NOT_EXTRACT_LIST: List[str] = [
    "text", "title", "header", "footer", "page_number", "page_footnote",
    "ref_text", "table_caption", "image_caption", "table_footnote",
    "image_footnote", "code_caption",
]


def _import_loader():
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    return torch, Qwen2VLForConditionalGeneration, AutoProcessor


class VLMExtractor:
    """Wraps MinerU2.5 VLM with mineru_vl_utils.MinerUClient.

    On Intel Mac CPU we force float32 + device_map='cpu' (BFloat16 isn't
    supported on MPS and the auto device-map otherwise picks MPS).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cpu",
        dtype: str = "float32",  # "float32" or "float16"
        use_tqdm: bool = False,
    ):
        torch, Qwen2VLForConditionalGeneration, AutoProcessor = _import_loader()
        from mineru_vl_utils import MinerUClient

        torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
        t0 = time.time()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=device,
        )
        # use_fast=False -> legacy numpy processor; fast variant calls
        # torch.compiler.is_compiling() which only exists from torch 2.4+.
        processor = AutoProcessor.from_pretrained(str(model_path), use_fast=False)
        self.load_seconds = time.time() - t0
        self.client = MinerUClient(
            backend="transformers",
            model=model,
            processor=processor,
            use_tqdm=use_tqdm,
        )

    @staticmethod
    def _denormalize_bbox(bbox: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
        x0 = int(round(bbox[0] * width))
        y0 = int(round(bbox[1] * height))
        x1 = int(round(bbox[2] * width))
        y1 = int(round(bbox[3] * height))
        return (max(0, x0), max(0, y0), min(width, x1), min(height, y1))

    def extract_page(
        self,
        image: Image.Image,
        not_extract_list: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        """Run VLM on a single PIL page image. Returns blocks with absolute
        pixel bboxes and (optional) content."""
        nle = list(not_extract_list) if not_extract_list is not None else NOT_EXTRACT_LIST
        result = self.client.batch_two_step_extract(images=[image], not_extract_list=nle)
        blocks_raw = result[0] if result else []
        w, h = image.size
        out: List[dict] = []
        for i, b in enumerate(blocks_raw, start=1):
            bx, by, bx2, by2 = self._denormalize_bbox(b["bbox"], w, h)
            if bx2 <= bx or by2 <= by:
                continue
            out.append({
                "type": b["type"],
                "bbox": [bx, by, bx2, by2],
                "content": b.get("content"),
                "angle": b.get("angle"),
                "index": i,
                "merge_prev": b.get("merge_prev", False),
            })
        return out
