"""OCR engine adapter — wraps OCRDet (DBNet ONNX) + PaddleOCRRec (ONNX CTC)
to expose the PaddleOCR ocr() interface.

Contract (mirrors the PaddleOCR ocr() contract):
  ocr(img, det=True,  rec=True ) -> [[(box4x2, (text, score)), ...]]
  ocr(img, det=True,  rec=False) -> [[box4x2, ...]]
  ocr(list_of_imgs, det=False, rec=True ) -> [[(text, score), ...]]
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from vn_parser.ocr_det import OCRDet
from vn_parser.ocr_rec_onnx import PaddleOCRRec


class OCREngine:
    """Adapter that exposes the PaddleOCR ocr() signature."""

    def __init__(self, ocr_det: OCRDet, ocr_rec: PaddleOCRRec):
        self.det = ocr_det
        self.rec = ocr_rec
        # Some downstream code reads .text_detector — keep a stub for compatibility.
        self.text_detector = _DetWrapper(ocr_det)

    @staticmethod
    def _to_bgr(img) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def ocr(
        self,
        img,
        det: bool = True,
        rec: bool = True,
        mfd_res=None,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
    ) -> List:
        if isinstance(img, list) and det:
            raise ValueError("When input is a list of images, det must be False")

        # Det=True path: single image input.
        # Detection runs once, then VietOCR recognition is BATCHED across
        # every detected line — one torch.predict_batch call instead of
        # N×predict. On GPU this is the difference between ~5 s and ~0.2 s
        # for a typical 60-cell table.
        if det and rec:
            bgr = self._to_bgr(img)
            boxes, _scores = self.det.detect(bgr)
            results: List = []
            quads: List = []
            crops: List = []
            empty_idx: set = set()
            for i, q in enumerate(boxes):
                crop = OCRDet.crop_quad(bgr, q)
                quads.append(np.asarray(q, dtype=np.float32))
                if crop.size == 0:
                    empty_idx.add(i)
                    crops.append(None)
                else:
                    crops.append(crop)
            non_empty = [c for c in crops if c is not None]
            texts_iter: List[str] = []
            if non_empty:
                texts_iter = self.rec.recognize_batch(non_empty)
            ti = iter(texts_iter)
            for i, (q, c) in enumerate(zip(quads, crops)):
                if i in empty_idx or c is None:
                    text, score = "", 0.0
                else:
                    text = next(ti, "")
                    score = 1.0
                results.append([q, (text, float(score))])
            return [results]

        if det and not rec:
            bgr = self._to_bgr(img)
            boxes, _ = self.det.detect(bgr)
            return [[np.asarray(q, dtype=np.float32) for q in boxes]]

        if not det and rec:
            crops = img if isinstance(img, list) else [img]
            valid: List = [
                c for c in crops
                if c is not None and not (hasattr(c, "size") and c.size == 0)
            ]
            valid_texts = self.rec.recognize_batch(valid) if valid else []
            ti = iter(valid_texts)
            results = []
            for c in crops:
                if c is None or (hasattr(c, "size") and c.size == 0):
                    results.append(("", 0.0))
                else:
                    results.append((next(ti, ""), 1.0))
            return [results]

        # det=False, rec=False is meaningless
        return [[]]


class _DetWrapper:
    """Mimics the `ocr_engine.text_detector(img)` interface returning (dt_boxes, elapse)
    and `text_detector.batch_predict(...)` for the batched OCR-det path.
    """
    def __init__(self, ocr_det: OCRDet):
        self.ocr_det = ocr_det

    def __call__(self, img):
        boxes, _ = self.ocr_det.detect(img)
        return ([np.asarray(q, dtype=np.float32) for q in boxes], 0.0)

    def batch_predict(self, imgs: Sequence[np.ndarray], batch_size: int = 1):
        out = []
        for im in imgs:
            boxes, scores = self.ocr_det.detect(im)
            out.append((
                [np.asarray(q, dtype=np.float32) for q in boxes],
                scores,
            ))
        return out
