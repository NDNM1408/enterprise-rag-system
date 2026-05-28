"""PaddleOCR DBNet text detector running on ONNX Runtime."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon


class _DBPostProcess:
    """Pure-numpy DBPostProcess (port from the PaddleOCR db_postprocess.py)."""

    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000,
                 unclip_ratio=1.6, score_mode="fast", use_dilation=False):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.dilation_kernel = (
            np.array([[1, 1], [1, 1]], dtype=np.uint8) if use_dilation else None
        )

    def _unclip(self, box: np.ndarray) -> List[np.ndarray]:
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / max(poly.length, 1e-6)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    @staticmethod
    def _get_mini_boxes(contour) -> Tuple[List[List[float]], float]:
        rect = cv2.minAreaRect(contour)
        pts = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])
        if pts[1][1] > pts[0][1]:
            i1, i4 = 0, 1
        else:
            i1, i4 = 1, 0
        if pts[3][1] > pts[2][1]:
            i2, i3 = 2, 3
        else:
            i2, i3 = 3, 2
        box = [pts[i1], pts[i2], pts[i3], pts[i4]]
        return box, min(rect[1])

    @staticmethod
    def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        b = box.copy()
        xmin = int(np.clip(np.floor(b[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(b[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(b[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(b[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        b[:, 0] = b[:, 0] - xmin
        b[:, 1] = b[:, 1] - ymin
        cv2.fillPoly(mask, b.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def boxes_from_bitmap(self, pred: np.ndarray, bitmap: np.ndarray,
                          dest_w: int, dest_h: int) -> Tuple[List[np.ndarray], List[float]]:
        h, w = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            points, sside = self._get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self._box_score_fast(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue
            expanded = self._unclip(points)
            if len(expanded) != 1:
                continue
            box = np.array(expanded[0]).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)
            boxes.append(box.astype(np.int32))
            scores.append(float(score))
        return boxes, scores

    def __call__(self, prob_map: np.ndarray, src_h: int, src_w: int) -> Tuple[List[np.ndarray], List[float]]:
        # prob_map: (1, 1, H, W)  ->  (H, W)
        pm = prob_map[0, 0]
        seg = pm > self.thresh
        mask = seg.astype(np.uint8)
        if self.dilation_kernel is not None:
            mask = cv2.dilate(mask, self.dilation_kernel)
        return self.boxes_from_bitmap(pm, mask, src_w, src_h)


class OCRDet:
    """DBNet text detector (paddleocr_torch ch_PP-OCRv5_det).

    Input: BGR image (np.uint8 HxWx3).
    Output: list of 4-point quad boxes (int) + list of scores.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_path: Union[str, Path],
        limit_side_len: int = 960,
        limit_type: str = "max",
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.6,
        providers: Optional[Sequence[str]] = None,
    ):
        self.onnx_path = str(onnx_path)
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_options,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.post = _DBPostProcess(
            thresh=thresh, box_thresh=box_thresh, unclip_ratio=unclip_ratio
        )

    def _resize(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize so that the limit side meets limit_type, both dims multiple of 32."""
        h, w = img_bgr.shape[:2]
        if self.limit_type == "max":
            ratio = self.limit_side_len / max(h, w) if max(h, w) > self.limit_side_len else 1.0
        else:  # "min"
            ratio = self.limit_side_len / min(h, w) if min(h, w) < self.limit_side_len else 1.0
        new_h = max(int(round(h * ratio / 32) * 32), 32)
        new_w = max(int(round(w * ratio / 32) * 32), 32)
        resized = cv2.resize(img_bgr, (new_w, new_h))
        return resized, (h, w)

    def _normalize(self, img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.MEAN) / self.STD
        chw = rgb.transpose(2, 0, 1)
        return chw[None, ...].astype(np.float32)

    def detect(self, img_bgr: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        if img_bgr is None or img_bgr.size == 0:
            return [], []
        resized, (src_h, src_w) = self._resize(img_bgr)
        x = self._normalize(resized)
        outs = self.session.run(None, {self.input_name: x})
        prob_map = outs[0]
        boxes, scores = self.post(prob_map, src_h, src_w)
        return boxes, scores

    @staticmethod
    def crop_quad(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Perspective-transform a quad box into an axis-aligned crop."""
        pts = np.array(quad, dtype=np.float32).reshape(4, 2)
        w0 = np.linalg.norm(pts[0] - pts[1])
        w1 = np.linalg.norm(pts[2] - pts[3])
        h0 = np.linalg.norm(pts[0] - pts[3])
        h1 = np.linalg.norm(pts[1] - pts[2])
        out_w = int(max(w0, w1))
        out_h = int(max(h0, h1))
        if out_w < 2 or out_h < 2:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
        if out_h > out_w * 1.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        return crop
