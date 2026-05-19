"""PP-DocLayoutV2 layout detector running on ONNX Runtime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

LAYOUT_LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer", "footer_image",
    "footnote", "formula_number", "header", "header_image", "image",
    "inline_formula", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote",
]

DEFAULT_CLASS_THRESHOLDS = [
    0.5, 0.5, 0.5, 0.5, 0.5,    # abstract..content
    0.4, 0.4, 0.5, 0.5, 0.5,    # display_formula..footer_image
    0.5, 0.5, 0.5, 0.5, 0.5,    # footnote..image
    0.4, 0.5, 0.4, 0.5, 0.5,    # inline_formula..reference_content
    0.45, 0.5, 0.4, 0.4, 0.5,   # seal..vision_footnote
]

DEFAULT_RESCALE_FACTOR = 1.0 / 255.0
DEFAULT_INPUT_SIZE = (800, 800)  # (W, H)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clip to prevent overflow in exp for very negative values.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _get_order_seqs(order_logits: np.ndarray) -> np.ndarray:
    """Pure-numpy port of PPDocLayoutV2LayoutModel._get_order_seqs.

    order_logits: (B, N, N).  Returns (B, N) integer sequence: position rank for each query.
    """
    order_scores = _sigmoid(order_logits)
    B, N, _ = order_scores.shape

    upper = np.triu(order_scores, k=1).sum(axis=1)
    lower_inv = np.tril(1.0 - np.transpose(order_scores, (0, 2, 1)), k=-1).sum(axis=1)
    order_votes = upper + lower_inv
    order_pointers = np.argsort(order_votes, axis=1)

    order_seq = np.empty_like(order_pointers)
    ranks = np.broadcast_to(np.arange(N, dtype=order_pointers.dtype), (B, N))
    rows = np.broadcast_to(np.arange(B)[:, None], (B, N))
    order_seq[rows, order_pointers] = ranks
    return order_seq


class LayoutDetector:
    """Run PP-DocLayoutV2 ONNX and return layout blocks per page."""

    def __init__(
        self,
        onnx_path: Union[str, Path],
        conf: float = 0.5,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        rescale_factor: float = DEFAULT_RESCALE_FACTOR,
        class_thresholds: Optional[Sequence[float]] = None,
        providers: Optional[Sequence[str]] = None,
    ):
        self.onnx_path = str(onnx_path)
        self.conf = conf
        self.input_size = input_size
        self.rescale_factor = rescale_factor
        self.class_thresholds = np.asarray(
            class_thresholds or DEFAULT_CLASS_THRESHOLDS, dtype=np.float32
        )
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_options,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Try to read preprocess_config.json next to the ONNX file (optional override).
        cfg_path = Path(self.onnx_path).parent / "layout_preprocess.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            size = cfg.get("size", {})
            self.input_size = (int(size.get("width", input_size[0])),
                               int(size.get("height", input_size[1])))
            self.rescale_factor = float(cfg.get("rescale_factor", rescale_factor))

    def _preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        pil = pil.convert("RGB")
        # Resize to (W, H) with bicubic
        resized = pil.resize(self.input_size, Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32)  # H,W,3
        arr = arr.transpose(2, 0, 1) * self.rescale_factor  # 3,H,W
        return arr

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> List[Dict]:
        return self.batch_predict([image])[0]

    def batch_predict(
        self,
        images: Sequence[Union[np.ndarray, Image.Image]],
        batch_size: int = 1,
    ) -> List[List[Dict]]:
        if not images:
            return []
        results: List[List[Dict]] = []
        # target_sizes: (h, w) per image -- needed for rescaling boxes back
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            x = np.stack([self._preprocess(img) for img in chunk], axis=0)
            target_sizes = [_image_size(img) for img in chunk]
            outs = self.session.run(None, {self.input_name: x})
            logits, pred_boxes, order_logits = outs[0], outs[1], outs[2]
            for i, ts in enumerate(target_sizes):
                results.append(self._postprocess(
                    logits[i:i+1], pred_boxes[i:i+1], order_logits[i:i+1], [ts]
                ))
        return results

    def _postprocess(
        self,
        logits: np.ndarray,           # (1, N, C)
        pred_boxes: np.ndarray,       # (1, N, 4)
        order_logits: np.ndarray,     # (1, N, N)
        target_sizes: Sequence[Tuple[int, int]],  # [(h, w)]
    ) -> List[Dict]:
        boxes = pred_boxes
        # Convert center-size to xyxy
        cxcy, wh = boxes[..., :2], boxes[..., 2:]
        boxes_xyxy = np.concatenate([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)
        h, w = target_sizes[0]
        scale = np.array([w, h, w, h], dtype=np.float32)
        boxes_scaled = boxes_xyxy * scale[None, None, :]

        order_seqs = _get_order_seqs(order_logits)  # (1, N)

        scores = _sigmoid(logits)   # (1, N, C)
        N = scores.shape[1]
        flat = scores.reshape(scores.shape[0], -1)  # (1, N*C)
        topk_idx = np.argsort(-flat, axis=-1)[:, :N]
        topk_score = np.take_along_axis(flat, topk_idx, axis=-1)
        labels = topk_idx % scores.shape[2]
        query_idx = topk_idx // scores.shape[2]

        # Gather boxes / order by query_idx
        gathered_boxes = np.take_along_axis(
            boxes_scaled, query_idx[..., None].repeat(4, axis=-1), axis=1
        )
        gathered_order = np.take_along_axis(order_seqs, query_idx, axis=1)

        score = topk_score[0]
        label = labels[0]
        box = gathered_boxes[0]
        order_seq = gathered_order[0]
        keep = score >= self.conf
        order_seq = order_seq[keep]
        sort_idx = np.argsort(order_seq)

        score_kept = score[keep][sort_idx]
        label_kept = label[keep][sort_idx]
        box_kept = box[keep][sort_idx]

        out = []
        for i, (s, lid, b) in enumerate(zip(score_kept, label_kept, box_kept), start=1):
            cls_id = int(lid)
            xmin, ymin, xmax, ymax = [float(v) for v in b.tolist()]
            xmin = max(0, int(round(xmin)))
            ymin = max(0, int(round(ymin)))
            xmax = min(w, int(round(xmax)))
            ymax = min(h, int(round(ymax)))
            if xmax <= xmin or ymax <= ymin:
                continue
            out.append({
                "cls_id": cls_id,
                "label": LAYOUT_LABELS[cls_id] if 0 <= cls_id < len(LAYOUT_LABELS) else str(cls_id),
                "score": round(float(s), 4),
                "bbox": [xmin, ymin, xmax, ymax],
                "index": i,
            })
        return out


def _image_size(img: Union[np.ndarray, Image.Image]) -> Tuple[int, int]:
    """Return (height, width)."""
    if isinstance(img, np.ndarray):
        return img.shape[0], img.shape[1]
    if isinstance(img, Image.Image):
        return img.size[1], img.size[0]
    raise TypeError(f"Unsupported image type: {type(img)}")


def visualize(
    image: Union[np.ndarray, Image.Image],
    results: Sequence[Dict],
) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()
    for r in results:
        x0, y0, x1, y1 = r["bbox"]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 2)
        text = f"{r['label']} {r['score']:.2f} #{r['index']}"
        cv2.putText(img, text, (x0, max(0, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 200), 1, cv2.LINE_AA)
    return img
