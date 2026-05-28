"""PaddleOCR recognition via ONNX + CTC greedy decode.

Drop-in replacement for the torch-based ``VietOCRRec`` — same public
contract (``recognize`` / ``recognize_batch``) so ``OCREngine`` and the
pipeline use it unchanged. Pure onnxruntime, no torch.

Preprocessing mirrors PaddleOCR's ``RecResizeImg``: keep aspect ratio at a
fixed height, normalise to (x/255 - 0.5)/0.5, right-pad zeros to the model
width. CTC decode drops blanks (index 0) and consecutive duplicates.

The model + char dictionary are mounted with the other ONNX weights
(``PARSER_MODELS_DIR``). Char dict convention (PaddleOCR): index 0 = blank,
1..N = dict chars, with a trailing space char when ``use_space=True``.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

log = logging.getLogger(__name__)


def _load_char_dict(path: str, use_space: bool) -> List[str]:
    """PaddleOCR CTC charset: ['blank'] + dict chars (+ ' ' if use_space)."""
    chars: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chars.append(line.rstrip("\n").rstrip("\r"))
    if use_space:
        chars.append(" ")
    return ["blank"] + chars


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class PaddleOCRRec:
    """ONNX CTC text recognizer with the VietOCRRec contract.

    Args:
        model_path:     path to the rec ONNX model.
        char_dict_path: path to the PaddleOCR char dictionary.
        providers:      onnxruntime providers list (CPU/CUDA), from
                        ``resolve_onnx_providers``.
        img_shape:      (C, H, W) — must match training (default 3x48x960).
        use_space:      append a space char to the dictionary (PaddleOCR).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        char_dict_path: Union[str, Path],
        providers: Optional[Sequence] = None,
        img_shape: Tuple[int, int, int] = (3, 48, 960),
        use_space: bool = True,
    ):
        self.img_shape = img_shape
        self.session = ort.InferenceSession(
            str(model_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.charset = _load_char_dict(str(char_dict_path), use_space)
        log.info(
            "PaddleOCRRec loaded: %s (%d chars + blank, providers=%s)",
            Path(model_path).name, len(self.charset) - 1,
            [p if isinstance(p, str) else p[0] for p in self.session.get_providers()],
        )

    # ------------------------------------------------------------------
    # Public API (matches VietOCRRec)
    # ------------------------------------------------------------------

    def recognize(self, img: Union[np.ndarray, Image.Image]) -> str:
        x = self._preprocess(self._to_bgr(img))
        logits = self.session.run(None, {self.input_name: x})[0]
        return self._ctc_decode(logits)[0][0]

    def recognize_batch(
        self,
        imgs: List[Union[np.ndarray, Image.Image]],
    ) -> List[str]:
        """Batched recognition — every crop is padded to the fixed model
        width, so the whole list stacks into one ONNX run."""
        if not imgs:
            return []
        batch = np.concatenate(
            [self._preprocess(self._to_bgr(im)) for im in imgs], axis=0
        )
        logits = self.session.run(None, {self.input_name: batch})[0]
        return [text for text, _conf in self._ctc_decode(logits)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bgr(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Pipeline crops are BGR numpy (OpenCV). PIL inputs → BGR numpy."""
        if isinstance(img, Image.Image):
            rgb = np.asarray(img.convert("RGB"))
            return rgb[..., ::-1].copy()
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        raise TypeError(f"Unsupported image type: {type(img)}")

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """RecResizeImg: aspect-preserving resize to height H, pad to W."""
        imgC, imgH, imgW = self.img_shape
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((1, imgC, imgH, imgW), dtype=np.float32)
        ratio = w / float(h)
        resized_w = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))
        resized_w = max(1, resized_w)

        resized = cv2.resize(img_bgr, (resized_w, imgH))
        resized = resized.astype("float32").transpose(2, 0, 1) / 255.0
        resized = (resized - 0.5) / 0.5

        padded = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        return padded[None]  # [1, C, H, W]

    def _ctc_decode(self, logits: np.ndarray) -> List[Tuple[str, float]]:
        """Greedy CTC: drop blanks (idx 0) and consecutive duplicates."""
        probs = _softmax(logits, axis=-1)
        preds_idx = probs.argmax(axis=-1)   # [B, T]
        preds_prob = probs.max(axis=-1)     # [B, T]

        out: List[Tuple[str, float]] = []
        for b in range(preds_idx.shape[0]):
            ids = preds_idx[b]
            confs = preds_prob[b]
            chars: List[str] = []
            char_confs: List[float] = []
            prev = -1
            for i, idx in enumerate(ids):
                if idx != prev and idx != 0 and idx < len(self.charset):
                    chars.append(self.charset[idx])
                    char_confs.append(float(confs[i]))
                prev = idx
            text = "".join(chars)
            conf = float(np.mean(char_confs)) if char_confs else 0.0
            out.append((text, conf))
        return out
