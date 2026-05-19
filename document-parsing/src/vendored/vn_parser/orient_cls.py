"""Page orientation classifier (PP-LCNet)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ANGLES = [0, 90, 180, 270]  # output index -> rotation angle (degrees, clockwise)


class OrientationClassifier:
    """Detect document page orientation. Returns the angle to rotate
    the image *back* to upright (0/90/180/270 degrees, clockwise)."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_path: Union[str, Path],
        providers: Optional[Sequence[str]] = None,
    ):
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img_bgr, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.MEAN) / self.STD
        return rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)

    def predict_angle(self, img: Union[np.ndarray, Image.Image]) -> int:
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))[..., ::-1]
        else:
            arr = img
        x = self._preprocess(arr)
        out = self.session.run(None, {self.input_name: x})[0]
        idx = int(np.argmax(out, axis=1)[0])
        return ANGLES[idx]

    def correct(self, img: Union[np.ndarray, Image.Image]) -> Tuple[Union[np.ndarray, Image.Image], int]:
        """Return (rotated image, detected angle)."""
        angle = self.predict_angle(img)
        if angle == 0:
            return img, 0
        if isinstance(img, Image.Image):
            rotated = img.rotate(-angle, expand=True)  # PIL: positive = ccw
        else:
            rot = {90: cv2.ROTATE_90_CLOCKWISE,
                   180: cv2.ROTATE_180,
                   270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]
            rotated = cv2.rotate(img, rot)
        return rotated, angle
