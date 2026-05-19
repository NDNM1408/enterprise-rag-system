"""Math formula recognizer — deferred PyTorch path.

UnimerNet (vision-encoder-decoder) is autoregressive; converting to a single
ONNX is non-trivial. This module loads the PyTorch HF checkpoint when needed
so the user can call MFR without converting it.

Requires: torch, transformers (already in .venv-tools).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image


class MFRRecognizer:
    def __init__(self, model_dir: Union[str, Path], device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer, AutoImageProcessor  # noqa
        self.device = device
        self.model_dir = str(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        except Exception:
            self.processor = None
        self.model = AutoModel.from_pretrained(self.model_dir).to(device).eval()

    def predict(
        self, image: Union[np.ndarray, Image.Image], max_new_tokens: int = 512
    ) -> str:
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image[..., ::-1] if image.ndim == 3 else image)
        else:
            pil = image.convert("RGB")
        if self.processor:
            pixel_values = self.processor(images=pil, return_tensors="pt").pixel_values
        else:
            import torch
            arr = np.array(pil).transpose(2, 0, 1) / 255.0
            pixel_values = torch.from_numpy(arr[None, ...]).float()
        pixel_values = pixel_values.to(self.device)
        import torch
        with torch.no_grad():
            ids = self.model.generate(pixel_values=pixel_values,
                                      max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)
