"""Vietnamese OCR recognizer (VietOCR transformer).

Uses pbcquoc/vietocr `vgg_transformer` for autoregressive transformer decoding.

Offline-friendly: by default we read VietOCR's YAML configs from a local
directory (``$VIETOCR_CONFIG_DIR``) so a flaky/blocked vocr.vn host can't
prevent the parser from initializing. Pre-fetch the configs once with:

    curl -o base.yml https://vocr.vn/data/vietocr/config/base.yml
    curl -o vgg-transformer.yml https://vocr.vn/data/vietocr/config/vgg-transformer.yml
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _load_cfg(config_name: str):
    """Load a VietOCR config — local YAML files first, fall back to upstream."""
    from vietocr.tool.config import Cfg

    cfg_dir = os.environ.get("VIETOCR_CONFIG_DIR", "/root/.cache/vietocr")
    base = Path(cfg_dir) / "base.yml"
    name_yaml = Path(cfg_dir) / f"{config_name.replace('_', '-')}.yml"

    if base.exists() and name_yaml.exists():
        # Mirror Cfg.load_config_from_name's logic but from local files.
        import yaml
        with base.open() as f:
            merged = yaml.safe_load(f)
        with name_yaml.open() as f:
            merged.update(yaml.safe_load(f) or {})
        log.info("VietOCR cfg loaded from %s + %s", base, name_yaml)
        return Cfg(merged)

    log.warning("VietOCR cfg dir %s missing files; falling back to vocr.vn", cfg_dir)
    return Cfg.load_config_from_name(config_name)


class VietOCRRec:
    """Wrapper around vietocr.tool.predictor.Predictor.

    First instantiation downloads pretrained weights to ~/.cache/...
    Pass weights=<local_path> to override.
    """

    def __init__(
        self,
        config_name: str = "vgg_transformer",
        device: str = "cpu",
        beamsearch: bool = False,
        weights: Optional[str] = None,
    ):
        from vietocr.tool.predictor import Predictor

        cfg = _load_cfg(config_name)
        cfg["device"] = device
        cfg["predictor"] = cfg.get("predictor", {})
        cfg["predictor"]["beamsearch"] = beamsearch
        if weights:
            cfg["weights"] = weights
        self.cfg = cfg
        self.predictor = Predictor(cfg)

    def recognize(self, img: Union[np.ndarray, Image.Image]) -> str:
        return self.predictor.predict(_to_pil_rgb(img))

    def recognize_batch(
        self,
        imgs: List[Union[np.ndarray, Image.Image]],
    ) -> List[str]:
        """True batched recognition via vietocr's ``Predictor.predict_batch``.

        ``predict_batch`` buckets crops by image width then issues one torch
        forward per bucket, so on GPU this is dramatically faster than
        looping ``recognize()``.
        """
        if not imgs:
            return []
        pil_list = [_to_pil_rgb(im) for im in imgs]
        return self.predictor.predict_batch(pil_list)


def _to_pil_rgb(img: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(img, np.ndarray):
        # vietocr expects RGB PIL. Crops in this project come as BGR
        # (OpenCV) — flip channels.
        if img.ndim == 3 and img.shape[2] == 3:
            return Image.fromarray(img[..., ::-1])
        return Image.fromarray(img)
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")
