"""Per-stage device resolution.

Each ``DEVICE_*`` env var (and matching ``settings.device_*`` field) accepts:

    "auto"       → CUDA if torch.cuda.is_available(), else CPU
    "cpu"        → force CPU (CPUExecutionProvider for ONNX, "cpu" for torch)
    "cuda"       → force CUDA index 0
    "cuda:N"     → force CUDA index N (multi-GPU systems)

Two helpers:

    resolve_onnx_providers(spec) → onnxruntime providers list
    resolve_torch_device(spec)   → torch device string ("cuda:N" | "cpu")

Both functions raise on a "cuda*" spec when the corresponding runtime is
not available, so misconfiguration surfaces at worker start rather than
silently falling back at the first parse.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

_CUDA_RE = re.compile(r"^cuda(?::(\d+))?$")


@lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def onnx_cuda_available() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def resolve_onnx_providers(spec: str) -> list:
    """Return an onnxruntime providers list for ``spec``.

    The returned list is suitable as the ``providers=`` keyword for
    ``InferenceSession`` and the ``LayoutDetector`` / ``OCRDet`` etc.
    constructors in vn_parser.
    """
    spec = (spec or "auto").strip().lower()
    if spec == "cpu":
        return ["CPUExecutionProvider"]

    m = _CUDA_RE.match(spec)
    if m:
        if not onnx_cuda_available():
            raise RuntimeError(
                f"onnxruntime CUDAExecutionProvider not available "
                f"(install onnxruntime-gpu and CUDA drivers); requested {spec!r}"
            )
        idx = int(m.group(1) or 0)
        return [
            ("CUDAExecutionProvider", {"device_id": idx}),
            "CPUExecutionProvider",   # graceful fallback for ops CUDA can't run
        ]

    if spec == "auto":
        if onnx_cuda_available():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    log.warning("Unknown ONNX device spec %r — falling back to CPU", spec)
    return ["CPUExecutionProvider"]


def resolve_torch_device(spec: str) -> str:
    """Return a torch device string for ``spec``."""
    spec = (spec or "auto").strip().lower()
    if spec == "cpu":
        return "cpu"

    m = _CUDA_RE.match(spec)
    if m:
        if not torch_cuda_available():
            raise RuntimeError(
                f"torch.cuda not available (install torch CUDA wheel); "
                f"requested {spec!r}"
            )
        idx = int(m.group(1) or 0)
        return f"cuda:{idx}"

    if spec == "auto":
        return "cuda:0" if torch_cuda_available() else "cpu"

    log.warning("Unknown torch device spec %r — falling back to CPU", spec)
    return "cpu"


def describe_provider(providers: list) -> str:
    """Human-readable summary like ``CUDA:0`` or ``CPU`` for log lines."""
    if not providers:
        return "?"
    p = providers[0]
    if isinstance(p, tuple):
        name, opts = p
        if name == "CUDAExecutionProvider":
            return f"CUDA:{opts.get('device_id', 0)}"
        return name
    return {
        "CUDAExecutionProvider": "CUDA",
        "CPUExecutionProvider": "CPU",
    }.get(p, p)
