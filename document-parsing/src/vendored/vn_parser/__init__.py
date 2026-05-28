"""Standalone Vietnamese document parser.

Fully ONNX: Layout / OCR-det / Orientation / Table-cls / Table-rec, plus
PaddleOCR CTC (ONNX) for text recognition — no torch in the runtime.
"""

from vn_parser.layout import LayoutDetector, LAYOUT_LABELS  # noqa: F401
from vn_parser.ocr_det import OCRDet  # noqa: F401
from vn_parser.ocr_rec_onnx import PaddleOCRRec  # noqa: F401
from vn_parser.orient_cls import OrientationClassifier  # noqa: F401
from vn_parser.pipeline import VNDocParser  # noqa: F401
