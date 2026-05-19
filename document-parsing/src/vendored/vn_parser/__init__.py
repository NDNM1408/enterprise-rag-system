"""Standalone Vietnamese document parser.

ONNX models for Layout / OCR-det / Orientation / Table-cls / Table-rec.
VietOCR (transformer) for text recognition.
"""

from vn_parser.layout import LayoutDetector, LAYOUT_LABELS  # noqa: F401
from vn_parser.ocr_det import OCRDet  # noqa: F401
from vn_parser.ocr_rec import VietOCRRec  # noqa: F401
from vn_parser.orient_cls import OrientationClassifier  # noqa: F401
from vn_parser.pipeline import VNDocParser  # noqa: F401
