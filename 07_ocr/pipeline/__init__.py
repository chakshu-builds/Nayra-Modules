"""NAYRA Module 07 — Unified Multilingual OCR Pipeline Package.

Exports:
- OCREngine: Main pipeline class.
- OCRResult: Complete pipeline result dataclass.
- TextRegion: Per-region detection & recognition dataclass.
- RegionStatus: Enum for per-region execution status.
- PrimaryLanguage: Enum for document-level language classification.
- extract_perspective_crop: Perspective transformation cropping utility.
- sort_regions_reading_order: Natural reading order sorter.
"""

from .batch_recognizer import predict_batch
from .ocr_engine import OCREngine
from .reading_order import format_combined_text, sort_regions_reading_order
from .result_schema import OCRResult, PrimaryLanguage, RegionStatus, TextRegion
from .text_cropper import extract_perspective_crop

__all__ = [
    "OCREngine",
    "OCRResult",
    "TextRegion",
    "RegionStatus",
    "PrimaryLanguage",
    "extract_perspective_crop",
    "sort_regions_reading_order",
    "format_combined_text",
    "predict_batch",
]
