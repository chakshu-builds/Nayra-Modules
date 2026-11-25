"""NAYRA — Public OCR Module Entrypoint.

Provides a standard Python identifier alias for '07_ocr' to eliminate
numeric prefix import limitations:
    from ocr import read_text, OCRResult, OCREngine, get_ocr_engine
"""

from __future__ import annotations

import importlib
import sys

# Import the actual module implementation
_ocr_pkg = importlib.import_module("07_ocr")

# Forward all public API functions and classes
read_text = _ocr_pkg.read_text
get_ocr_engine = _ocr_pkg.get_ocr_engine
OCREngine = _ocr_pkg.OCREngine
OCRResult = _ocr_pkg.OCRResult
TextRegion = _ocr_pkg.TextRegion
PrimaryLanguage = _ocr_pkg.PrimaryLanguage
RegionStatus = _ocr_pkg.RegionStatus
AsyncOCRWorker = _ocr_pkg.AsyncOCRWorker
render_ocr_overlay = _ocr_pkg.render_ocr_overlay
extract_perspective_crop = _ocr_pkg.extract_perspective_crop
sort_regions_reading_order = _ocr_pkg.sort_regions_reading_order
format_combined_text = _ocr_pkg.format_combined_text

# Ensure sys.modules has 'ocr' pointing to this module
if "ocr" not in sys.modules:
    sys.modules["ocr"] = sys.modules[__name__]

__all__ = [
    "read_text",
    "get_ocr_engine",
    "OCREngine",
    "OCRResult",
    "TextRegion",
    "PrimaryLanguage",
    "RegionStatus",
    "AsyncOCRWorker",
    "render_ocr_overlay",
    "extract_perspective_crop",
    "sort_regions_reading_order",
    "format_combined_text",
]
