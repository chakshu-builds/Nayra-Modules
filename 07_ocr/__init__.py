"""NAYRA Module 07 — Public Multilingual OCR Package.

Exposes the unified on-demand OCR API:
    result = read_text(frame)
And persistent engine access:
    engine = get_ocr_engine()
And camera integration components:
    worker = AsyncOCRWorker()
    display_frame = render_ocr_overlay(frame, result, is_busy)
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("NAYRA-OCR")

# Ensure 07_ocr directory is on sys.path for internal imports
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Ensure root directory is on sys.path
_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

# Import pipeline components
try:
    from .pipeline.ocr_engine import OCREngine
    from .pipeline.reading_order import format_combined_text, sort_regions_reading_order
    from .pipeline.result_schema import OCRResult, PrimaryLanguage, RegionStatus, TextRegion
    from .pipeline.text_cropper import extract_perspective_crop
    from .visualizer import render_ocr_overlay
    from .worker import AsyncOCRWorker
except (ImportError, ValueError):
    from pipeline.ocr_engine import OCREngine
    from pipeline.reading_order import format_combined_text, sort_regions_reading_order
    from pipeline.result_schema import OCRResult, PrimaryLanguage, RegionStatus, TextRegion
    from pipeline.text_cropper import extract_perspective_crop
    from visualizer import render_ocr_overlay
    from worker import AsyncOCRWorker

# Register 'ocr' in sys.modules if not present
if "ocr" not in sys.modules:
    sys.modules["ocr"] = sys.modules[__name__]

# Singleton OCR Engine instance for zero-reload inference
_GLOBAL_ENGINE: Optional[OCREngine] = None


def get_ocr_engine(
    preferred_provider: str = "CUDA",
    force_new: bool = False,
    **kwargs: Any,
) -> OCREngine:
    """Get or initialize the persistent global OCREngine instance.

    Args:
        preferred_provider: Preferred ORT provider ("CUDA" or "CPU").
        force_new: If True, re-initializes and replaces the persistent engine.
        **kwargs: Additional parameters passed to OCREngine constructor.

    Returns:
        Persistent OCREngine instance.
    """
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None or force_new:
        _GLOBAL_ENGINE = OCREngine(preferred_provider=preferred_provider, **kwargs)
    return _GLOBAL_ENGINE


def read_text(
    frame: Any,
    preferred_provider: str = "CUDA",
    **kwargs: Any,
) -> OCRResult:
    """Public NAYRA OCR API: Read all detected text from an image frame.

    Preferred usage:
        result = read_text(frame)

    Args:
        frame: OpenCV BGR uint8 image array (numpy.ndarray).
        preferred_provider: Preferred execution provider ("CUDA" or "CPU").
        **kwargs: Optional keyword arguments passed to get_ocr_engine.

    Returns:
        Structured OCRResult containing detected regions, recognized text,
        reading order, language classification, and latency metrics.
    """
    try:
        engine = get_ocr_engine(preferred_provider=preferred_provider, **kwargs)
        return engine.process_image(frame)
    except Exception as exc:
        logger.error("read_text caught unhandled exception: %s", exc, exc_info=True)
        orig_shape = (0, 0)
        if isinstance(frame, np.ndarray) and frame.ndim >= 2:
            orig_shape = frame.shape[:2]
        return OCRResult(
            original_shape=orig_shape,
            execution_providers={"error": "Exception"},
            success=False,
            error_message=f"OCR execution failure: {exc}",
        )


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
