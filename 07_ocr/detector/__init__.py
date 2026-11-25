"""NAYRA Module 07 — DBNet Text Detection Package.

Exports:
- DBNetDetector: Full detection pipeline wrapper with ORT session management.
- DetectionResult: Structured detection output dataclass.
- DBPostProcessor: Standalone DBNet probability map postprocessor.
- sort_reading_order: Geometry utility for reading-order sorting.
"""

from .dbnet_detector import DBNetDetector, DetectionResult
from .geometry import (
    box_score_fast,
    order_points_clockwise,
    polygon_to_xyxy,
    restore_coordinates,
    sort_reading_order,
    unclip_rotated_rect,
)
from .postprocess import DBPostProcessor

__all__ = [
    "DBNetDetector",
    "DetectionResult",
    "DBPostProcessor",
    "order_points_clockwise",
    "unclip_rotated_rect",
    "box_score_fast",
    "polygon_to_xyxy",
    "restore_coordinates",
    "sort_reading_order",
]
