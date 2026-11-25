"""NAYRA Module 07 — Structured Output Schema for Unified OCR Pipeline.

Defines the data contract for individual recognized text regions and the overall
frame OCR result, ensuring consistent Unicode NFC normalization and execution telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import numpy as np


class RegionStatus(str, Enum):
    """Execution status for an individual text region."""
    SUCCESS = "SUCCESS"
    UNRESOLVED_SCRIPT = "UNRESOLVED_SCRIPT"
    FAILED_RECOGNITION = "FAILED_RECOGNITION"
    SKIPPED = "SKIPPED"


class PrimaryLanguage(str, Enum):
    """Document/frame-level language classification."""
    HINDI = "HINDI"
    ENGLISH = "ENGLISH"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


@dataclass
class TextRegion:
    """Represents a single detected and routed/recognized text region.

    Attributes:
        region_id: Unique sequential identifier for the region within the frame.
        polygon: Original-frame polygon or 4-point corner vertices (N, 2) in float32.
        box_xyxy: Axis-aligned bounding box (x_min, y_min, x_max, y_max) in original image space.
        crop_shape: Dimensions (height, width) of the perspective-corrected crop.
        script: Routed script category ('DEVANAGARI', 'LATIN', 'SHARED_NUMERIC', 'UNKNOWN').
        routing_confidence: Confidence score from the ScriptRouter in [0.0, 1.0].
        text: Recognized text string, normalized to Unicode NFC.
        confidence: Recognition confidence score in [0.0, 1.0].
        status: Region processing outcome (RegionStatus).
        reading_order: Sorted reading-order index (0-indexed).
        error_message: Optional error diagnostics if processing failed or was unresolved.
    """

    region_id: int
    polygon: np.ndarray
    box_xyxy: Tuple[float, float, float, float]
    crop_shape: Tuple[int, int]
    script: str
    routing_confidence: float
    text: str = ""
    confidence: float = 0.0
    status: RegionStatus = RegionStatus.SUCCESS
    reading_order: int = 0
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Enforce Unicode NFC normalization on recognized text."""
        if self.text:
            self.text = unicodedata.normalize("NFC", self.text)


@dataclass
class OCRResult:
    """Complete structured result for an input image frame.

    Attributes:
        original_shape: (H, W) pixel dimensions of the input image.
        regions: Ordered list of TextRegion instances in visual reading order.
        combined_text: Full assembled text from all successful regions.
        primary_language: Frame-level language classification (HINDI, ENGLISH, MIXED, etc.).
        detection_count: Total number of text regions detected by DBNet.
        recognition_count: Number of successfully recognized text regions.
        unresolved_count: Number of regions unresolved or failed.
        detection_time_ms: Elapsed time for DBNet detection in milliseconds.
        routing_time_ms: Elapsed time for script routing across all crops in milliseconds.
        recognition_time_ms: Elapsed time for line recognizers across all crops in milliseconds.
        total_time_ms: Total end-to-end processing latency in milliseconds.
        execution_providers: Active execution providers for each subsystem.
        success: Boolean flag indicating if pipeline completed without fatal error.
        error_message: Optional error message if entire frame failed.
    """

    original_shape: Tuple[int, int]
    regions: List[TextRegion] = field(default_factory=list)
    combined_text: str = ""
    primary_language: PrimaryLanguage = PrimaryLanguage.NONE
    detection_count: int = 0
    recognition_count: int = 0
    unresolved_count: int = 0
    detection_time_ms: float = 0.0
    routing_time_ms: float = 0.0
    recognition_time_ms: float = 0.0
    total_time_ms: float = 0.0
    execution_providers: Dict[str, str] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Enforce Unicode NFC normalization on assembled text."""
        if self.combined_text:
            self.combined_text = unicodedata.normalize("NFC", self.combined_text)
