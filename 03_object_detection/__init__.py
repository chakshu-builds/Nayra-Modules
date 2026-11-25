"""
NAYRA - Assistive System
Module 03: Assistive Object Detection

Exposes public interface for model loading, inference, and assistive result processing.
"""

from .detector import ObjectDetector
from .model_registry import (
    ASSISTIVE_CLASSES,
    COCO_CLASSES,
    UNSUPPORTED_ASSISTIVE_CLASSES,
    ModelMetadata,
    ensure_model_weights,
    get_model_metadata,
)
from .postprocessor import Detection, DetectionMetrics, DetectionPostprocessor

__all__ = [
    "ObjectDetector",
    "Detection",
    "DetectionMetrics",
    "DetectionPostprocessor",
    "COCO_CLASSES",
    "ASSISTIVE_CLASSES",
    "UNSUPPORTED_ASSISTIVE_CLASSES",
    "ModelMetadata",
    "ensure_model_weights",
    "get_model_metadata",
]
