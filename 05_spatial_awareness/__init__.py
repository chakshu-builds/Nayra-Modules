"""
NAYRA - Assistive System
Module 05: Obstacle & Spatial Awareness

Public package exports for:
- SpatialZone, ProximityCategory, ObstacleStatus
- SpatialObject, SpatialMetrics, SpatialAnalysisResult
- DepthEstimator
- SpatialAnalyzer
- SpatialVisualizer
"""

from __future__ import annotations

from .depth_estimator import DEPTH_MODEL_CONFIGS, DepthEstimator
from .spatial_analyzer import (
    NON_OBSTACLE_CLASSES,
    PHYSICAL_OBSTACLE_CLASSES,
    SpatialAnalyzer,
)
from .spatial_types import (
    ObstacleStatus,
    ProximityCategory,
    SpatialAnalysisResult,
    SpatialMetrics,
    SpatialObject,
    SpatialZone,
)
from .visualizer import SpatialVisualizer

__all__ = [
    "DEPTH_MODEL_CONFIGS",
    "DepthEstimator",
    "NON_OBSTACLE_CLASSES",
    "ObstacleStatus",
    "PHYSICAL_OBSTACLE_CLASSES",
    "ProximityCategory",
    "SpatialAnalysisResult",
    "SpatialAnalyzer",
    "SpatialMetrics",
    "SpatialObject",
    "SpatialVisualizer",
    "SpatialZone",
]
