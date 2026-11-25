"""
NAYRA - Assistive System
Module 05: Obstacle & Spatial Awareness - Data Structures & Types

Defines strongly-typed representations for:
- Horizontal spatial zones (LEFT, CENTER, RIGHT)
- Relative proximity categories (NEAR, MEDIUM, FAR, UNKNOWN)
- Obstacle statuses (NONE, POTENTIAL, POTENTIALLY_CLOSE, UNKNOWN)
- SpatialObject detections associated with tracking IDs and depth estimates
- Performance and latency metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class SpatialZone(str, Enum):
    """Horizontal region in the camera's field of view."""
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class ProximityCategory(str, Enum):
    """Approximate relative distance category derived from monocular depth estimation.

    Note: Monocular depth models provide relative scene disparity or ordinal depth,
    not calibrated metric distances. These categories indicate relative proximity.
    """
    NEAR = "NEAR"
    MEDIUM = "MEDIUM"
    FAR = "FAR"
    UNKNOWN = "UNKNOWN"


class ObstacleStatus(str, Enum):
    """Potential collision risk status for navigation assistance.

    Safety Notice: Camera-only perception does NOT guarantee collision avoidance.
    This classification is purely assistive and heuristic.
    """
    NONE = "NONE"                     # Object present but not in immediate forward path or non-obstacle class
    POTENTIAL = "POTENTIAL"           # Obstacle candidate in scene (e.g. forward medium distance or close side)
    POTENTIALLY_CLOSE = "POTENTIALLY_CLOSE"  # High-priority obstacle directly ahead in forward view and near
    UNKNOWN = "UNKNOWN"               # Depth estimation unavailable, invalid, or unreliable


@dataclass
class SpatialObject:
    """Represents a tracked object enriched with spatial positioning, depth, and obstacle classification.

    Attributes:
        track_id: Persistent tracking session ID preserved from Module 04 (or None if untracked).
        class_id: Model taxonomy numeric class ID.
        class_name: Human-readable object label (e.g., 'chair', 'person').
        confidence: Object detection confidence score [0.0, 1.0].
        box_xyxy: Integer bounding box pixel coordinates (x1, y1, x2, y2).
        box_normalized: Normalized coordinates [0.0, 1.0] (x1_n, y1_n, x2_n, y2_n).
        zone: Horizontal sector in camera view (LEFT, CENTER, RIGHT).
        relative_depth: Normalized relative proximity score [0.0, 1.0] (1.0 = closest, 0.0 = farthest).
        raw_depth_value: Median raw depth/disparity model output within the object ROI.
        proximity: Relative proximity classification (NEAR, MEDIUM, FAR, UNKNOWN).
        obstacle_status: Obstacle assessment (NONE, POTENTIAL, POTENTIALLY_CLOSE, UNKNOWN).
        spatial_confidence: Reliability indicator [0.0, 1.0] based on depth variance / valid pixel ratio.
        frame_timestamp: Wall-clock timestamp when frame was acquired.
        frame_width: Original image width in pixels.
        frame_height: Original image height in pixels.
        movement_direction: 2D image-plane movement direction preserved from Module 04.
        spatial_summary: Human-readable assistive summary (e.g., 'Person ahead, near').
    """

    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    box_normalized: Tuple[float, float, float, float]
    zone: SpatialZone
    relative_depth: Optional[float]
    raw_depth_value: Optional[float]
    proximity: ProximityCategory
    obstacle_status: ObstacleStatus
    spatial_confidence: float
    frame_timestamp: float
    frame_width: int
    frame_height: int
    movement_direction: Optional[str] = None
    spatial_summary: str = ""

    @property
    def center_x(self) -> float:
        """Horizontal center in pixel coordinates."""
        return (self.box_xyxy[0] + self.box_xyxy[2]) / 2.0

    @property
    def center_y(self) -> float:
        """Vertical center in pixel coordinates."""
        return (self.box_xyxy[1] + self.box_xyxy[3]) / 2.0

    @property
    def center_x_norm(self) -> float:
        """Horizontal center normalized in range [0.0, 1.0]."""
        if self.frame_width <= 0:
            return 0.5
        return self.center_x / float(self.frame_width)

    @property
    def box_area_ratio(self) -> float:
        """Bounding box area as fraction of total frame area."""
        if self.frame_width <= 0 or self.frame_height <= 0:
            return 0.0
        w = max(0, self.box_xyxy[2] - self.box_xyxy[0])
        h = max(0, self.box_xyxy[3] - self.box_xyxy[1])
        return (w * h) / float(self.frame_width * self.frame_height)


@dataclass
class SpatialMetrics:
    """Inference and processing latency telemetry for spatial analysis.

    Attributes:
        depth_latency_ms: Milliseconds for depth estimation model forward pass & postprocessing.
        analysis_latency_ms: Milliseconds for spatial zoning, ROI sampling, and obstacle tagging.
        total_spatial_latency_ms: Sum of depth estimation and spatial analysis latency.
        active_device: Active hardware execution device (e.g., 'CUDAExecutionProvider' or 'CPU').
        inference_fps: Calculated inference frames per second for the depth model.
        gpu_memory_allocated_mb: PyTorch allocated GPU memory, if available.
        gpu_memory_reserved_mb: PyTorch reserved GPU memory, if available.
    """

    depth_latency_ms: float = 0.0
    analysis_latency_ms: float = 0.0
    total_spatial_latency_ms: float = 0.0
    active_device: str = "CPU"
    inference_fps: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0


@dataclass
class SpatialAnalysisResult:
    """Consolidated spatial awareness output for an individual video frame.

    Attributes:
        objects: Spatially analyzed objects with zones, depths, and obstacle tags.
        depth_map: Full-frame 2D relative depth map aligned with frame dimensions (or None).
        depth_normalized: Normalized depth map [0.0, 1.0] for visual rendering (or None).
        metrics: Spatial and depth estimation performance metrics.
        has_close_obstacle: True if one or more objects have POTENTIALLY_CLOSE status.
        close_obstacles_count: Count of objects flagged as POTENTIALLY_CLOSE.
        potential_obstacles_count: Count of objects flagged as POTENTIAL or POTENTIALLY_CLOSE.
        frame_timestamp: Timestamp when the source frame was captured.
    """

    objects: List[SpatialObject] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    depth_normalized: Optional[np.ndarray] = None
    metrics: SpatialMetrics = field(default_factory=SpatialMetrics)
    has_close_obstacle: bool = False
    close_obstacles_count: int = 0
    potential_obstacles_count: int = 0
    frame_timestamp: float = 0.0
