"""Spatial awareness analysis combining detections, tracks, and depth estimation."""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import cv2
import numpy as np

from config.settings import SpatialAwarenessSettings, get_default_spatial_settings
from .depth_estimator import DepthEstimator
from .spatial_types import (
    ObstacleStatus,
    ProximityCategory,
    SpatialAnalysisResult,
    SpatialMetrics,
    SpatialObject,
    SpatialZone,
)

logger = logging.getLogger(__name__)

PHYSICAL_OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "bench",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "refrigerator",
    "suitcase",
    "backpack",
    "handbag",
    "stairs",
    "door",
}

NON_OBSTACLE_CLASSES = {
    "cell phone",
    "remote",
    "mouse",
    "keyboard",
    "cup",
    "bottle",
    "wine glass",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "potted plant",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "book",
    "clock",
}


class SpatialAnalyzer:
    """Orchestrates horizontal spatial zoning, depth sampling, and obstacle identification."""

    def __init__(
        self,
        depth_estimator: DepthEstimator | None = None,
        settings: SpatialAwarenessSettings | None = None,
    ) -> None:
        self.settings = settings or get_default_spatial_settings()
        self.depth_estimator = depth_estimator or DepthEstimator(self.settings)

        self._track_depth_cache: dict[int, tuple[float, float, ProximityCategory, ObstacleStatus]] = {}
        self._frame_count: int = 0
        self._cached_depth_map: np.ndarray | None = None
        self._cached_proximity_norm: np.ndarray | None = None

        logger.info(
            "SpatialAnalyzer initialized (bounds: [%.2f, %.2f], near: %.2f, far: %.2f)",
            self.settings.zone_left_boundary,
            self.settings.zone_right_boundary,
            self.settings.near_threshold,
            self.settings.far_threshold,
        )

    def classify_horizontal_zone(self, center_x_norm: float) -> SpatialZone:
        """Classify horizontal sector (LEFT, CENTER, RIGHT)."""
        if center_x_norm < self.settings.zone_left_boundary:
            return SpatialZone.LEFT
        elif center_x_norm > self.settings.zone_right_boundary:
            return SpatialZone.RIGHT
        return SpatialZone.CENTER

    def sample_roi_depth(
        self,
        proximity_norm: np.ndarray,
        depth_raw: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
    ) -> tuple[float | None, float | None, float]:
        """Sample representative median depth within bounding box interior crop."""
        img_h, img_w = frame_shape
        x1, y1, x2, y2 = box_xyxy

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return None, None, 0.0

        crop_ratio = self.settings.depth_crop_ratio
        margin_x = int(w * (1.0 - crop_ratio) / 2.0)
        margin_y = int(h * (1.0 - crop_ratio) / 2.0)

        cx1 = max(0, min(img_w - 1, x1 + margin_x))
        cy1 = max(0, min(img_h - 1, y1 + margin_y))
        cx2 = max(cx1 + 1, min(img_w, x2 - margin_x))
        cy2 = max(cy1 + 1, min(img_h, y2 - margin_y))

        roi_prox = proximity_norm[cy1:cy2, cx1:cx2]
        roi_raw = depth_raw[cy1:cy2, cx1:cx2]

        if roi_prox.size == 0:
            return None, None, 0.0

        valid_mask = np.isfinite(roi_prox) & np.isfinite(roi_raw)
        valid_prox = roi_prox[valid_mask]
        valid_raw = roi_raw[valid_mask]

        if valid_prox.size < 4:
            return None, None, 0.0

        median_prox = float(np.median(valid_prox))
        median_raw = float(np.median(valid_raw))

        valid_ratio = float(valid_prox.size) / float(roi_prox.size)
        dispersion_std = float(np.std(valid_prox))
        consistency_score = max(0.2, min(1.0, 1.0 - 2.0 * dispersion_std))
        spatial_confidence = float(np.clip(valid_ratio * consistency_score, 0.0, 1.0))

        return median_prox, median_raw, spatial_confidence

    def classify_proximity(self, proximity_score: float | None) -> ProximityCategory:
        """Map normalized proximity score to categorical distance."""
        if proximity_score is None or not np.isfinite(proximity_score):
            return ProximityCategory.UNKNOWN

        if proximity_score >= self.settings.near_threshold:
            return ProximityCategory.NEAR
        elif proximity_score <= self.settings.far_threshold:
            return ProximityCategory.FAR
        return ProximityCategory.MEDIUM

    def classify_obstacle(
        self,
        class_name: str,
        zone: SpatialZone,
        proximity: ProximityCategory,
        box_area_ratio: float,
    ) -> ObstacleStatus:
        """Evaluate assistive obstacle candidate risk for walking navigation."""
        if proximity == ProximityCategory.UNKNOWN:
            return ObstacleStatus.UNKNOWN

        clean_class = class_name.lower().strip()
        if clean_class in NON_OBSTACLE_CLASSES:
            return ObstacleStatus.NONE

        is_physical_obstacle = clean_class in PHYSICAL_OBSTACLE_CLASSES
        if not is_physical_obstacle:
            if zone != SpatialZone.CENTER or box_area_ratio < self.settings.min_obstacle_box_area_ratio * 2.0:
                return ObstacleStatus.NONE

        if proximity == ProximityCategory.FAR:
            return ObstacleStatus.NONE

        if proximity == ProximityCategory.NEAR:
            return ObstacleStatus.POTENTIALLY_CLOSE if zone == SpatialZone.CENTER else ObstacleStatus.POTENTIAL

        if proximity == ProximityCategory.MEDIUM:
            if zone == SpatialZone.CENTER and box_area_ratio >= self.settings.min_obstacle_box_area_ratio:
                return ObstacleStatus.POTENTIAL
            return ObstacleStatus.NONE

        return ObstacleStatus.NONE

    def generate_assistive_summary(
        self,
        class_name: str,
        zone: SpatialZone,
        proximity: ProximityCategory,
        obstacle_status: ObstacleStatus,
    ) -> str:
        """Create natural language summary for assistive perception."""
        zone_desc = {
            SpatialZone.LEFT: "on left",
            SpatialZone.CENTER: "ahead in center",
            SpatialZone.RIGHT: "on right",
        }.get(zone, "in view")

        prox_desc = {
            ProximityCategory.NEAR: "near",
            ProximityCategory.MEDIUM: "medium distance",
            ProximityCategory.FAR: "far",
            ProximityCategory.UNKNOWN: "distance unknown",
        }.get(proximity, "")

        base = f"{class_name.capitalize()} {zone_desc}, {prox_desc}"
        if obstacle_status == ObstacleStatus.POTENTIALLY_CLOSE:
            return f"[CLOSE OBSTACLE] {base}"
        elif obstacle_status == ObstacleStatus.POTENTIAL:
            return f"[POTENTIAL OBSTACLE] {base}"
        return base

    def analyze(
        self,
        frame: np.ndarray,
        tracks_or_detections: Sequence[Any],
        timestamp: float = 0.0,
    ) -> SpatialAnalysisResult:
        """Execute full spatial awareness pipeline for current frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame is invalid or empty.")

        t_total_start = time.perf_counter()
        img_h, img_w = frame.shape[:2]
        self._frame_count += 1

        run_depth = (
            self.settings.depth_interval <= 1
            or (self._frame_count % self.settings.depth_interval == 1)
            or self._cached_depth_map is None
        )

        depth_raw = self._cached_depth_map
        proximity_norm = self._cached_proximity_norm
        depth_latency_ms = 0.0

        if run_depth:
            depth_raw, proximity_norm, depth_latency_ms = self.depth_estimator.estimate(frame)
            self._cached_depth_map = depth_raw
            self._cached_proximity_norm = proximity_norm

        t_analysis_start = time.perf_counter()
        spatial_objects: list[SpatialObject] = []
        close_obstacles_count = 0
        potential_obstacles_count = 0

        for item in tracks_or_detections:
            track_id = getattr(item, "track_id", None)
            class_id = getattr(item, "class_id", -1)
            class_name = getattr(item, "class_name", "object")
            confidence = getattr(item, "confidence", 0.0)
            box_xyxy = getattr(item, "box_xyxy", (0, 0, 0, 0))
            movement_dir = getattr(item, "movement_direction", None)
            if hasattr(movement_dir, "value"):
                movement_dir = movement_dir.value

            center_x_norm = ((box_xyxy[0] + box_xyxy[2]) / 2.0) / float(img_w)
            zone = self.classify_horizontal_zone(center_x_norm)

            box_w = max(0, box_xyxy[2] - box_xyxy[0])
            box_h = max(0, box_xyxy[3] - box_xyxy[1])
            box_area_ratio = (box_w * box_h) / float(img_w * img_h)

            box_norm = (
                box_xyxy[0] / float(img_w),
                box_xyxy[1] / float(img_h),
                box_xyxy[2] / float(img_w),
                box_xyxy[3] / float(img_h),
            )

            if proximity_norm is not None and depth_raw is not None:
                prox_score, raw_val, spat_conf = self.sample_roi_depth(
                    proximity_norm,
                    depth_raw,
                    box_xyxy,
                    (img_h, img_w),
                )
                prox_cat = self.classify_proximity(prox_score)
                obst_status = self.classify_obstacle(class_name, zone, prox_cat, box_area_ratio)

                if track_id is not None and prox_score is not None:
                    self._track_depth_cache[track_id] = (prox_score, raw_val, prox_cat, obst_status)
            elif track_id is not None and track_id in self._track_depth_cache:
                prox_score, raw_val, prox_cat, obst_status = self._track_depth_cache[track_id]
                spat_conf = 0.8
            else:
                prox_score, raw_val, spat_conf = None, None, 0.0
                prox_cat = ProximityCategory.UNKNOWN
                obst_status = ObstacleStatus.UNKNOWN

            if obst_status == ObstacleStatus.POTENTIALLY_CLOSE:
                close_obstacles_count += 1
            elif obst_status == ObstacleStatus.POTENTIAL:
                potential_obstacles_count += 1

            summary = self.generate_assistive_summary(class_name, zone, prox_cat, obst_status)

            spatial_objects.append(
                SpatialObject(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    box_xyxy=box_xyxy,
                    box_normalized=box_norm,
                    zone=zone,
                    relative_depth=prox_score,
                    raw_depth_value=raw_val,
                    proximity=prox_cat,
                    obstacle_status=obst_status,
                    spatial_confidence=spat_conf,
                    frame_timestamp=timestamp,
                    frame_width=img_w,
                    frame_height=img_h,
                    movement_direction=movement_dir,
                    spatial_summary=summary,
                )
            )

        t_analysis_end = time.perf_counter()
        analysis_latency_ms = (t_analysis_end - t_analysis_start) * 1000.0
        total_latency_ms = (t_analysis_end - t_total_start) * 1000.0

        gpu_alloc_mb = 0.0
        gpu_res_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        except Exception:
            pass

        metrics = SpatialMetrics(
            depth_latency_ms=depth_latency_ms,
            analysis_latency_ms=analysis_latency_ms,
            total_spatial_latency_ms=total_latency_ms,
            active_device=self.depth_estimator.active_device,
            inference_fps=self.depth_estimator.fps,
            gpu_memory_allocated_mb=gpu_alloc_mb,
            gpu_memory_reserved_mb=gpu_res_mb,
        )

        return SpatialAnalysisResult(
            objects=spatial_objects,
            depth_map=depth_raw,
            depth_normalized=proximity_norm,
            metrics=metrics,
            has_close_obstacle=(close_obstacles_count > 0),
            close_obstacles_count=close_obstacles_count,
            potential_obstacles_count=potential_obstacles_count + close_obstacles_count,
            frame_timestamp=timestamp,
        )
