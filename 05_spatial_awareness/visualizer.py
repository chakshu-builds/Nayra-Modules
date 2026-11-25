"""Assistive spatial perception overlay visualizer."""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from .spatial_types import (
    ObstacleStatus,
    ProximityCategory,
    SpatialAnalysisResult,
    SpatialMetrics,
    SpatialObject,
)

logger = logging.getLogger(__name__)

COLOR_CLOSE_OBSTACLE = (0, 40, 230)
COLOR_POTENTIAL_OBSTACLE = (0, 190, 255)
COLOR_NEAR = (50, 70, 240)
COLOR_MEDIUM = (30, 180, 240)
COLOR_FAR = (80, 210, 40)
COLOR_UNKNOWN = (160, 160, 160)
COLOR_ZONE_LINE = (90, 90, 90)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_BLACK = (10, 10, 10)


class SpatialVisualizer:
    """Renders spatial awareness and depth overlays onto video frames."""

    def __init__(
        self,
        zone_left_boundary: float = 0.33,
        zone_right_boundary: float = 0.66,
        show_depth_pip: bool = True,
        pip_scale: float = 0.25,
    ) -> None:
        self.zone_left_boundary = zone_left_boundary
        self.zone_right_boundary = zone_right_boundary
        self.show_depth_pip = show_depth_pip
        self.pip_scale = pip_scale

    def draw_zone_dividers(self, canvas: np.ndarray) -> np.ndarray:
        """Render vertical dashed lines and zone header tags for LEFT, CENTER, RIGHT."""
        h, w = canvas.shape[:2]
        x_left = int(w * self.zone_left_boundary)
        x_right = int(w * self.zone_right_boundary)

        for x in (x_left, x_right):
            cv2.line(canvas, (x, 44), (x, h - 30), COLOR_ZONE_LINE, 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        cv2.putText(canvas, "[ LEFT ]", (int(x_left * 0.38), 60), font, font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[ FORWARD / CENTER ]", (int(x_left + (x_right - x_left) * 0.28), 60), font, font_scale, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[ RIGHT ]", (int(x_right + (w - x_right) * 0.38), 60), font, font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        return canvas

    def draw_spatial_objects(
        self,
        canvas: np.ndarray,
        objects: Sequence[SpatialObject],
    ) -> np.ndarray:
        """Draw bounding boxes, proximity badges, tracking IDs, and obstacle tags."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        for obj in objects:
            x1, y1, x2, y2 = obj.box_xyxy
            track_id_str = f"ID:{obj.track_id}" if obj.track_id is not None else "DET"

            if obj.obstacle_status == ObstacleStatus.POTENTIALLY_CLOSE:
                box_color = COLOR_CLOSE_OBSTACLE
                thickness = 3
                badge_prefix = "! CLOSE OBSTACLE ! "
            elif obj.obstacle_status == ObstacleStatus.POTENTIAL:
                box_color = COLOR_POTENTIAL_OBSTACLE
                thickness = 2
                badge_prefix = "? OBSTACLE ? "
            else:
                thickness = 2
                badge_prefix = ""
                if obj.proximity == ProximityCategory.NEAR:
                    box_color = COLOR_NEAR
                elif obj.proximity == ProximityCategory.MEDIUM:
                    box_color = COLOR_MEDIUM
                elif obj.proximity == ProximityCategory.FAR:
                    box_color = COLOR_FAR
                else:
                    box_color = COLOR_UNKNOWN

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

            depth_str = f"{obj.relative_depth:.2f}" if obj.relative_depth is not None else "?"
            motion_str = f" -> {obj.movement_direction.upper()}" if obj.movement_direction and obj.movement_direction != "unknown" else ""
            label_text = f"{badge_prefix}{track_id_str} {obj.class_name.upper()} | {obj.zone.value} | {obj.proximity.value} ({depth_str}){motion_str}"

            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, 0.44, 1)
            label_y = max(y1 - 6, text_h + 8)

            cv2.rectangle(
                canvas,
                (x1, label_y - text_h - 4),
                (x1 + text_w + 6, label_y + baseline),
                box_color,
                -1,
            )

            text_color = COLOR_TEXT_WHITE if obj.obstacle_status != ObstacleStatus.POTENTIAL else COLOR_TEXT_BLACK
            cv2.putText(
                canvas,
                label_text,
                (x1 + 3, label_y - 2),
                font,
                0.44,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return canvas

    def draw_depth_pip(
        self,
        canvas: np.ndarray,
        proximity_norm: np.ndarray | None,
    ) -> np.ndarray:
        """Render picture-in-picture depth colormap thumbnail in bottom-right corner."""
        if not self.show_depth_pip or proximity_norm is None:
            return canvas

        h, w = canvas.shape[:2]
        pip_w = int(w * self.pip_scale)
        pip_h = int(h * self.pip_scale)

        if pip_w < 50 or pip_h < 50:
            return canvas

        depth_u8 = (np.clip(proximity_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        thumbnail = cv2.resize(depth_color, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

        x_offset = w - pip_w - 12
        y_offset = h - pip_h - 12

        cv2.rectangle(canvas, (x_offset - 2, y_offset - 20), (x_offset + pip_w + 2, y_offset + pip_h + 2), (40, 40, 40), -1)
        cv2.rectangle(canvas, (x_offset - 1, y_offset - 19), (x_offset + pip_w + 1, y_offset - 1), (25, 25, 25), -1)
        cv2.putText(
            canvas,
            "DEPTH MAP (APPROX RELATIVE)",
            (x_offset + 6, y_offset - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        canvas[y_offset : y_offset + pip_h, x_offset : x_offset + pip_w] = thumbnail
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + pip_w, y_offset + pip_h), (80, 80, 80), 1)
        return canvas

    def draw_obstacle_banner(
        self,
        canvas: np.ndarray,
        has_close_obstacle: bool,
        close_count: int,
    ) -> np.ndarray:
        """Render a prominent visual warning banner at the bottom if close obstacles exist."""
        if not has_close_obstacle:
            return canvas

        h, w = canvas.shape[:2]
        banner_h = 32
        banner_y = h - banner_h - 8

        overlay = canvas.copy()
        cv2.rectangle(overlay, (20, banner_y), (w - 20, banner_y + banner_h), COLOR_CLOSE_OBSTACLE, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (20, banner_y), (w - 20, banner_y + banner_h), (255, 255, 255), 1)

        warning_text = f"[!] CAUTION: {close_count} POTENTIAL CLOSE OBSTACLE(S) IN FORWARD PATH"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(warning_text, font, 0.52, 2)
        tx = max(25, int((w - tw) / 2))
        cv2.putText(canvas, warning_text, (tx, banner_y + 22), font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def draw_spatial_hud(
        self,
        canvas: np.ndarray,
        metrics: SpatialMetrics,
        pipeline_latency_ms: float = 0.0,
    ) -> np.ndarray:
        """Render spatial telemetry onto bottom-left corner of canvas."""
        h, _ = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38

        fps_str = f" | Pipe FPS: {1000.0/max(pipeline_latency_ms, 1e-3):.1f}" if pipeline_latency_ms > 0 else ""
        hud_text = (
            f"Depth: {metrics.depth_latency_ms:.1f}ms ({metrics.active_device}) | "
            f"Spatial: {metrics.analysis_latency_ms:.2f}ms | "
            f"Pipe: {pipeline_latency_ms:.1f}ms{fps_str}"
        )

        y_pos = h - 14
        (tw, th), baseline = cv2.getTextSize(hud_text, font, font_scale, 1)
        cv2.rectangle(canvas, (10, y_pos - th - 4), (10 + tw + 8, y_pos + baseline), (20, 20, 20), -1)
        cv2.putText(canvas, hud_text, (14, y_pos - 1), font, font_scale, (200, 220, 240), 1, cv2.LINE_AA)
        return canvas

    def render(
        self,
        frame: np.ndarray,
        spatial_result: SpatialAnalysisResult,
        pipeline_latency_ms: float = 0.0,
    ) -> np.ndarray:
        """Full rendering pipeline for spatial awareness."""
        canvas = frame.copy()
        canvas = self.draw_zone_dividers(canvas)
        canvas = self.draw_spatial_objects(canvas, spatial_result.objects)
        canvas = self.draw_depth_pip(canvas, spatial_result.depth_normalized)
        canvas = self.draw_obstacle_banner(
            canvas,
            spatial_result.has_close_obstacle,
            spatial_result.close_obstacles_count,
        )
        canvas = self.draw_spatial_hud(canvas, spatial_result.metrics, pipeline_latency_ms)
        return canvas
