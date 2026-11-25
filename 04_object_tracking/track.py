"""Track trajectory representation, lifecycle states, and 2D motion estimation."""

from __future__ import annotations

import importlib
from enum import Enum

import numpy as np

# Dynamically import Detection to handle numeric folder prefixes
_postprocessor = importlib.import_module("03_object_detection.postprocessor")
Detection = _postprocessor.Detection

from .kalman_filter import KalmanBoxTracker, xyah_to_xyxy, xyxy_to_xyah


class TrackState(str, Enum):
    """Lifecycle state of an object trajectory."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


class MovementDirection(str, Enum):
    """2D image-plane movement direction."""
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    STATIONARY = "stationary"
    UNKNOWN = "unknown"


class Track:
    """Continuous trajectory of a detected object across video frames."""

    _count: int = 0

    def __init__(
        self,
        detection: Detection,
        kalman_tracker: KalmanBoxTracker,
        timestamp: float,
    ) -> None:
        Track._count += 1
        self.track_id: int = Track._count
        self.class_id: int = detection.class_id
        self.class_name: str = detection.class_name
        self.confidence: float = detection.confidence

        self.box_xyxy: tuple[int, int, int, int] = detection.box_xyxy
        self.previous_box_xyxy: tuple[int, int, int, int] | None = None
        self.center: tuple[float, float] = (detection.center_x, detection.center_y)

        self.spatial_zone: str = detection.spatial_zone
        self.spatial_hint: str = detection.spatial_hint
        self.frame_width: int = detection.frame_width
        self.frame_height: int = detection.frame_height

        self.state: TrackState = TrackState.TENTATIVE
        self.track_age: int = 1
        self.hits: int = 1
        self.consecutive_missed_frames: int = 0
        self.timestamp: float = timestamp

        self.movement_direction: MovementDirection = MovementDirection.UNKNOWN
        self._center_history: list[tuple[float, float]] = [self.center]
        self._max_history: int = 10

        self._kalman = kalman_tracker
        measurement = xyxy_to_xyah(self.box_xyxy)
        self.mean, self.covariance = self._kalman.initiate(measurement)

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset global track ID counter."""
        cls._count = 0

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_removed(self) -> bool:
        return self.state == TrackState.REMOVED

    def predict(self) -> tuple[int, int, int, int]:
        """Advance Kalman filter state by one time step."""
        if self.mean[7] > 0 and self.consecutive_missed_frames > 0:
            self.mean[7] *= 0.8
        self.mean, self.covariance = self._kalman.predict(self.mean, self.covariance)
        return xyah_to_xyxy(self.mean)

    def update(
        self,
        detection: Detection,
        timestamp: float,
        stationary_threshold_px: float = 6.0,
    ) -> None:
        """Incorporate new detection measurement into the active track."""
        measurement = xyxy_to_xyah(detection.box_xyxy)
        self.mean, self.covariance = self._kalman.update(self.mean, self.covariance, measurement)

        self.previous_box_xyxy = self.box_xyxy
        self.box_xyxy = xyah_to_xyxy(self.mean)
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name

        self.center = (
            (self.box_xyxy[0] + self.box_xyxy[2]) / 2.0,
            (self.box_xyxy[1] + self.box_xyxy[3]) / 2.0,
        )
        self.spatial_zone = detection.spatial_zone
        self.spatial_hint = detection.spatial_hint
        self.frame_width = detection.frame_width
        self.frame_height = detection.frame_height

        self.track_age += 1
        self.hits += 1
        self.consecutive_missed_frames = 0
        self.timestamp = timestamp

        self._center_history.append(self.center)
        if len(self._center_history) > self._max_history:
            self._center_history.pop(0)

        self._estimate_movement_direction(stationary_threshold_px)

    def mark_missed(self, max_lost_frames: int = 30) -> None:
        """Mark track as unobserved in the current frame."""
        self.track_age += 1
        self.consecutive_missed_frames += 1

        self.previous_box_xyxy = self.box_xyxy
        self.box_xyxy = xyah_to_xyxy(self.mean)
        self.center = (
            (self.box_xyxy[0] + self.box_xyxy[2]) / 2.0,
            (self.box_xyxy[1] + self.box_xyxy[3]) / 2.0,
        )

        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

        if self.consecutive_missed_frames >= max_lost_frames:
            self.state = TrackState.REMOVED

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def confirm(self) -> None:
        self.state = TrackState.CONFIRMED

    def _estimate_movement_direction(self, stationary_threshold_px: float) -> None:
        if len(self._center_history) < 3:
            self.movement_direction = MovementDirection.UNKNOWN
            return

        old_center = self._center_history[0]
        new_center = self._center_history[-1]

        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        displacement = np.hypot(dx, dy)

        if displacement < stationary_threshold_px:
            self.movement_direction = MovementDirection.STATIONARY
            return

        if abs(dx) >= abs(dy):
            self.movement_direction = MovementDirection.RIGHT if dx > 0 else MovementDirection.LEFT
        else:
            self.movement_direction = MovementDirection.DOWN if dy > 0 else MovementDirection.UP
