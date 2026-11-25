"""ByteTrack multi-object tracking implementation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config.settings import ObjectTrackingSettings, get_default_tracking_settings
from .kalman_filter import KalmanBoxTracker
from .track import MovementDirection, Track, TrackState

logger = logging.getLogger(__name__)


@dataclass
class TrackingMetrics:
    """Performance telemetry for the tracking pipeline."""

    update_latency_ms: float = 0.0
    active_tracks_count: int = 0
    confirmed_tracks_count: int = 0
    lost_tracks_count: int = 0
    tracking_fps: float = 0.0


class ByteTracker:
    """Multi-Object Tracker based on the ByteTrack association algorithm."""

    def __init__(self, settings: ObjectTrackingSettings | None = None) -> None:
        self.settings = settings or get_default_tracking_settings()
        self.kalman_tracker = KalmanBoxTracker()

        self.tracked_tracks: list[Track] = []
        self.tentative_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.removed_tracks: list[Track] = []

        self.frame_id: int = 0
        self._fps_ema: float = 0.0
        self._ema_alpha: float = 0.15

        logger.info(
            "ByteTracker initialized. Track thresh: %.2f | Low thresh: %.2f | Buffer: %d frames",
            self.settings.track_thresh,
            self.settings.low_thresh,
            self.settings.track_buffer,
        )

    def reset(self) -> None:
        """Clear all active tracks and reset track counter."""
        self.tracked_tracks.clear()
        self.tentative_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_id = 0
        self._fps_ema = 0.0
        Track.reset_id_counter()
        logger.info("ByteTracker reset: all tracks and IDs cleared.")

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute pairwise IoU matrix between two sets of (x1, y1, x2, y2) boxes."""
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

        x1_a, y1_a, x2_a, y2_a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        x1_b, y1_b, x2_b, y2_b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        area_a = np.maximum(0.0, x2_a - x1_a) * np.maximum(0.0, y2_a - y1_a)
        area_b = np.maximum(0.0, x2_b - x1_b) * np.maximum(0.0, y2_b - y1_b)

        inter_x1 = np.maximum(x1_a[:, None], x1_b[None, :])
        inter_y1 = np.maximum(y1_a[:, None], y1_b[None, :])
        inter_x2 = np.minimum(x2_a[:, None], x2_b[None, :])
        inter_y2 = np.minimum(y2_a[:, None], y2_b[None, :])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = area_a[:, None] + area_b[None, :] - inter_area
        return np.where(union_area > 0.0, inter_area / union_area, 0.0)

    def _compute_cost_matrix(
        self,
        tracks: Sequence[Track],
        detections: Sequence[Any],
    ) -> np.ndarray:
        """Compute cost matrix (1 - IoU) with class consistency penalty."""
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        track_boxes = np.array([t.box_xyxy for t in tracks], dtype=np.float32)
        det_boxes = np.array([d.box_xyxy for d in detections], dtype=np.float32)

        cost_matrix = 1.0 - self._compute_iou_matrix(track_boxes, det_boxes)

        if self.settings.enforce_class_consistency:
            for i, t in enumerate(tracks):
                for j, d in enumerate(detections):
                    if t.class_id != d.class_id:
                        cost_matrix[i, j] = 1e5

        return cost_matrix

    @staticmethod
    def _linear_assignment(
        cost_matrix: np.ndarray, thresh: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Perform optimal bipartite matching using the Hungarian algorithm."""
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched: list[tuple[int, int]] = []
        unmatched_rows = set(range(cost_matrix.shape[0]))
        unmatched_cols = set(range(cost_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matched.append((int(r), int(c)))
                unmatched_rows.discard(int(r))
                unmatched_cols.discard(int(c))

        return matched, sorted(list(unmatched_rows)), sorted(list(unmatched_cols))

    def update(
        self,
        detections: Sequence[Any],
        timestamp: float | None = None,
    ) -> tuple[list[Track], TrackingMetrics]:
        """Update tracker state with detections from current video frame."""
        t0 = time.perf_counter()
        self.frame_id += 1
        if timestamp is None:
            timestamp = time.time()

        for t in self.tracked_tracks:
            t.predict()
        for t in self.lost_tracks:
            t.predict()
        for t in self.tentative_tracks:
            t.predict()

        dets_high = [d for d in detections if d.confidence >= self.settings.track_thresh]
        dets_low = [
            d for d in detections
            if self.settings.low_thresh <= d.confidence < self.settings.track_thresh
        ]

        # Stage 1: match confirmed/lost tracks with high-confidence detections
        pool_tracks = self.tracked_tracks + self.lost_tracks
        cost_matrix = self._compute_cost_matrix(pool_tracks, dets_high)
        matches_1, unmatched_tracks_1_idx, unmatched_dets_1_idx = self._linear_assignment(
            cost_matrix, self.settings.match_thresh
        )

        matched_tracks_1: list[Track] = []
        for t_idx, d_idx in matches_1:
            track = pool_tracks[t_idx]
            track.update(dets_high[d_idx], timestamp, self.settings.stationary_threshold_px)
            if track.state in (TrackState.TENTATIVE, TrackState.LOST):
                if track.state == TrackState.LOST or track.hits >= self.settings.min_hits:
                    track.confirm()
            matched_tracks_1.append(track)

        # Stage 2: match remaining confirmed tracks with low-confidence detections
        unmatched_tracked_and_lost = [pool_tracks[i] for i in unmatched_tracks_1_idx]
        candidate_rescue_tracks = [
            t for t in unmatched_tracked_and_lost if t.state in (TrackState.CONFIRMED, TrackState.LOST)
        ]

        cost_matrix_2 = self._compute_cost_matrix(candidate_rescue_tracks, dets_low)
        matches_2, _, _ = self._linear_assignment(cost_matrix_2, self.settings.match_thresh_second)

        matched_tracks_2: list[Track] = []
        for t_idx, d_idx in matches_2:
            track = candidate_rescue_tracks[t_idx]
            track.update(dets_low[d_idx], timestamp, self.settings.stationary_threshold_px)
            if track.state == TrackState.LOST:
                track.confirm()
            matched_tracks_2.append(track)

        all_updated = set(matched_tracks_1 + matched_tracks_2)
        all_prior = set(self.tracked_tracks + self.lost_tracks + self.tentative_tracks)
        for track in all_prior - all_updated:
            track.mark_missed(max_lost_frames=self.settings.track_buffer)

        # Stage 3: match remaining high-confidence detections with tentative tracks
        unmatched_dets_high = [dets_high[i] for i in unmatched_dets_1_idx]
        cost_matrix_tentative = self._compute_cost_matrix(self.tentative_tracks, unmatched_dets_high)
        matches_tent, _, unmatched_dets_tent_idx = self._linear_assignment(
            cost_matrix_tentative, self.settings.match_thresh
        )

        for t_idx, d_idx in matches_tent:
            t = self.tentative_tracks[t_idx]
            t.update(unmatched_dets_high[d_idx], timestamp, self.settings.stationary_threshold_px)
            if t.hits >= self.settings.min_hits:
                t.confirm()

        # Initialize new tracks for unmatched high-confidence detections
        brand_new_dets = [unmatched_dets_high[i] for i in unmatched_dets_tent_idx]
        new_tracks: list[Track] = []
        for d in brand_new_dets:
            new_t = Track(d, self.kalman_tracker, timestamp)
            if self.settings.min_hits <= 1:
                new_t.confirm()
            new_tracks.append(new_t)

        full_pool = list(all_prior | set(new_tracks))
        self.tracked_tracks = [t for t in full_pool if t.is_confirmed and not t.is_lost and not t.is_removed]
        self.tentative_tracks = [t for t in full_pool if t.state == TrackState.TENTATIVE and not t.is_removed]
        self.lost_tracks = [t for t in full_pool if t.is_lost and not t.is_removed]
        self.removed_tracks.extend([t for t in full_pool if t.is_removed])
        if len(self.removed_tracks) > 200:
            self.removed_tracks = self.removed_tracks[-100:]

        lat_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / lat_ms if lat_ms > 0 else 0.0
        self._fps_ema = fps if self._fps_ema == 0.0 else (self._ema_alpha * fps) + ((1.0 - self._ema_alpha) * self._fps_ema)

        visible_tracks = [t for t in self.tracked_tracks if t.consecutive_missed_frames == 0]

        metrics = TrackingMetrics(
            update_latency_ms=round(lat_ms, 2),
            active_tracks_count=len(self.tracked_tracks),
            confirmed_tracks_count=len(visible_tracks),
            lost_tracks_count=len(self.lost_tracks),
            tracking_fps=round(self._fps_ema, 1),
        )

        return visible_tracks, metrics

    @staticmethod
    def get_track_color(track_id: int) -> tuple[int, int, int]:
        """Generate a distinct, consistent BGR color from track ID."""
        palette = [
            (50, 220, 50),
            (255, 178, 50),
            (50, 150, 255),
            (255, 100, 200),
            (0, 230, 230),
            (200, 130, 255),
            (100, 255, 100),
            (80, 200, 255),
            (255, 220, 100),
            (180, 180, 50),
        ]
        return palette[(track_id - 1) % len(palette)]

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        metrics: TrackingMetrics | None = None,
        draw_hud: bool = True,
    ) -> np.ndarray:
        """Render tracked bounding boxes, track IDs, movement directions, and telemetry onto frame."""
        output = frame.copy()
        h, w = output.shape[:2]

        for t in tracks:
            x1, y1, x2, y2 = t.box_xyxy
            color = self.get_track_color(t.track_id)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            dir_str = f" [{t.movement_direction.value.upper()}]" if t.movement_direction != MovementDirection.UNKNOWN else ""
            conf_pct = int(round(t.confidence * 100))
            label_text = f"ID: {t.track_id} | {t.class_name.capitalize()} {conf_pct}%{dir_str}"

            (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            banner_y1 = max(0, y1 - lh - 8)
            banner_y2 = y1
            cv2.rectangle(output, (x1, banner_y1), (x1 + lw + 8, banner_y2), color, -1)
            cv2.putText(
                output,
                label_text,
                (x1 + 4, banner_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            cx, cy = int(round(t.center[0])), int(round(t.center[1]))
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)

        if draw_hud and metrics is not None:
            hud_lines = [
                "Tracker: ByteTrack (CPU)",
                f"Track Latency: {metrics.update_latency_ms:.2f} ms ({metrics.tracking_fps:.1f} FPS)",
                f"Active Tracks: {metrics.confirmed_tracks_count} confirmed | {metrics.lost_tracks_count} lost",
            ]
            hud_y = h - 20 - (len(hud_lines) * 20)
            cv2.rectangle(output, (w - 360, hud_y - 6), (w - 10, h - 15), (20, 20, 20), -1)
            cv2.rectangle(output, (w - 360, hud_y - 6), (w - 10, h - 15), (50, 200, 50), 1)

            for i, line in enumerate(hud_lines):
                cv2.putText(
                    output,
                    line,
                    (w - 350, hud_y + 14 + (i * 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (50, 255, 50),
                    1,
                    cv2.LINE_AA,
                )

        return output
