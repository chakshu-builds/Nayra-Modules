"""NAYRA Module 10 — Observation Filter and Deduplication Pipeline.

Filters transient detections, requires track confirmation, maps relational spatial context,
and suppresses duplicate observations unless significant movement (>15%) or heartbeat elapses.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ..db.models import ObjectObservation
    from .spatial_context import ANCHOR_CLASSES, SpatialContextEngine
except (ImportError, ValueError):
    from db.models import ObjectObservation
    from perception.spatial_context import ANCHOR_CLASSES, SpatialContextEngine

logger = logging.getLogger("NAYRA-ObservationFilter")


class ObservationFilter:
    """Filters, deduplicates, and annotates tracked object observations for memory persistence."""

    def __init__(
        self,
        spatial_engine: Optional[SpatialContextEngine] = None,
        min_hits: int = 3,
        movement_threshold_ratio: float = 0.15,
        heartbeat_interval_sec: float = 60.0,
        save_snapshots: bool = False,
        snapshot_dir: Optional[Path | str] = None,
    ) -> None:
        self.spatial_engine = spatial_engine or SpatialContextEngine()
        self.min_hits = min_hits
        self.movement_threshold_ratio = movement_threshold_ratio
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.save_snapshots = save_snapshots
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else Path("data/memory/snapshots")

        if self.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Cache of last recorded state: track_id -> dict(centroid, timestamp, relation, anchor)
        self._last_recorded_by_track: Dict[int, Dict[str, Any]] = {}
        # Also track by canonical_name to avoid duplicating when track ID changes but object is stationary
        self._last_recorded_by_name: Dict[str, Dict[str, Any]] = {}
        # Scene motion detection thumbnail cache
        self._last_thumbnail: Optional[np.ndarray] = None
        self._last_motion_time: float = 0.0

    def reset(self) -> None:
        """Clear filter internal tracking caches."""
        self._last_recorded_by_track.clear()
        self._last_recorded_by_name.clear()
        self._last_thumbnail = None
        self._last_motion_time = 0.0

    def detect_scene_motion(self, frame: np.ndarray, motion_threshold: float = 1.8) -> bool:
        """Lightweight scene motion evaluation on a downsampled 64x48 thumbnail (< 0.1ms).

        Returns True if motion or scene change is detected; False if scene is static.
        """
        if frame is None or frame.size == 0:
            return False

        try:
            small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if self._last_thumbnail is None:
                self._last_thumbnail = gray
                self._last_motion_time = time.time()
                return True

            diff = cv2.absdiff(gray, self._last_thumbnail)
            mean_diff = float(np.mean(diff))

            if mean_diff >= motion_threshold:
                self._last_thumbnail = gray
                self._last_motion_time = time.time()
                return True

            return False
        except Exception:
            return True

    def process_frame(
        self,
        tracks: Sequence[Any],
        detections: Sequence[Any],
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> List[ObjectObservation]:
        """Process tracks and detections from current frame into persistent ObjectObservations.

        Args:
            tracks: Sequence of Track objects (from ByteTrack or mock)
            detections: Sequence of Detection objects (from ObjectDetector or mock)
            frame: Optional BGR image array for snapshot extraction
            timestamp: Epoch timestamp

        Returns:
            List of new ObjectObservation instances to be saved to DB.
        """
        now = timestamp if timestamp is not None else time.time()
        observations_to_commit: List[ObjectObservation] = []

        h, w = (frame.shape[:2] if frame is not None else (self.spatial_engine.frame_height, self.spatial_engine.frame_width))
        diag = math.hypot(w, h)

        # 1. Identify all anchor furniture and salient landmark objects from raw detections
        SALIENT_ANCHOR_CLASSES = ANCHOR_CLASSES | {"laptop", "tv", "monitor"}
        anchors: List[Tuple[str, Tuple[int, int, int, int]]] = []
        for det in detections:
            class_name = det.class_name.lower()
            if class_name in SALIENT_ANCHOR_CLASSES:
                anchors.append((class_name, det.box_xyxy))

        # 2. Process each track
        for track in tracks:
            # Check confirmation: confirmed track or hits >= min_hits
            is_confirmed = (
                getattr(track, "is_confirmed", False)
                or getattr(track, "hits", 0) >= self.min_hits
            )
            if not is_confirmed:
                continue

            class_name = track.class_name.lower()
            # Skip anchor furniture themselves from being saved as personal items
            if class_name in ANCHOR_CLASSES:
                continue

            canon_name, hi_alias = self.spatial_engine.get_canonical_info(class_name)
            box = track.box_xyxy
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            # 3. Relational context & zones (prevent self-anchoring)
            eligible_anchors = [a for a in anchors if a[0] != class_name and a[0] != canon_name]
            anchor_result = self.spatial_engine.find_best_anchor(box, eligible_anchors)
            if anchor_result:
                anchor_name, rel_type, en_phrase, hi_phrase = anchor_result
            else:
                anchor_name, rel_type = None, None

            zone_code, _, _ = self.spatial_engine.compute_zone(box, width=w)
            vzone_code, _, _ = self.spatial_engine.compute_vertical_zone(box, height=h)

            # 4. Deduplication logic
            track_id = getattr(track, "track_id", 0)
            should_record = self._should_record(
                track_id=track_id,
                canonical_name=canon_name,
                centroid=(cx, cy),
                rel_type=rel_type,
                anchor_name=anchor_name,
                now=now,
                diag=diag,
            )

            if should_record:
                snapshot_path = None
                if self.save_snapshots and frame is not None:
                    snapshot_path = self._save_crop(frame, box, canon_name, now)

                obs = ObjectObservation(
                    id=None,
                    object_label=track.class_name,
                    canonical_name=canon_name,
                    hindi_alias=hi_alias,
                    timestamp=now,
                    confidence=float(track.confidence),
                    bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    track_id=track_id,
                    spatial_relation=rel_type,
                    anchor_object=anchor_name,
                    horizontal_zone=zone_code,
                    vertical_zone=vzone_code,
                    frame_width=w,
                    frame_height=h,
                    snapshot_path=snapshot_path,
                    is_latest=True,
                )

                # Update memory caches
                state_record = {
                    "centroid": (cx, cy),
                    "timestamp": now,
                    "rel_type": rel_type,
                    "anchor_name": anchor_name,
                }
                self._last_recorded_by_track[track_id] = state_record
                self._last_recorded_by_name[canon_name] = state_record

                observations_to_commit.append(obs)
                logger.info(
                    "Generated observation for '%s' (Track %d, relation: %s %s, zone: %s)",
                    canon_name,
                    track_id,
                    rel_type,
                    anchor_name,
                    zone_code,
                )

        return observations_to_commit

    def _should_record(
        self,
        track_id: int,
        canonical_name: str,
        centroid: Tuple[float, float],
        rel_type: Optional[str],
        anchor_name: Optional[str],
        now: float,
        diag: float,
    ) -> bool:
        """Determine whether this observation should be recorded or deduplicated."""
        last_state = self._last_recorded_by_track.get(track_id)
        if not last_state:
            # Check by canonical_name in case track ID reset
            last_state = self._last_recorded_by_name.get(canonical_name)

        if not last_state:
            # First time seeing this object track
            return True

        last_cx, last_cy = last_state["centroid"]
        dist_px = math.hypot(centroid[0] - last_cx, centroid[1] - last_cy)
        shift_ratio = dist_px / max(1.0, diag)

        # Condition 1: Significant movement (> 15% frame diagonal)
        if shift_ratio >= self.movement_threshold_ratio:
            return True

        # Condition 2: Spatial relation or anchor changed
        if rel_type != last_state["rel_type"] or anchor_name != last_state["anchor_name"]:
            return True

        # Condition 3: Heartbeat elapsed (e.g. 60s)
        time_elapsed = now - last_state["timestamp"]
        if time_elapsed >= self.heartbeat_interval_sec:
            return True

        # Otherwise stationary and unchanged: suppress duplicate
        return False

    def _save_crop(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        canonical_name: str,
        now: float,
    ) -> Optional[str]:
        """Crop and save image patch with 10% padding."""
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)

            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w, x2 + pad_x)
            crop_y2 = min(h, y2 + pad_y)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                return None

            filename = f"{canonical_name}_{int(now)}_{int(time.time() * 1000) % 1000}.jpg"
            save_path = self.snapshot_dir / filename
            cv2.imwrite(str(save_path), crop)
            return str(save_path)
        except Exception as exc:
            logger.warning("Failed to save snapshot crop: %s", exc)
            return None
