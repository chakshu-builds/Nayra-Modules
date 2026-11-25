"""Temporal face tracking, identity stabilization, and speech announcement rate limiting."""

from __future__ import annotations

import collections
import time
from typing import Sequence

import numpy as np

from recognition_types import (
    FaceDetection,
    RecognitionEvent,
    RecognitionStatus,
    RecognizedPerson,
)


class FaceTrack:
    """Tracked face trajectory maintaining temporal identity stability."""

    def __init__(self, track_id: int, detection: FaceDetection, timestamp: float) -> None:
        self.track_id = track_id
        self.box_xyxy = detection.box_xyxy
        self.landmarks = detection.landmarks_5
        self.detection_confidence = detection.confidence

        self.hits = 1
        self.consecutive_missed = 0
        self.last_seen = timestamp

        # Sliding window history of (is_known, identity_id, display_name, score)
        self.history: collections.deque[tuple[bool, str | None, str, float]] = collections.deque(maxlen=7)

    def update(self, detection: FaceDetection, timestamp: float) -> None:
        self.box_xyxy = detection.box_xyxy
        self.landmarks = detection.landmarks_5
        self.detection_confidence = detection.confidence
        self.hits += 1
        self.consecutive_missed = 0
        self.last_seen = timestamp

    def mark_missed(self) -> None:
        self.consecutive_missed += 1

    def add_recognition_observation(
        self,
        is_known: bool,
        identity_id: str | None,
        display_name: str,
        score: float,
    ) -> None:
        self.history.append((is_known, identity_id, display_name, score))

    def get_stabilized_identity(
        self,
        current_is_known: bool,
        current_identity_id: str | None,
        current_name: str,
        current_score: float,
    ) -> tuple[bool, str | None, str, RecognitionStatus]:
        """Aggregate sliding window observations to avoid identity flickering."""
        self.add_recognition_observation(current_is_known, current_identity_id, current_name, current_score)

        if not current_is_known:
            # Check if majority in history is unknown
            known_votes = [obs for obs in self.history if obs[0] and obs[1] is not None]
            if len(known_votes) >= 3 and len(known_votes) > len(self.history) // 2:
                # Majority consensus still favors known identity during brief drop
                best_known = max(known_votes, key=lambda x: x[3])
                return True, best_known[1], best_known[2], RecognitionStatus.RECOGNIZED
            return False, None, "Unknown", RecognitionStatus.UNKNOWN

        # Count votes for candidate identities
        name_counts: dict[str, int] = collections.defaultdict(int)
        for obs in self.history:
            if obs[0] and obs[1] is not None:
                name_counts[obs[2]] += 1

        best_name, count = max(name_counts.items(), key=lambda x: x[1])
        # Require at least 2 positive votes to stabilize recognized name
        if count >= 2:
            return True, current_identity_id, best_name, RecognitionStatus.RECOGNIZED

        return False, None, "Unknown", RecognitionStatus.LOW_CONFIDENCE


class FaceTracker:
    """Maintains temporary face-track IDs across frames and rate-limits assistive announcements."""

    def __init__(
        self,
        iou_threshold: float = 0.40,
        max_lost_frames: int = 15,
        announcement_cooldown_sec: float = 10.0,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.announcement_cooldown_sec = announcement_cooldown_sec

        self._tracks: dict[int, FaceTrack] = {}
        self._next_id: int = 1
        self._last_announced: dict[str, float] = {}  # identifier -> timestamp

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_announced.clear()

    @staticmethod
    def _compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter_area = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area
        return inter_area / float(union_area) if union_area > 0 else 0.0

    def associate_and_stabilize(
        self,
        detections_with_rec: Sequence[tuple[FaceDetection, bool, str | None, str, float, RecognitionStatus]],
        timestamp: float,
    ) -> tuple[list[RecognizedPerson], list[RecognitionEvent]]:
        """Associate detections with tracks, stabilize identities, and emit announcement events."""
        # Pair detections to tracks via greedy IoU matching
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        matches: list[tuple[int, int]] = []
        for det_idx, (det, *_) in enumerate(detections_with_rec):
            best_iou = self.iou_threshold
            best_track_id = None
            for track_id, track in self._tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._compute_iou(det.box_xyxy, track.box_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id is not None:
                matches.append((best_track_id, det_idx))
                matched_track_ids.add(best_track_id)
                matched_det_indices.add(det_idx)

        # Update matched tracks
        for track_id, det_idx in matches:
            det, *_ = detections_with_rec[det_idx]
            self._tracks[track_id].update(det, timestamp)

        # Create new tracks for unmatched detections
        for det_idx, (det, *_) in enumerate(detections_with_rec):
            if det_idx not in matched_det_indices:
                new_track = FaceTrack(self._next_id, det, timestamp)
                self._tracks[self._next_id] = new_track
                matched_track_ids.add(self._next_id)
                self._next_id += 1

        # Mark missed tracks and prune expired ones
        expired_ids = []
        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.mark_missed()
                if track.consecutive_missed >= self.max_lost_frames:
                    expired_ids.append(track_id)
        for tid in expired_ids:
            del self._tracks[tid]

        # Process recognition stability and construct recognized persons
        recognized_persons: list[RecognizedPerson] = []
        events: list[RecognitionEvent] = []

        # Map each active detection to its track
        det_to_track_id: dict[int, int] = {}
        for track_id, det_idx in matches:
            det_to_track_id[det_idx] = track_id
        # Also assign newly created tracks
        for det_idx, _ in enumerate(detections_with_rec):
            if det_idx not in det_to_track_id:
                # Find track with hits == 1 matching box
                det = detections_with_rec[det_idx][0]
                for tid, track in self._tracks.items():
                    if track.box_xyxy == det.box_xyxy and track.hits == 1:
                        det_to_track_id[det_idx] = tid
                        break

        for det_idx, (det, is_known, identity_id, name, score, raw_status) in enumerate(detections_with_rec):
            track_id = det_to_track_id.get(det_idx)
            track = self._tracks.get(track_id) if track_id else None

            if track:
                stab_is_known, stab_id, stab_name, stab_status = track.get_stabilized_identity(
                    is_known, identity_id, name, score
                )
            else:
                stab_is_known, stab_id, stab_name, stab_status = is_known, identity_id, name, raw_status

            person = RecognizedPerson(
                face_track_id=track_id,
                identity_id=stab_id,
                display_name=stab_name,
                is_known=stab_is_known,
                similarity_score=score,
                detection_confidence=det.confidence,
                recognition_status=stab_status,
                bounding_box=det.box_xyxy,
                landmarks=det.landmarks_5,
                timestamp=timestamp,
            )
            recognized_persons.append(person)

            # Speech announcement rate limiting
            announce_key = stab_id if stab_is_known and stab_id else f"unknown_{track_id}"
            last_time = self._last_announced.get(announce_key, 0.0)

            if stab_is_known and stab_status == RecognitionStatus.RECOGNIZED:
                if (timestamp - last_time) >= self.announcement_cooldown_sec:
                    self._last_announced[announce_key] = timestamp
                    events.append(
                        RecognitionEvent(
                            event_type="PERSON_RECOGNIZED",
                            person=person,
                            timestamp=timestamp,
                            message=f"{stab_name} is in front of you.",
                        )
                    )
            elif not stab_is_known and track and track.hits == 5:
                # Announce unknown person once after 5 consecutive confirmed frames
                if (timestamp - last_time) >= self.announcement_cooldown_sec:
                    self._last_announced[announce_key] = timestamp
                    events.append(
                        RecognitionEvent(
                            event_type="UNKNOWN_PERSON_DETECTED",
                            person=person,
                            timestamp=timestamp,
                            message="An unfamiliar person is visible.",
                        )
                    )

        return recognized_persons, events
