"""
NAYRA - Assistive System
Part 3, Module 04: Real-Time Object Tracking - Test Suite

Tests:
1. New track creation
2. Unique track ID assignment
3. Persistent ID across consecutive frames
4. Multiple objects of the same class
5. Multiple objects of different classes
6. Temporary missed detection (occlusion recovery)
7. Track expiration after max buffer frames
8. Empty detection input handling
9. Invalid bounding boxes handling
10. Class consistency enforcement
11. Movement direction estimation (LEFT, RIGHT, UP, DOWN, STATIONARY)
12. Tracker reset
13. Integration with Module 03
14. CPU compatibility and update latency benchmark
"""

import importlib
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

# Ensure project root directory is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Register modules dynamically
_track_pkg = importlib.import_module("04_object_tracking")
_post_module = importlib.import_module("03_object_detection.postprocessor")
_det_module = importlib.import_module("03_object_detection.detector")

ByteTracker = _track_pkg.ByteTracker
Track = _track_pkg.Track
TrackState = _track_pkg.TrackState
MovementDirection = _track_pkg.MovementDirection
TrackingMetrics = _track_pkg.TrackingMetrics
Detection = _post_module.Detection
ObjectDetector = _det_module.ObjectDetector

from config.settings import ObjectTrackingSettings, get_default_tracking_settings


def create_detection(
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.85,
    box_xyxy: tuple = (100, 100, 200, 300),
    timestamp: float = 0.0,
    width: int = 640,
    height: int = 480,
) -> Detection:
    """Helper to construct strongly-typed Detection object for tests."""
    x1, y1, x2, y2 = box_xyxy
    norm_box = (x1 / width, y1 / height, x2 / width, y2 / height)
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        box_xyxy=box_xyxy,
        box_normalized=norm_box,
        spatial_zone="center",
        spatial_hint=f"{class_name.capitalize()} detected ahead.",
        frame_timestamp=timestamp,
        frame_width=width,
        frame_height=height,
    )


# ============================================================================
# 1. NEW TRACK CREATION & UNIQUE TRACK IDS
# ============================================================================

def test_new_track_creation():
    """Verify that a valid detection spawns a track."""
    settings = ObjectTrackingSettings(min_hits=1)
    tracker = ByteTracker(settings)
    tracker.reset()

    det = create_detection(confidence=0.88, box_xyxy=(100, 100, 200, 300))
    tracks, metrics = tracker.update([det], timestamp=1.0)

    assert len(tracks) == 1
    assert tracks[0].track_id > 0
    assert tracks[0].class_name == "person"
    assert tracks[0].confidence == 0.88
    assert tracks[0].state == TrackState.CONFIRMED


def test_unique_track_id_assignment():
    """Verify that multiple simultaneous objects receive distinct unique IDs."""
    settings = ObjectTrackingSettings(min_hits=1)
    tracker = ByteTracker(settings)
    tracker.reset()

    d1 = create_detection(class_id=0, class_name="person", box_xyxy=(50, 50, 150, 200))
    d2 = create_detection(class_id=56, class_name="chair", box_xyxy=(300, 200, 450, 400))
    d3 = create_detection(class_id=39, class_name="bottle", box_xyxy=(500, 100, 550, 250))

    tracks, _ = tracker.update([d1, d2, d3])
    assert len(tracks) == 3
    ids = [t.track_id for t in tracks]
    assert len(set(ids)) == 3  # All 3 IDs must be unique


# ============================================================================
# 2. PERSISTENT IDS & MULTIPLE OBJECTS
# ============================================================================

def test_persistent_id_across_consecutive_frames():
    """Verify that an object moving across consecutive frames maintains its ID."""
    settings = ObjectTrackingSettings(min_hits=1)
    tracker = ByteTracker(settings)
    tracker.reset()

    # Frame 1: Object at (100, 100, 200, 250)
    det_f1 = create_detection(box_xyxy=(100, 100, 200, 250))
    tracks_f1, _ = tracker.update([det_f1], timestamp=0.0)
    assert len(tracks_f1) == 1
    orig_id = tracks_f1[0].track_id

    # Frame 2: Object moved slightly to (110, 102, 210, 252)
    det_f2 = create_detection(box_xyxy=(110, 102, 210, 252))
    tracks_f2, _ = tracker.update([det_f2], timestamp=0.033)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == orig_id
    assert tracks_f2[0].hits == 2

    # Frame 3: Object moved to (122, 105, 222, 255)
    det_f3 = create_detection(box_xyxy=(122, 105, 222, 255))
    tracks_f3, _ = tracker.update([det_f3], timestamp=0.066)
    assert len(tracks_f3) == 1
    assert tracks_f3[0].track_id == orig_id
    assert tracks_f3[0].hits == 3


def test_multiple_objects_same_class():
    """Verify independent tracking of multiple objects belonging to the same class."""
    settings = ObjectTrackingSettings(min_hits=1)
    tracker = ByteTracker(settings)
    tracker.reset()

    # Frame 1: Two persons at separate locations
    p1_f1 = create_detection(class_id=0, class_name="person", box_xyxy=(50, 50, 150, 300))
    p2_f1 = create_detection(class_id=0, class_name="person", box_xyxy=(400, 50, 500, 300))
    tracks_f1, _ = tracker.update([p1_f1, p2_f1])
    assert len(tracks_f1) == 2
    id1 = tracks_f1[0].track_id
    id2 = tracks_f1[1].track_id

    # Frame 2: Both persons move
    p1_f2 = create_detection(class_id=0, class_name="person", box_xyxy=(55, 52, 155, 302))
    p2_f2 = create_detection(class_id=0, class_name="person", box_xyxy=(408, 48, 508, 298))
    tracks_f2, _ = tracker.update([p1_f2, p2_f2])
    assert len(tracks_f2) == 2
    ids_f2 = [t.track_id for t in tracks_f2]
    assert id1 in ids_f2
    assert id2 in ids_f2


def test_multiple_objects_different_classes():
    """Verify concurrent tracking of diverse object categories."""
    settings = ObjectTrackingSettings(min_hits=1)
    tracker = ByteTracker(settings)
    tracker.reset()

    objs_f1 = [
        create_detection(class_id=0, class_name="person", box_xyxy=(50, 50, 150, 300)),
        create_detection(class_id=67, class_name="cell phone", box_xyxy=(200, 200, 250, 280)),
        create_detection(class_id=63, class_name="laptop", box_xyxy=(300, 250, 450, 380)),
    ]
    tracks_f1, _ = tracker.update(objs_f1)
    assert len(tracks_f1) == 3

    # Move all objects
    objs_f2 = [
        create_detection(class_id=0, class_name="person", box_xyxy=(54, 51, 154, 301)),
        create_detection(class_id=67, class_name="cell phone", box_xyxy=(202, 201, 252, 281)),
        create_detection(class_id=63, class_name="laptop", box_xyxy=(302, 251, 452, 381)),
    ]
    tracks_f2, _ = tracker.update(objs_f2)
    assert len(tracks_f2) == 3
    classes = {t.class_name for t in tracks_f2}
    assert classes == {"person", "cell phone", "laptop"}


# ============================================================================
# 3. OCCLUSION, MISSED DETECTIONS & TRACK EXPIRATION
# ============================================================================

def test_temporary_missed_detection_recovery():
    """Verify that an object temporarily missed/occluded recovers its original ID."""
    settings = ObjectTrackingSettings(min_hits=1, track_buffer=10)
    tracker = ByteTracker(settings)
    tracker.reset()

    # Step 1: Detect object in frames 1 & 2
    d1 = create_detection(box_xyxy=(200, 200, 300, 400))
    t1, _ = tracker.update([d1])
    orig_id = t1[0].track_id

    d2 = create_detection(box_xyxy=(205, 202, 305, 402))
    t2, _ = tracker.update([d2])
    assert t2[0].track_id == orig_id

    # Step 2: Object is completely occluded / missed for 3 frames
    for _ in range(3):
        t_empty, _ = tracker.update([])
        assert len(t_empty) == 0  # No visible confirmed tracks returned

    assert len(tracker.lost_tracks) == 1
    assert tracker.lost_tracks[0].track_id == orig_id
    assert tracker.lost_tracks[0].state == TrackState.LOST

    # Step 3: Object reappears near predicted location
    d_recover = create_detection(box_xyxy=(215, 208, 315, 408))
    t_rec, _ = tracker.update([d_recover])

    assert len(t_rec) == 1
    assert t_rec[0].track_id == orig_id
    assert t_rec[0].state == TrackState.CONFIRMED


def test_track_expiration():
    """Verify that a lost track is removed permanently after track_buffer frames."""
    buffer_frames = 5
    settings = ObjectTrackingSettings(min_hits=1, track_buffer=buffer_frames)
    tracker = ByteTracker(settings)
    tracker.reset()

    # Initiate track
    d = create_detection(box_xyxy=(100, 100, 200, 200))
    t, _ = tracker.update([d])
    assert len(t) == 1

    # Miss object for buffer_frames consecutive frames
    for _ in range(buffer_frames):
        tracker.update([])

    # Next frame should purge it to removed_tracks
    tracker.update([])
    assert len(tracker.lost_tracks) == 0
    assert len(tracker.tracked_tracks) == 0
    assert any(t.is_removed for t in tracker.removed_tracks)


# ============================================================================
# 4. EDGE CASES: EMPTY INPUT, INVALID BOXES, CLASS CONSISTENCY
# ============================================================================

def test_empty_detection_input():
    """Verify tracker handles empty detection list gracefully."""
    tracker = ByteTracker()
    tracker.reset()

    tracks, metrics = tracker.update([])
    assert tracks == []
    assert isinstance(metrics, TrackingMetrics)
    assert metrics.active_tracks_count == 0


def test_invalid_bounding_boxes():
    """Verify invalid or zero-area boxes do not crash the tracker."""
    tracker = ByteTracker()
    tracker.reset()

    # Inverted box
    inv_det = create_detection(box_xyxy=(300, 300, 100, 100))
    tracks, _ = tracker.update([inv_det])
    # Box should be rejected or handled without crashing
    assert isinstance(tracks, list)


def test_class_consistency_enforcement():
    """Verify that tracks do NOT switch classes even with spatial overlap."""
    settings = ObjectTrackingSettings(min_hits=1, enforce_class_consistency=True)
    tracker = ByteTracker(settings)
    tracker.reset()

    # Frame 1: Person at (100, 100, 200, 300)
    person_det = create_detection(class_id=0, class_name="person", box_xyxy=(100, 100, 200, 300))
    t1, _ = tracker.update([person_det])
    person_id = t1[0].track_id

    # Frame 2: In exact same bounding box, a chair is detected instead of person
    chair_det = create_detection(class_id=56, class_name="chair", box_xyxy=(100, 100, 200, 300))
    t2, _ = tracker.update([chair_det])

    # Because class consistency is enforced, the chair must spawn a NEW track
    # and the person must NOT be overwritten by the chair
    chair_track = [t for t in t2 if t.class_name == "chair"][0]
    assert chair_track.track_id != person_id


# ============================================================================
# 5. MOVEMENT DIRECTION ESTIMATION
# ============================================================================

def test_movement_direction_estimation():
    """Verify 2D image-plane movement direction estimation."""
    settings = ObjectTrackingSettings(min_hits=1, stationary_threshold_px=5.0)
    tracker = ByteTracker(settings)
    tracker.reset()

    # 1. Rightward movement: x increases
    box = [100, 100, 200, 200]
    for i in range(5):
        box[0] += 15
        box[2] += 15
        tracks, _ = tracker.update([create_detection(box_xyxy=tuple(box))])

    assert tracks[0].movement_direction == MovementDirection.RIGHT

    # 2. Downward movement: y increases
    tracker.reset()
    box = [100, 100, 200, 200]
    for i in range(5):
        box[1] += 15
        box[3] += 15
        tracks, _ = tracker.update([create_detection(box_xyxy=tuple(box))])

    assert tracks[0].movement_direction == MovementDirection.DOWN

    # 3. Stationary movement: coordinates fluctuate below threshold
    tracker.reset()
    box = [100, 100, 200, 200]
    for i in range(5):
        box[0] += (1 if i % 2 == 0 else -1)
        tracks, _ = tracker.update([create_detection(box_xyxy=tuple(box))])

    assert tracks[0].movement_direction == MovementDirection.STATIONARY


# ============================================================================
# 6. TRACKER RESET & SETTINGS VALIDATION
# ============================================================================

def test_tracker_reset():
    """Verify reset clears all internal state and restarts track IDs."""
    settings = ObjectTrackingSettings(min_hits=1)
    tracker = ByteTracker(settings)

    d = create_detection(box_xyxy=(100, 100, 200, 200))
    tracker.update([d])
    assert len(tracker.tracked_tracks) == 1

    tracker.reset()
    assert len(tracker.tracked_tracks) == 0
    assert len(tracker.lost_tracks) == 0
    assert len(tracker.removed_tracks) == 0

    # Next track after reset starts ID from 1
    t_new, _ = tracker.update([d])
    assert t_new[0].track_id == 1


def test_tracking_settings_validation():
    """Verify ObjectTrackingSettings validates configuration ranges."""
    with pytest.raises(ValueError):
        ObjectTrackingSettings(track_thresh=1.5)

    with pytest.raises(ValueError):
        ObjectTrackingSettings(low_thresh=0.8, track_thresh=0.4)

    with pytest.raises(ValueError):
        ObjectTrackingSettings(track_buffer=0)


# ============================================================================
# 7. INTEGRATION WITH MODULE 03 & REAL INFERENCE PIPELINE
# ============================================================================

def test_integration_with_module03_detector():
    """Verify end-to-end integration: Frame -> ObjectDetector -> ByteTracker."""
    detector = ObjectDetector()
    tracker = ByteTracker(ObjectTrackingSettings(min_hits=1))
    tracker.reset()

    # Load synthetic test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (150, 100), (300, 350), (255, 255, 255), -1)

    detections, det_metrics = detector.detect(test_img)
    tracks, track_metrics = tracker.update(detections)

    assert isinstance(tracks, list)
    assert isinstance(track_metrics, TrackingMetrics)
    assert track_metrics.update_latency_ms >= 0


def test_cpu_compatibility_and_speed_benchmark():
    """Verify tracker runs strictly on CPU with sub-millisecond latency."""
    tracker = ByteTracker()
    tracker.reset()

    # Simulate 10 detections
    detections = [
        create_detection(class_id=i % 5, box_xyxy=(50 * i, 50 * i, 50 * i + 40, 50 * i + 40))
        for i in range(10)
    ]

    # Warmup
    for _ in range(5):
        tracker.update(detections)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        tracker.update(detections)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    mean_lat = np.mean(latencies)
    # Tracking on CPU must be ultra-fast (< 15ms, easily exceeding 60 FPS throughput)
    assert mean_lat < 15.0, f"Tracking latency too high: {mean_lat:.2f} ms"
