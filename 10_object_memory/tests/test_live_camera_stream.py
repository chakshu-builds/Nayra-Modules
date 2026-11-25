"""NAYRA Module 10 — Unit tests for LiveCameraStream, Async DB writes, and Motion Gating."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pytest

from object_memory.perception.live_camera_stream import LiveCameraStream
from object_memory.perception.observation_filter import ObservationFilter
from object_memory.memory_manager import ObjectMemoryManager
from object_memory.db.models import ObjectObservation


def test_observation_filter_motion_detection():
    filter_engine = ObservationFilter()
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)

    # First frame always returns True (initial baseline)
    assert filter_engine.detect_scene_motion(frame1) is True

    # Identical frame returns False (static scene)
    assert filter_engine.detect_scene_motion(frame1) is False

    # Significant change in frame returns True
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    assert filter_engine.detect_scene_motion(frame2) is True


def test_async_database_writes_flush(tmp_path: Path):
    db_file = tmp_path / "test_async_db.db"
    manager = ObjectMemoryManager(db_path=db_file, async_db_writes=True)

    obs = ObjectObservation(
        id=None,
        object_label="cell phone",
        canonical_name="phone",
        timestamp=time.time(),
        confidence=0.9,
        bbox=(100, 100, 150, 150),
    )

    manager._db_write_queue.put(obs)
    # Drain and flush
    manager.flush_db_writes()

    latest = manager.get_latest_observation("phone")
    assert latest is not None
    assert latest.canonical_name == "phone"

    manager.shutdown()


def test_live_camera_stream_synthetic(tmp_path: Path):
    # Verify LiveCameraStream initializes and tracks telemetry correctly
    stream = LiveCameraStream(source=0, connection_timeout_sec=0.5)
    assert stream.is_running is False
    assert stream.capture_fps == 0.0

    # Test thread-safe 1-frame buffer directly
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with stream._lock:
        stream._latest_frame = test_frame
        stream._latest_timestamp = 12345.0

    has_frame, frame_out, ts = stream.read_latest()
    assert has_frame is True
    assert frame_out.shape == (480, 640, 3)
    assert ts == 12345.0
