"""NAYRA Module 10 — Unit tests for Observation Filter and Deduplication."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Tuple
import pytest

from object_memory.perception.spatial_context import SpatialContextEngine
from object_memory.perception.observation_filter import ObservationFilter


@dataclass
class MockDetection:
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]


@dataclass
class MockTrack:
    track_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    hits: int = 4
    is_confirmed: bool = True


def test_filter_tentative_tracks():
    filter_engine = ObservationFilter(min_hits=3)

    # Track with only 1 hit (unconfirmed)
    tentative_track = MockTrack(
        track_id=1,
        class_name="cell phone",
        confidence=0.85,
        box_xyxy=(100, 100, 150, 150),
        hits=1,
        is_confirmed=False,
    )
    detections = [MockDetection("cell phone", 0.85, (100, 100, 150, 150))]

    observations = filter_engine.process_frame([tentative_track], detections)
    assert len(observations) == 0  # Should be filtered out


def test_filter_confirmed_track_with_anchor():
    filter_engine = ObservationFilter(min_hits=3)

    # Confirmed track
    phone_track = MockTrack(
        track_id=1,
        class_name="cell phone",
        confidence=0.90,
        box_xyxy=(200, 220, 280, 270),
        hits=5,
        is_confirmed=True,
    )
    table_det = MockDetection("dining table", 0.95, (100, 250, 500, 450))

    observations = filter_engine.process_frame([phone_track], [table_det])
    assert len(observations) == 1
    obs = observations[0]
    assert obs.canonical_name == "phone"
    assert obs.anchor_object == "dining table"
    assert obs.spatial_relation == "on"
    assert obs.track_id == 1


def test_deduplication_suppresses_stationary_object():
    filter_engine = ObservationFilter(min_hits=3, heartbeat_interval_sec=60.0)

    t0 = 1000.0
    phone_track = MockTrack(
        track_id=2,
        class_name="cell phone",
        confidence=0.90,
        box_xyxy=(200, 220, 280, 270),
        hits=5,
        is_confirmed=True,
    )
    table_det = MockDetection("dining table", 0.95, (100, 250, 500, 450))

    # Frame 1 at t0 -> recorded
    obs1 = filter_engine.process_frame([phone_track], [table_det], timestamp=t0)
    assert len(obs1) == 1

    # Frame 2 at t0 + 2 seconds (stationary) -> suppressed
    obs2 = filter_engine.process_frame([phone_track], [table_det], timestamp=t0 + 2.0)
    assert len(obs2) == 0

    # Frame 3 at t0 + 30 seconds (still stationary) -> suppressed
    obs3 = filter_engine.process_frame([phone_track], [table_det], timestamp=t0 + 30.0)
    assert len(obs3) == 0

    # Frame 4 at t0 + 61 seconds (heartbeat expired) -> recorded
    obs4 = filter_engine.process_frame([phone_track], [table_det], timestamp=t0 + 61.0)
    assert len(obs4) == 1


def test_deduplication_records_on_significant_movement():
    filter_engine = ObservationFilter(min_hits=3, movement_threshold_ratio=0.15)

    t0 = 1000.0
    phone_track1 = MockTrack(
        track_id=3,
        class_name="cell phone",
        confidence=0.90,
        box_xyxy=(100, 100, 160, 160),
        hits=5,
        is_confirmed=True,
    )
    obs1 = filter_engine.process_frame([phone_track1], [], timestamp=t0)
    assert len(obs1) == 1

    # Moved across the screen to (450, 400)
    phone_track2 = MockTrack(
        track_id=3,
        class_name="cell phone",
        confidence=0.90,
        box_xyxy=(450, 400, 510, 460),
        hits=6,
        is_confirmed=True,
    )
    obs2 = filter_engine.process_frame([phone_track2], [], timestamp=t0 + 1.0)
    assert len(obs2) == 1
    assert obs2[0].bbox == (450, 400, 510, 460)
