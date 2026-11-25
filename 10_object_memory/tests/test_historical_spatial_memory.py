"""NAYRA Module 10 — Historical Object Location Memory & Follow-up Verification Tests.

Comprehensive regression tests covering:
1. Currently visible object returns current spatial position.
2. Historical object query retrieves and speaks last recorded horizontal/vertical location and elapsed time.
3. Historical object with nearby landmark relationship (e.g. near laptop).
4. Historical object with missing spatial metadata honestly discloses that position was not recorded.
5. Unseen object returns honest not-seen response.
6. Target grounding: Old memory of another object (e.g. laptop) never substitutes for requested object (e.g. book).
7. Conversational follow-up: Preserves queried object context across pronoun follow-ups ("Where was it?", "Where was it exactly?").
8. Pronoun follow-up without context asks for clarification.
9. Database schema and persistence: Stores and retrieves vertical_zone, frame_width, frame_height.
10. Low-confidence noise rejection vs high-confidence track confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Optional, Tuple
import pytest

from object_memory.db.database import ObjectMemoryDB
from object_memory.db.models import ObjectObservation
from object_memory.memory_manager import ObjectMemoryManager
from object_memory.nlp.query_parser import MemoryQueryParser
from object_memory.nlp.response_generator import ResponseGenerator
from object_memory.perception.spatial_context import SpatialContextEngine
from object_memory.perception.observation_filter import ObservationFilter


@dataclass
class MockTrack:
    track_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    hits: int = 3
    is_confirmed: bool = True
    consecutive_missed_frames: int = 0


@dataclass
class MockDetection:
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]


class MockTracker:
    def __init__(self, tracks: Optional[List[MockTrack]] = None):
        self.tracked_tracks = tracks or []
        self.tentative_tracks: List[MockTrack] = []

    def update(self, detections, timestamp=None):
        return self.tracked_tracks, []


@pytest.fixture
def temp_manager(tmp_path: Path) -> ObjectMemoryManager:
    db_file = tmp_path / "test_historical_memory.db"
    return ObjectMemoryManager(db_path=db_file)


# =========================================================================
# 1. Spatial Memory & Coordinate Persistence Tests
# =========================================================================

def test_database_stores_and_retrieves_vertical_and_frame_dimensions(tmp_path: Path):
    """Ensure SQLite schema stores and loads vertical_zone, frame_width, frame_height."""
    db_file = tmp_path / "test_dimensions.db"
    db = ObjectMemoryDB(db_path=db_file)

    obs = ObjectObservation(
        id=None,
        object_label="book",
        canonical_name="book",
        hindi_alias="किताब",
        timestamp=time.time() - 40,
        confidence=0.88,
        bbox=(50, 60, 180, 220),
        spatial_relation=None,
        anchor_object=None,
        horizontal_zone="LEFT",
        vertical_zone="UPPER",
        frame_width=1280,
        frame_height=720,
    )

    inserted_id = db.record_observation(obs)
    assert inserted_id > 0

    retrieved = db.get_latest_observation("book")
    assert retrieved is not None
    assert retrieved.canonical_name == "book"
    assert retrieved.horizontal_zone == "LEFT"
    assert retrieved.vertical_zone == "UPPER"
    assert retrieved.frame_width == 1280
    assert retrieved.frame_height == 720


def test_spatial_engine_computes_vertical_zone():
    """Verify compute_vertical_zone divides frame into upper, middle, lower thirds."""
    engine = SpatialContextEngine(frame_width=640, frame_height=480)

    # Upper third (cy < 160)
    vzone_u, en_u, hi_u = engine.compute_vertical_zone((100, 20, 200, 100), height=480)
    assert vzone_u == "UPPER"
    assert "upper" in en_u

    # Middle third (160 <= cy <= 320)
    vzone_m, en_m, hi_m = engine.compute_vertical_zone((100, 200, 200, 280), height=480)
    assert vzone_m == "MIDDLE"
    assert "middle" in en_m

    # Lower third (cy > 320)
    vzone_l, en_l, hi_l = engine.compute_vertical_zone((100, 350, 200, 450), height=480)
    assert vzone_l == "LOWER"
    assert "lower" in en_l


# =========================================================================
# 2. Historical Location Query Response Tests
# =========================================================================

def test_query_visible_book_returns_current_position(temp_manager: ObjectMemoryManager):
    """When book is actively visible in camera, returns current location."""
    # Active book track on the left
    book_track = MockTrack(126, "book", 0.85, (50, 200, 180, 320))
    temp_manager.tracker = MockTracker([book_track])

    res = temp_manager.query("Where is my book?", preferred_language="en")
    assert res.found is True
    assert res.is_currently_visible is True
    assert res.canonical_name == "book"
    assert "visible on your left side" in res.spoken_response_en.lower()


def test_query_historical_book_speaks_last_known_location(temp_manager: ObjectMemoryManager):
    """When book is not visible now, speaks last recorded horizontal location and age."""
    # Camera view has only TV, no book
    tv_track = MockTrack(5, "tv", 0.90, (200, 150, 450, 350))
    temp_manager.tracker = MockTracker([tv_track])

    # Record historical observation 40 seconds ago on the left side
    t_seen = time.time() - 40.0
    obs = ObjectObservation(
        id=None,
        object_label="book",
        canonical_name="book",
        hindi_alias="किताब",
        timestamp=t_seen,
        confidence=0.82,
        bbox=(40, 180, 160, 300),
        spatial_relation=None,
        anchor_object=None,
        horizontal_zone="LEFT",
        vertical_zone="MIDDLE",
    )
    temp_manager.record_manual_observation(obs)

    res = temp_manager.query("Where were my books kept?", preferred_language="en")
    assert res.found is True
    assert res.is_currently_visible is False
    assert res.canonical_name == "book"
    # MUST contain last known location ("left side")
    assert "left side" in res.spoken_response_en.lower()
    # MUST contain elapsed time ("40 seconds ago")
    assert "seconds ago" in res.spoken_response_en.lower()
    # MUST state it is not currently visible
    assert "not currently visible" in res.spoken_response_en.lower()


def test_query_historical_book_near_landmark(temp_manager: ObjectMemoryManager):
    """When book was seen near a landmark (e.g. laptop), speaks relational anchor and zone."""
    temp_manager.tracker = MockTracker([])

    t_seen = time.time() - 35.0
    obs = ObjectObservation(
        id=None,
        object_label="book",
        canonical_name="book",
        hindi_alias="किताब",
        timestamp=t_seen,
        confidence=0.88,
        bbox=(420, 200, 520, 320),
        spatial_relation="near",
        anchor_object="laptop",
        horizontal_zone="RIGHT",
    )
    temp_manager.record_manual_observation(obs)

    res = temp_manager.query("Where is my book?", preferred_language="en")
    assert res.found is True
    assert res.is_currently_visible is False
    assert "near the laptop" in res.spoken_response_en.lower()
    assert "right side" in res.spoken_response_en.lower()
    assert "not currently visible" in res.spoken_response_en.lower()


def test_query_historical_book_without_spatial_data_is_honest(temp_manager: ObjectMemoryManager):
    """When book was recorded without spatial information, honestly discloses it."""
    temp_manager.tracker = MockTracker([])

    t_seen = time.time() - 50.0
    obs = ObjectObservation(
        id=None,
        object_label="book",
        canonical_name="book",
        hindi_alias="किताब",
        timestamp=t_seen,
        confidence=0.75,
        bbox=(0, 0, 0, 0),
        spatial_relation=None,
        anchor_object=None,
        horizontal_zone=None,
    )
    temp_manager.record_manual_observation(obs)

    res = temp_manager.query("Where did I leave my book?", preferred_language="en")
    assert res.found is True
    assert res.is_currently_visible is False
    assert "didn't record its exact position" in res.spoken_response_en.lower()
    assert "not currently visible" in res.spoken_response_en.lower()


def test_unseen_book_returns_honest_unseen(temp_manager: ObjectMemoryManager):
    """When book has never been seen in camera or memory, says so clearly."""
    temp_manager.tracker = MockTracker([])

    res = temp_manager.query("Where is my book?", preferred_language="en")
    assert res.found is False
    assert res.canonical_name == "book"
    assert "haven't seen your book yet" in res.spoken_response_en.lower()


def test_strict_grounding_never_substitutes_tv_or_laptop_for_book(temp_manager: ObjectMemoryManager):
    """When book is queried, active TV track and old laptop memory must never be returned."""
    # Active track is TV
    tv_track = MockTrack(126, "tv", 0.92, (200, 150, 450, 350))
    temp_manager.tracker = MockTracker([tv_track])

    # Old laptop memory in DB
    t_seen = time.time() - 36.0
    obs_laptop = ObjectObservation(
        id=None,
        object_label="laptop",
        canonical_name="laptop",
        hindi_alias="लैपटॉप",
        timestamp=t_seen,
        confidence=0.91,
        bbox=(450, 200, 600, 350),
        spatial_relation=None,
        anchor_object=None,
        horizontal_zone="RIGHT",
    )
    temp_manager.record_manual_observation(obs_laptop)

    # User asks for book
    res = temp_manager.query("Where is my book?", preferred_language="en")
    assert res.found is False
    assert res.canonical_name == "book"
    assert "laptop" not in res.spoken_response_en.lower()
    assert "tv" not in res.spoken_response_en.lower()


# =========================================================================
# 3. Conversational Follow-Up Tests
# =========================================================================

def test_conversational_follow_up_maintains_target(temp_manager: ObjectMemoryManager):
    """User asks 'Where is my book?', then asks 'Where was it exactly?'."""
    temp_manager.tracker = MockTracker([])

    t_seen = time.time() - 40.0
    obs = ObjectObservation(
        id=None,
        object_label="book",
        canonical_name="book",
        hindi_alias="किताब",
        timestamp=t_seen,
        confidence=0.85,
        bbox=(50, 80, 180, 240),
        spatial_relation=None,
        anchor_object=None,
        horizontal_zone="LEFT",
        vertical_zone="UPPER",
    )
    temp_manager.record_manual_observation(obs)

    # First query
    res1 = temp_manager.query("Where is my book?", preferred_language="en")
    assert res1.canonical_name == "book"
    assert "left side" in res1.spoken_response_en.lower()

    # Follow-up query using pronoun and exact request
    res2 = temp_manager.query("Where was it exactly?", preferred_language="en")
    assert res2.canonical_name == "book"
    assert "book" in res2.spoken_response_en.lower()
    assert "upper" in res2.spoken_response_en.lower() or "left" in res2.spoken_response_en.lower()
    assert "not currently visible" in res2.spoken_response_en.lower()


def test_follow_up_pronoun_without_context_asks_clarification(temp_manager: ObjectMemoryManager):
    """Asking 'Where was it?' with no prior query asks user to specify the object."""
    temp_manager._last_queried_target = None

    res = temp_manager.query("Where was it?", preferred_language="en")
    assert res.found is False
    assert "not sure which object you mean" in res.spoken_response_en.lower()


# =========================================================================
# 4. Track Confirmation & Noise Rejection
# =========================================================================

def test_noise_detection_below_threshold_ignored():
    """Unstable detector noise (conf < 0.30) is not registered."""
    filter_engine = ObservationFilter(min_hits=2)
    mock_noise = MockTrack(1, "book", 0.20, (10, 10, 50, 50), hits=1, is_confirmed=False)
    detections = [MockDetection("book", 0.20, (10, 10, 50, 50))]

    observations = filter_engine.process_frame([mock_noise], detections)
    assert len(observations) == 0
