"""NAYRA Module 10 — Automated Regression Tests for Object Query Accuracy, Dynamic Resolution & Memory Grounding.

Validates:
1. Dynamic Query Resolution & Filler Stripping:
   - TV resolves to 'tv'.
   - Television resolves to 'tv'.
   - TV remote resolves to 'remote' (never 'tv').
   - Laptop resolves to 'laptop'.
   - Conversational filler words do not prevent target extraction.
   - Hindi/Hinglish query parsing remains functional.
   - Unknown targets do not produce fabricated categories.
2. Strict Target Grounding:
   - TV requested + TV detected -> TV response.
   - TV requested + only laptop detected -> TV not visible response (never laptop).
   - TV requested + historical TV observation -> historical TV response.
   - TV requested + no TV history -> not previously seen response.
   - Laptop requested + laptop detected -> laptop response.
   - TV requested + laptop and TV both detected -> TV response only.
   - TV requested + stale laptop memory -> laptop memory is never returned.
3. Memory Correctness & Freshness:
   - No cross-object memory substitution.
   - Freshness checks work (stale observations are never described as currently visible).
   - Historical timestamps are preserved.
   - Current visibility is based on fresh camera verification.
4. Voice Reliability & Acoustic Isolation:
   - Pending audio does not trigger queries after shutdown.
   - Filler-heavy transcripts handled correctly.
5. General Scene Enumeration:
   - 'What can you see?' enumerates currently visible objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Sequence, Tuple
from unittest.mock import MagicMock
import numpy as np
import pytest

from object_memory.db.models import ObjectObservation
from object_memory.memory_manager import ObjectMemoryManager
from object_memory.nlp.query_parser import MemoryQueryParser
from object_memory.nlp.response_generator import ResponseGenerator
from object_memory.voice_guidance import VoiceGuidedObjectFinder


@dataclass
class MockTrack:
    track_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    consecutive_missed_frames: int = 0
    hits: int = 5


class MockTracker:
    def __init__(self, tracks=None, tentative=None):
        self.tracked_tracks = tracks or []
        self.tentative_tracks = tentative or []

    def update(self, detections, timestamp=None):
        return self.tracked_tracks, None


@pytest.fixture
def manager(tmp_path: Path) -> ObjectMemoryManager:
    db_file = tmp_path / "test_grounding.db"
    return ObjectMemoryManager(
        db_path=db_file,
        async_db_writes=False,
        live_freshness_threshold_sec=2.0,
    )


# =========================================================================
# 1. Dynamic Query Resolution & Filler Stripping Tests
# =========================================================================

def test_tv_query_resolution():
    parser = MemoryQueryParser()

    # Exact production case
    q1 = parser.parse("Hey NAYRA, I asked about you where is my TV.")
    assert q1.target_object == "tv"
    assert q1.is_supported is True
    assert q1.query_type == "WHERE"

    # Variations
    q2 = parser.parse("Where is the television?")
    assert q2.target_object == "tv"
    assert q2.is_supported is True

    q3 = parser.parse("Where is my TV?")
    assert q3.target_object == "tv"
    assert q3.is_supported is True


def test_tv_remote_resolves_to_remote_not_tv():
    """Critical specificity requirement: 'TV remote' must resolve to remote, not tv."""
    parser = MemoryQueryParser()

    q1 = parser.parse("Where is the TV remote?")
    assert q1.target_object == "remote"
    assert q1.target_object != "tv"
    assert q1.is_supported is True

    q2 = parser.parse("TV remote kidhar hai?")
    assert q2.target_object == "remote"
    assert q2.is_supported is True

    q3 = parser.parse("NAYRA, where did I leave the television remote?")
    assert q3.target_object == "remote"
    assert q3.is_supported is True


def test_filler_heavy_transcripts():
    parser = MemoryQueryParser()

    q1 = parser.parse("Or where is my laptop?")
    assert q1.target_object == "laptop"
    assert q1.is_supported is True

    q2 = parser.parse("Can you please tell me where did I leave my bottle?")
    assert q2.target_object == "bottle"
    assert q2.is_supported is True

    q3 = parser.parse("Hey NAYRA, tell me where is my dining table?")
    assert q3.target_object == "dining table"
    assert q3.is_supported is True


def test_multilingual_and_unsupported_resolution():
    parser = MemoryQueryParser()

    # Hindi / Hinglish
    q_hi = parser.parse("Mera phone kahan hai?")
    assert q_hi.target_object == "phone"
    assert q_hi.detected_language in ("hi", "hinglish")

    # Unsupported items
    q_keys = parser.parse("Where are my keys?")
    assert q_keys.target_object == "keys"
    assert q_keys.is_supported is False

    q_glasses = parser.parse("Mera chashma kahan hai?")
    assert q_glasses.target_object == "glasses"
    assert q_glasses.is_supported is False


def test_unknown_target_no_fabrication():
    parser = MemoryQueryParser()
    q = parser.parse("what is the weather today outside in Mumbai")
    assert q.target_object is None
    assert q.query_type == "UNKNOWN"


# =========================================================================
# 2. Strict Target Grounding Tests
# =========================================================================

def test_tv_requested_and_tv_detected(manager: ObjectMemoryManager):
    """TV requested + TV detected -> TV response."""
    tv_track = MockTrack(5, "tv", 0.61, (200, 150, 400, 350))
    manager.tracker = MockTracker([tv_track])

    res = manager.query("Hey NAYRA, I asked about you where is my TV.", preferred_language="en")
    assert res.found is True
    assert res.canonical_name == "tv"
    assert res.is_currently_visible is True
    assert "tv" in res.spoken_response_en.lower()
    assert "laptop" not in res.spoken_response_en.lower()


def test_tv_requested_only_laptop_detected_no_substitution(manager: ObjectMemoryManager):
    """TV requested + only laptop detected -> TV not visible response (NEVER laptop!)."""
    laptop_track = MockTrack(1, "laptop", 0.88, (300, 200, 450, 320))
    manager.tracker = MockTracker([laptop_track])

    res = manager.query("Where is my TV?", preferred_language="en")
    assert res.found is False
    assert res.canonical_name == "tv"
    assert res.is_currently_visible is False
    assert "haven't seen your tv yet" in res.spoken_response_en.lower()
    # Must never mention laptop
    assert "laptop" not in res.spoken_response_en.lower()


def test_tv_requested_with_historical_tv_observation(manager: ObjectMemoryManager):
    """TV requested + historical TV observation -> historical TV response."""
    # Live tracker has no TV
    manager.tracker = MockTracker([])

    # Historical TV observation in DB recorded 10 minutes ago on table
    t_seen = time.time() - 600
    obs_tv = ObjectObservation(
        id=None,
        object_label="tv",
        canonical_name="tv",
        hindi_alias="टीवी",
        timestamp=t_seen,
        confidence=0.85,
        bbox=(100, 100, 300, 300),
        spatial_relation="on",
        anchor_object="dining table",
        horizontal_zone="CENTER",
    )
    manager.record_manual_observation(obs_tv)

    res = manager.query("Where is my TV?", preferred_language="en")
    assert res.found is True
    assert res.canonical_name == "tv"
    assert res.is_currently_visible is False
    assert "i saw your tv on the dining table" in res.spoken_response_en.lower()
    assert "not currently visible in the camera" in res.spoken_response_en.lower()


def test_tv_requested_both_tv_and_laptop_detected(manager: ObjectMemoryManager):
    """TV requested + laptop and TV both detected -> TV response only."""
    tv_track = MockTrack(5, "tv", 0.70, (200, 150, 400, 350))
    laptop_track = MockTrack(1, "laptop", 0.90, (50, 50, 150, 150))
    manager.tracker = MockTracker([tv_track, laptop_track])

    res = manager.query("Where is my TV?", preferred_language="en")
    assert res.found is True
    assert res.canonical_name == "tv"
    assert "tv" in res.spoken_response_en.lower()
    assert "laptop" not in res.spoken_response_en.lower()


def test_tv_requested_with_stale_laptop_memory_not_returned(manager: ObjectMemoryManager):
    """TV requested + stale laptop memory -> laptop memory must NEVER be returned."""
    manager.tracker = MockTracker([])

    # Stale laptop in DB
    t_seen = time.time() - 40
    obs_laptop = ObjectObservation(
        id=None,
        object_label="laptop",
        canonical_name="laptop",
        hindi_alias="लैपटॉप",
        timestamp=t_seen,
        confidence=0.92,
        bbox=(200, 200, 400, 400),
        spatial_relation="on",
        anchor_object="dining table",
        horizontal_zone="RIGHT",
    )
    manager.record_manual_observation(obs_laptop)

    res = manager.query("Where is my TV?", preferred_language="en")
    assert res.found is False
    assert res.canonical_name == "tv"
    assert "laptop" not in res.spoken_response_en.lower()


# =========================================================================
# 3. Memory Correctness & Freshness Tests
# =========================================================================

def test_stale_cache_never_described_as_currently_visible(manager: ObjectMemoryManager):
    """A stale observation (age > live_freshness_threshold_sec) is NEVER described as currently visible."""
    # Pre-populate cache with a 36-second old laptop observation
    old_ts = time.time() - 36.4
    old_obs = ObjectObservation(
        id=None,
        object_label="laptop",
        canonical_name="laptop",
        hindi_alias="लैपटॉप",
        timestamp=old_ts,
        confidence=0.90,
        bbox=(350, 200, 500, 350),
        spatial_relation=None,
        anchor_object=None,
        horizontal_zone="RIGHT",
    )
    manager._active_live_observations["laptop"] = old_obs
    manager.record_manual_observation(old_obs)

    # Empty tracker (laptop is not in view right now)
    manager.tracker = MockTracker([])

    res = manager.query("Or where is my laptop?", preferred_language="en")
    assert res.found is True
    assert res.canonical_name == "laptop"
    # MUST NOT be described as currently visible!
    assert res.is_currently_visible is False
    assert "visible on your right side" not in res.spoken_response_en.lower()
    assert "saw your laptop" in res.spoken_response_en.lower()
    assert "not currently visible in the camera" in res.spoken_response_en.lower()


# =========================================================================
# 4. Voice Reliability & Acoustic Isolation Tests
# =========================================================================

def test_no_query_processed_after_shutdown(manager: ObjectMemoryManager):
    """Verify that stopping listener prevents trailing audio from triggering queries."""
    mock_recognizer = MagicMock()
    mock_recognizer.recognize.return_value = ("WHERE", "Where is my TV?", 10.0)

    finder = VoiceGuidedObjectFinder(
        memory_manager=manager,
        voice_recognizer=mock_recognizer,
    )
    # Stop finder
    finder.stop_listening()

    dummy_audio = np.zeros(16000, dtype=np.float32)
    finder._handle_speech_utterance(dummy_audio, 16000)

    # Recognizer should NOT be called because finder is stopped
    assert not mock_recognizer.recognize.called


# =========================================================================
# 5. Scene Enumeration ('What can you see?') Tests
# =========================================================================

def test_scene_enumeration_query(manager: ObjectMemoryManager):
    """General scene questions ('What can you see?') enumerate currently visible detected objects."""
    tv_track = MockTrack(1, "tv", 0.75, (100, 100, 250, 250))
    laptop_track = MockTrack(2, "laptop", 0.88, (300, 200, 450, 320))
    manager.tracker = MockTracker([tv_track, laptop_track])

    res = manager.query("What can you see?", preferred_language="en")
    assert res.found is True
    assert res.canonical_name == "all"
    assert res.is_currently_visible is True
    assert "tv" in res.spoken_response_en.lower()
    assert "laptop" in res.spoken_response_en.lower()
    assert "right now, i can see" in res.spoken_response_en.lower()


def test_scene_enumeration_query_empty(manager: ObjectMemoryManager):
    """When no objects are in view, 'What objects are around me?' states none are recognized."""
    manager.tracker = MockTracker([])

    res = manager.query("What objects are around me?", preferred_language="en")
    assert res.found is False
    assert res.canonical_name == "all"
    assert "cannot see any recognized objects" in res.spoken_response_en.lower()
