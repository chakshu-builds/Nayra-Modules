"""NAYRA Module 10.1 — Unit tests for Voice-Guided Object Finding and Live Spatial Guidance.

Validates:
1. Current live visibility (object in camera view right now).
2. Actionable spatial positioning (furniture anchor + left/center/right directional guidance).
3. Historical fallback (object seen previously in DB, not currently visible in camera).
4. Unobserved object reporting.
5. Unsupported object transparent refusal (chashma, keys).
6. Multilingual natural spoken responses (Hindi, Hinglish, English).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Tuple
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
    def __init__(self, tracks=None):
        self.tracked_tracks = tracks or []

    def update(self, detections, timestamp=None):
        return self.tracked_tracks, None


@pytest.fixture
def manager(tmp_path: Path) -> ObjectMemoryManager:
    db_file = tmp_path / "test_voice_guidance.db"
    return ObjectMemoryManager(db_path=db_file, async_db_writes=False)


# =========================================================================
# 1. Currently Visible Tests
# =========================================================================

def test_live_visible_phone_on_table_center(manager: ObjectMemoryManager):
    """User: 'Mera phone kahan hai?' -> 'Aapka phone saamne table par hai.'"""
    # Phone resting on table in center of 640x480 frame
    # Center zone is 211 to 422
    phone_track = MockTrack(1, "cell phone", 0.92, (250, 220, 310, 270))
    table_track = MockTrack(2, "dining table", 0.95, (100, 250, 500, 450))

    manager.tracker = MockTracker([phone_track, table_track])

    res = manager.query("Mera phone kahan hai?")
    assert res.found is True
    assert res.is_currently_visible is True
    assert "saamne table par hai" in res.spoken_response_hi.lower() or "table par hai" in res.spoken_response_hi.lower()
    assert "on the dining table" in res.spoken_response_en.lower() or "on the table" in res.spoken_response_en.lower()


def test_live_visible_remote_near_sofa_left(manager: ObjectMemoryManager):
    """User: 'Remote kahan hai?' -> 'Remote sofa ke paas, aapki left side mein hai.'"""
    # Remote on the left side (x < 211) near sofa (sofa is at x=120..300)
    remote_track = MockTrack(3, "remote", 0.89, (50, 180, 100, 230))
    sofa_track = MockTrack(4, "couch", 0.94, (120, 150, 300, 380))

    manager.tracker = MockTracker([remote_track, sofa_track])

    res = manager.query("Remote kahan hai?")
    assert res.found is True
    assert res.is_currently_visible is True
    assert "sofa ke paas" in res.spoken_response_hi.lower()
    assert "left side mein" in res.spoken_response_hi.lower() or "baayein" in res.spoken_response_hi.lower()
    assert "near the couch" in res.spoken_response_en.lower() or "near the sofa" in res.spoken_response_en.lower()
    assert "left" in res.spoken_response_en.lower()


def test_live_visible_bottle_on_right_no_anchor(manager: ObjectMemoryManager):
    """User: 'Meri bottle kahan hai?' -> 'Bottle aapke right side mein dikh rahi hai.'"""
    # Bottle on the right side (x > 422) without furniture anchor
    bottle_track = MockTrack(5, "bottle", 0.91, (480, 150, 530, 280))

    manager.tracker = MockTracker([bottle_track])

    res = manager.query("Meri bottle kahan hai?")
    assert res.found is True
    assert res.is_currently_visible is True
    assert "right side mein" in res.spoken_response_hi.lower() or "daayein" in res.spoken_response_hi.lower()
    assert "dikh rahi hai" in res.spoken_response_hi.lower()
    assert "right side" in res.spoken_response_en.lower()


# =========================================================================
# 2. Historical Fallback Tests (Not Currently Visible in Camera)
# =========================================================================

def test_historical_phone_not_currently_visible(manager: ObjectMemoryManager):
    """User: 'Mera phone kahan hai?' -> 'Maine aapka phone kuch der pehle table par dekha tha. Abhi camera mein nahi dikh raha.'"""
    # Tracker has NO phone currently visible
    manager.tracker = MockTracker([])

    # But database has an observation recorded 5 minutes ago on dining table
    t_seen = time.time() - 300
    obs = ObjectObservation(
        id=None,
        object_label="cell phone",
        canonical_name="phone",
        hindi_alias="फ़ोन",
        timestamp=t_seen,
        confidence=0.90,
        bbox=(200, 220, 270, 270),
        spatial_relation="on",
        anchor_object="dining table",
        horizontal_zone="CENTER",
    )
    manager.record_manual_observation(obs)

    res = manager.query("Mera phone kahan hai?")
    assert res.found is True
    assert res.is_currently_visible is False
    assert "dekha tha" in res.spoken_response_hi.lower()
    assert "table par" in res.spoken_response_hi.lower()
    assert "abhi camera mein nahi dikh raha" in res.spoken_response_hi.lower()
    assert "not currently visible in the camera" in res.spoken_response_en.lower()


def test_historical_remote_seen_earlier(manager: ObjectMemoryManager):
    """Historical observation seen 1 hour ago near sofa."""
    manager.tracker = MockTracker([])

    t_seen = time.time() - 3600
    obs = ObjectObservation(
        id=None,
        object_label="remote",
        canonical_name="remote",
        hindi_alias="रिमोट",
        timestamp=t_seen,
        confidence=0.88,
        bbox=(80, 150, 130, 200),
        spatial_relation="near",
        anchor_object="couch",
        horizontal_zone="LEFT",
    )
    manager.record_manual_observation(obs)

    res = manager.query("Where is the TV remote?")
    assert res.found is True
    assert res.is_currently_visible is False
    assert "1 ghante pehle" in res.spoken_response_hi.lower() or "ghante" in res.spoken_response_hi.lower()
    assert "sofa ke paas" in res.spoken_response_hi.lower() or "couch" in res.spoken_response_hi.lower()
    assert "abhi camera mein nahi dikh raha" in res.spoken_response_hi.lower()
    assert "not currently visible in the camera" in res.spoken_response_en.lower()


# =========================================================================
# 3. Unobserved & Unsupported Tests
# =========================================================================

def test_unobserved_object_not_in_memory(manager: ObjectMemoryManager):
    """Never-observed object returns honest not-found guidance."""
    manager.tracker = MockTracker([])

    res = manager.query("Where is my book?")
    assert res.found is False
    assert res.is_currently_visible is False
    assert "dikhai nahi diya" in res.spoken_response_hi.lower()
    assert "abhi camera mein bhi nahi dikh raha" in res.spoken_response_hi.lower()
    assert "not currently visible in the camera" in res.spoken_response_en.lower()


def test_unsupported_chashma(manager: ObjectMemoryManager):
    """Chashma (out of vocabulary) is transparently declined without hallucination."""
    res = manager.query("Mera chashma kahan hai?")
    assert res.found is False
    assert res.canonical_name == "glasses"
    assert "saksham nahi" in res.spoken_response_hi.lower()
    assert "cannot currently recognize" in res.spoken_response_en.lower()


def test_unsupported_keys(manager: ObjectMemoryManager):
    """Keys (out of vocabulary) are transparently declined without hallucination."""
    res = manager.query("Meri keys kahan rakhi thi?")
    assert res.found is False
    assert res.canonical_name == "keys"
    assert "saksham nahi" in res.spoken_response_hi.lower()
    assert "cannot currently recognize" in res.spoken_response_en.lower()


# =========================================================================
# 4. Multilingual Variations (Hindi, English, Hinglish)
# =========================================================================

def test_multilingual_query_handling(manager: ObjectMemoryManager):
    parser = MemoryQueryParser()

    # Hindi
    q_hi = parser.parse("मेरा फ़ोन कहाँ है?")
    assert q_hi.target_object == "phone"
    assert q_hi.detected_language == "hi"

    # Hinglish
    q_hinglish = parser.parse("Mera phone kahan rakha hai?")
    assert q_hinglish.target_object == "phone"
    assert q_hinglish.detected_language == "hinglish"

    # English
    q_en = parser.parse("Where did I leave my bottle?")
    assert q_en.target_object == "bottle"
    assert q_en.detected_language == "en"


# =========================================================================
# 5. Acoustic Echo Guard & Instant Candidate Visibility Tests
# =========================================================================

def test_acoustic_feedback_shield(manager: ObjectMemoryManager):
    """VoiceGuidedObjectFinder suppresses utterances captured during TTS playback and post-TTS cooldown."""
    mock_tts = MagicMock()
    mock_tts.is_speaking.return_value = True

    finder = VoiceGuidedObjectFinder(memory_manager=manager, tts_manager=mock_tts, feedback_cooldown_sec=1.0)
    finder.recognizer = MagicMock()
    dummy_audio = np.zeros(16000, dtype=np.float32)

    # 1. Speaking active -> utterance dropped
    finder._handle_speech_utterance(dummy_audio, 16000)
    assert not finder.recognizer.recognize.called

    # 2. Cooldown active -> utterance dropped
    mock_tts.is_speaking.return_value = False
    finder._last_tts_end_time = time.time() - 0.2  # 0.2s elapsed (< 1.0s cooldown)
    finder._handle_speech_utterance(dummy_audio, 16000)
    assert not finder.recognizer.recognize.called

    # 3. Cooldown expired -> utterance processed
    finder._last_tts_end_time = time.time() - 1.5  # 1.5s elapsed (> 1.0s cooldown)
    finder.recognizer.recognize.return_value = ("WHERE", "Where is my phone?", 100.0)
    finder._handle_speech_utterance(dummy_audio, 16000)
    assert finder.recognizer.recognize.called


def test_tentative_track_instant_visibility(manager: ObjectMemoryManager):
    """Objects in tentative tracks (hits < min_hits) are recognized as currently visible."""
    tentative_track = MockTrack(
        track_id=10,
        class_name="cell phone",
        confidence=0.89,
        box_xyxy=(200, 150, 300, 350),
        hits=1,  # Frame 1 tentative track
    )
    # Tracker has empty tracked_tracks, but has tentative track
    class TentativeTracker:
        def __init__(self, t):
            self.tracked_tracks = []
            self.tentative_tracks = [t]
        def update(self, *args, **kwargs):
            return [], MagicMock()

    manager.tracker = TentativeTracker(tentative_track)
    obs = manager.get_currently_visible_observation("phone")
    assert obs is not None
    assert obs.canonical_name == "phone"
    assert obs.confidence == 0.89
    assert obs.horizontal_zone == "CENTER"


# =========================================================================
# 6. Module 10 English Voice Reliability & Query Parser Verification
# =========================================================================

def test_five_core_english_queries(manager: ObjectMemoryManager):
    """Test 5 core user queries:
    1. 'Where is my laptop?'
    2. 'Where is my phone?'
    3. 'Where is the TV remote?'
    4. 'Where did I last see my book?'
    5. 'Where are my keys?' (unsupported)
    """
    parser = MemoryQueryParser()

    # 1. Laptop
    q_laptop = parser.parse("Where is my laptop?")
    assert q_laptop.target_object == "laptop"
    assert q_laptop.is_supported is True
    res_laptop = manager.query("Where is my laptop?", preferred_language="en")
    assert res_laptop.canonical_name == "laptop"

    # 2. Phone
    q_phone = parser.parse("Where is my phone?")
    assert q_phone.target_object == "phone"
    assert q_phone.is_supported is True
    res_phone = manager.query("Where is my phone?", preferred_language="en")
    assert res_phone.canonical_name == "phone"

    # 3. TV Remote
    q_remote = parser.parse("Where is the TV remote?")
    assert q_remote.target_object == "remote"
    assert q_remote.is_supported is True
    res_remote = manager.query("Where is the TV remote?", preferred_language="en")
    assert res_remote.canonical_name == "remote"

    # 4. Book
    q_book = parser.parse("Where did I last see my book?")
    assert q_book.target_object == "book"
    assert q_book.is_supported is True
    res_book = manager.query("Where did I last see my book?", preferred_language="en")
    assert res_book.canonical_name == "book"

    # 5. Keys (transparent refusal, unsupported)
    q_keys = parser.parse("Where are my keys?")
    assert q_keys.target_object == "keys"
    assert q_keys.is_supported is False
    res_keys = manager.query("Where are my keys?", preferred_language="en")
    assert res_keys.found is False
    assert res_keys.canonical_name == "keys"
    assert "cannot currently recognize" in res_keys.spoken_response_en.lower()


def test_garbled_query_returns_honest_response(manager: ObjectMemoryManager):
    """Ensure garbled/unrecognized queries do not fabricate 'I haven't seen your object yet'."""
    generator = ResponseGenerator()
    parser = MemoryQueryParser()

    q_garbled = parser.parse("something completely unrelated and nonsensical 12345")
    assert q_garbled.target_object is None

    res = generator.generate_response(q_garbled, obs=None, is_currently_visible=False)
    assert res.found is False
    assert res.canonical_name == "unknown"
    assert "could not identify the object you are looking for" in res.spoken_response_en.lower()
    assert "object" not in res.spoken_response_en.lower().replace("could not identify the object", "")


def test_forced_english_voice_guidance_tts(manager: ObjectMemoryManager):
    """Verify that process_voice_query forces preferred_language='en' and dispatches English TTS."""
    mock_tts = MagicMock()
    mock_recognizer = MagicMock()
    mock_recognizer.recognize.return_value = ("WHERE", "Where is my laptop?", 12.0)

    finder = VoiceGuidedObjectFinder(
        memory_manager=manager,
        voice_recognizer=mock_recognizer,
        tts_manager=mock_tts,
    )

    audio_dummy = np.zeros(16000, dtype=np.float32)
    result = finder.process_voice_query(audio_dummy, sample_rate=16000)

    assert result is not None
    assert result.canonical_name == "laptop"
    # Verify mock TTS was called with English text
    assert mock_tts.speak.called
    args, kwargs = mock_tts.speak.call_args
    spoken_text = args[0] if args else kwargs.get("text", "")
    assert spoken_text == result.spoken_response_en
    if "language_hint" in kwargs:
        assert kwargs["language_hint"] == "en"


def test_whisper_direct_model_english_only(manager: ObjectMemoryManager):
    """Verify that _recognize_with_fallback configures Whisper with language='en' and no 2nd pass."""
    from object_memory.voice_guidance import ENGLISH_OBJECT_PROMPT_BIAS

    mock_whisper = MagicMock()
    mock_seg = MagicMock()
    mock_seg.text = "Where is my phone?"
    mock_info = MagicMock()
    mock_info.language = "th"  # Simulate Whisper hallucinating Thai
    mock_whisper.transcribe.return_value = ([mock_seg], mock_info)

    class DummyRecognizer:
        def __init__(self, m):
            self._model = m
            self.PROMPT_BIAS = ENGLISH_OBJECT_PROMPT_BIAS

    finder = VoiceGuidedObjectFinder(
        memory_manager=manager,
        voice_recognizer=DummyRecognizer(mock_whisper),
    )
    assert finder.prompt_bias == ENGLISH_OBJECT_PROMPT_BIAS

    audio_data = np.zeros(16000, dtype=np.float32)
    _, transcript, _ = finder._recognize_with_fallback(audio_data, sample_rate=16000)

    assert transcript == "Where is my phone?"
    # Verify transcribe was called EXACTLY ONCE with language="en" (no second decode pass)
    assert mock_whisper.transcribe.call_count == 1
    call_args, call_kwargs = mock_whisper.transcribe.call_args
    assert call_kwargs.get("language") == "en"
    assert call_kwargs.get("initial_prompt") == ENGLISH_OBJECT_PROMPT_BIAS
    assert call_kwargs.get("beam_size") == 1

