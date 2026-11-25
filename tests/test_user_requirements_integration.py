"""End-to-End Integration Tests for 5 Core User Requirements:
1. Obstacle Navigation for the Blind (proactive collision warnings).
2. Live Object Memory & Wake Word Gating ("Hey NAYRA" gating, secondary landmarks).
3. General Conversational Assistant (normal companion conversation).
4. SOS Voice Trigger & Dispatch to +918607553356.
5. Scheduled Timed Voice Reminders ("Aapka time ho gaya hai yeh cheez lene ka").
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Tuple
from unittest.mock import MagicMock
import pytest

from shared.schemas import IntentRequest, SpeechPriority, UserIntent, ObstacleHazard
from obstacle_awareness import ObstacleDetector, ObstacleService
from object_memory.db.models import ObjectObservation
from object_memory.memory_manager import ObjectMemoryManager
from assistant_orchestrator.intent_router import IntentClassifier
from assistant_orchestrator.orchestrator import AssistantOrchestrator
from safety_sos import SOSManager, SafetyService, EmergencyContact, SOSState
from daily_assistance import DailyService, ReminderDB, ReminderScheduler, ReminderRecord


# =========================================================================
# 1. Obstacle Navigation for the Blind
# =========================================================================

def test_obstacle_navigation_front_collision_warning():
    """When an obstacle like a chair is detected in front and near, proactively warns the user."""
    detector = ObstacleDetector()
    mock_voice = MagicMock()
    service = ObstacleService(voice_output=mock_voice, detector=detector, preferred_language="en")
    service.initialize()

    class MockTrack:
        track_id = 1
        class_name = "chair"
        box_xyxy = (200, 250, 400, 470)  # Center, bottom of frame (NEAR)
        confidence = 0.90

    hazards = service.evaluate_frame([MockTrack()], frame_width=640, frame_height=480, auto_speak=True)

    assert len(hazards) == 1
    assert hazards[0].class_name == "chair"
    assert hazards[0].zone == "CENTER"
    assert hazards[0].is_urgent is True
    assert "Move carefully, you have a chair in your front." in hazards[0].warning_en
    assert "dhyan se chaliye" in hazards[0].warning_hi.lower()
    assert "chair hai" in hazards[0].warning_hi.lower()

    # Voice output must have been called with the English warning
    assert mock_voice.speak.called
    args, kwargs = mock_voice.speak.call_args
    assert "Move carefully, you have a chair in your front." in args[0]


# =========================================================================
# 2. Live Memory & Wake Word Gating
# =========================================================================

def test_wake_word_gating_silent_when_not_spoken():
    """Queries without 'Hey Nayra' remain completely silent when wake word is required."""
    classifier = IntentClassifier(require_wake_word=True)
    voice_mock = MagicMock()
    orchestrator = AssistantOrchestrator(voice_output=voice_mock, classifier=classifier)

    # Ambient room chatter without wake word
    resp = orchestrator.process_voice_query("Where is my book?")
    assert resp.intent == UserIntent.SILENT_IGNORE
    # Voice output must NOT speak anything
    voice_mock.speak.assert_not_called()


def test_wake_word_gating_active_when_spoken():
    """Query with 'Hey Nayra' is stripped and processed cleanly."""
    classifier = IntentClassifier(require_wake_word=True)
    voice_mock = MagicMock()
    orchestrator = AssistantOrchestrator(voice_output=voice_mock, classifier=classifier)

    # User says "Hey Nayra" only
    resp_greet = orchestrator.process_voice_query("Hey Nayra")
    assert resp_greet.intent == UserIntent.WAKE_GREETING
    assert "main sun rahi hoon" in resp_greet.spoken_response_hi.lower() or "listening" in resp_greet.spoken_response_en.lower()

    # User says "Hey Nayra, where is my phone?"
    req_query = classifier.classify("Hey Nayra, where is my phone?")
    assert req_query.intent == UserIntent.FIND_OBJECT
    assert req_query.target_entity == "phone"


def test_historical_memory_with_secondary_landmark(tmp_path: Path):
    """Historical object kept 'on the table near the couch' is accurately spoken."""
    db_file = tmp_path / "test_landmark_memory.db"
    manager = ObjectMemoryManager(db_path=db_file)

    t_seen = time.time() - 40.0
    obs = ObjectObservation(
        id=None,
        object_label="book",
        canonical_name="book",
        hindi_alias="किताब",
        timestamp=t_seen,
        confidence=0.88,
        bbox=(200, 200, 300, 300),
        spatial_relation="on",
        anchor_object="table near the couch",
        horizontal_zone=None,
    )
    manager.record_manual_observation(obs)

    res = manager.query("Where were my books kept?", preferred_language="en")
    assert res.found is True
    assert res.is_currently_visible is False
    assert "on the table near the couch" in res.spoken_response_en.lower()


# =========================================================================
# 3. General Conversational Assistant
# =========================================================================

def test_conversational_assistant_interactions():
    """NAYRA behaves as a normal, warm companion for everyday conversation."""
    classifier = IntentClassifier(require_wake_word=False)
    voice_mock = MagicMock()
    orchestrator = AssistantOrchestrator(voice_output=voice_mock, classifier=classifier)

    # 1. Greeting
    resp_hi = orchestrator.process_voice_query("Namaste")
    assert resp_hi.intent == UserIntent.GENERAL_CHAT
    assert "namaste" in resp_hi.spoken_response_hi.lower() or "hello" in resp_hi.spoken_response_en.lower()

    # 2. How are you
    resp_well = orchestrator.process_voice_query("How are you?")
    assert resp_well.intent == UserIntent.GENERAL_CHAT
    assert "theek hoon" in resp_well.spoken_response_hi.lower() or "doing very well" in resp_well.spoken_response_en.lower()

    # 3. Who are you
    resp_id = orchestrator.process_voice_query("Who are you?")
    assert resp_id.intent == UserIntent.GENERAL_CHAT
    assert "assistive companion" in resp_id.spoken_response_en.lower()

    # 4. Companionship / boredom
    resp_comp = orchestrator.process_voice_query("Mujhe akele lag raha hai, baat karo")
    assert resp_comp.intent == UserIntent.GENERAL_CHAT
    assert "saath hoon" in resp_comp.spoken_response_hi.lower()


# =========================================================================
# 4. SOS Feature to +919215793330
# =========================================================================

def test_sos_direct_voice_dispatch_to_user_number(tmp_path: Path):
    """User command 'Nayra send SOS' dispatches alert to +919215793330."""
    contacts_file = tmp_path / "contacts.json"
    alerts_file = tmp_path / "alerts.jsonl"
    voice_mock = MagicMock()

    sos_mgr = SOSManager(
        voice_output=voice_mock,
        contacts_file=contacts_file,
        alerts_log_file=alerts_file,
        countdown_sec=5.0,
    )
    service = SafetyService(voice_output=voice_mock, sos_manager=sos_mgr)
    service.initialize()

    req = IntentRequest(
        intent=UserIntent.TRIGGER_SOS,
        raw_query="Nayra send SOS",
        parameters={"direct_dispatch": True},
        detected_language="en",
    )

    resp = service.process_query(req)
    assert resp.success is True
    assert resp.intent == UserIntent.CONFIRM_SOS
    assert "+919215793330" in resp.spoken_response_en
    assert sos_mgr.state == SOSState.TRIGGERED

    # Verify log entry in JSONL
    assert alerts_file.exists()
    with open(alerts_file, "r", encoding="utf-8") as f:
        log_entry = json.loads(f.readline())
        phones = [c["phone"] for c in log_entry["contacts_notified"]]
        assert "+919215793330" in phones


# =========================================================================
# 5. Scheduled Timed Reminders
# =========================================================================

def test_scheduled_voice_reminder_firing(tmp_path: Path):
    """When a reminder's time arrives, NAYRA proactively speaks 'Aapka time ho gaya hai yeh cheez lene ka'."""
    db_file = tmp_path / "test_reminders.db"
    db = ReminderDB(db_path=db_file)
    voice_mock = MagicMock()
    fired_records = []

    scheduler = ReminderScheduler(
        db=db,
        voice_output=voice_mock,
        check_interval_sec=0.1,
        reminder_callback=lambda rec: fired_records.append(rec),
    )
    scheduler.start()

    try:
        now = time.time()
        # Add reminder due 0.3s ago
        db.add_reminder("Dawai", due_timestamp=now - 0.3, category="medication")
        time.sleep(0.35)

        assert len(fired_records) == 1
        assert fired_records[0].title == "Dawai"

        # Verify exact spoken phrase
        assert voice_mock.speak.called
        spoken_text = voice_mock.speak.call_args[0][0]
        assert ("time to take your" in spoken_text) or ("Aapka time ho gaya hai" in spoken_text)
    finally:
        scheduler.stop()


def test_clock_time_parsing():
    """Verify DailyService parses 12-hour AM/PM and Hindi clock times."""
    service = DailyService()

    # 12-hour AM/PM
    ts_5pm = service.parse_time_offset("Remind me at 5:00 PM to take insulin")
    assert ts_5pm > time.time()

    # Hindi clock time
    ts_6baje = service.parse_time_offset("Shaam 6 baje dawai yaad dilana")
    assert ts_6baje > time.time()
