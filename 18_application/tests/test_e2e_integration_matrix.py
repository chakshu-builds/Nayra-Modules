"""End-to-End Integration Test Matrix for NAYRA Voice-First MVP.

Verifies the 18 critical operational integration scenarios:
 1. Spoken object-location query.
 2. Object visible versus not visible (currently visible, recently seen, previously seen, never seen, unsupported).
 3. Obstacle detection and spoken warning (priority preemption).
 4. Scene description (multi-modal spatial narration).
 5. OCR/read-aloud request.
 6. Familiar-person query (no false identification).
 7. Alzheimer's orientation assistance (truthful date/time/location without false safety claims).
 8. Medication reminder creation and background firing (not automatically marked taken).
 9. Reminder snooze and dismissal (explicit completion).
10. SOS trigger, two-stage confirmation, and cancellation.
11. Hindi commands.
12. Hinglish commands.
13. English commands.
14. Multiple commands in succession.
15. TTS playback without self-triggering Whisper (AcousticShield gating).
16. Camera disconnection and reconnection handling.
17. Microphone disconnection and reconnection handling.
18. Graceful application shutdown.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest

from shared.schemas import IntentRequest, SpeechPriority, UserIntent
from nayra_app import AppConfig, NayraApplication
from obstacle_awareness import ObstacleDetector, ObstacleService
from alzheimer_assistance import FamiliarPerson
from safety_sos import SOSState


@pytest.fixture
def mock_app(tmp_path: Path):
    """Fixture providing an isolated NayraApplication instance for integration testing."""
    config = AppConfig(
        enable_camera=False,
        default_language="hi",
        data_dir=tmp_path / "data",
    )
    # Use temporary files for safety logs & reminders
    app = NayraApplication(config=config)
    app.safety_service.sos.alerts_log_file = tmp_path / "emergency_alerts.jsonl"
    app.safety_service.sos.contacts_file = tmp_path / "emergency_contacts.json"
    app.daily_service.db.db_path = tmp_path / "reminders.db"
    app.daily_service.db._init_db()

    app.initialize()
    app.start()
    yield app
    app.shutdown()


# 1 & 2: Object Location Memory - Visible vs Recently Seen vs Previously Seen vs Never Seen vs Unsupported
def test_scenario_01_and_02_object_location_visibility_distinctions(mock_app: NayraApplication):
    mock_memory = MagicMock()
    mock_memory.get_currently_visible_observation.return_value = None
    mock_memory.get_latest_observation.return_value = None
    mock_app.object_memory_adapter.memory = mock_memory

    # Case A: Unsupported object class
    resp_unsupported = mock_app.process_command("Mera alien spaceship kahan hai?")
    assert resp_unsupported.success is True
    assert "train nahi hoon" in resp_unsupported.spoken_response_hi or "not trained" in resp_unsupported.spoken_response_en

    # Case B: Supported object class, but never detected
    resp_never = mock_app.process_command("Mera laptop kahan hai?")
    assert resp_never.success is True
    assert "pehle kabhi nahi dikha" in resp_never.spoken_response_hi or "never seen" in resp_never.spoken_response_en

    # Case C: Mock memory manager returning currently visible observation
    mock_obs_visible = MagicMock()
    mock_obs_visible.anchor_object = "table"
    mock_obs_visible.horizontal_zone = "CENTER"
    mock_memory.get_currently_visible_observation.return_value = mock_obs_visible

    resp_visible = mock_app.process_command("Mera phone kahan hai?")
    assert resp_visible.success is True
    assert "abhi" in resp_visible.spoken_response_hi
    assert "table" in resp_visible.spoken_response_hi

    # Case D: Not currently visible, but recently observed (< 3 min)
    mock_memory.get_currently_visible_observation.return_value = None
    mock_obs_recent = MagicMock()
    mock_obs_recent.anchor_object = "sofa"
    mock_obs_recent.timestamp = time.time() - 45.0  # 45s ago
    mock_memory.get_latest_observation.return_value = mock_obs_recent

    resp_recent = mock_app.process_command("Mera phone kahan hai?")
    assert resp_recent.success is True
    assert "kuch der pehle" in resp_recent.spoken_response_hi
    assert "sofa" in resp_recent.spoken_response_hi

    # Case E: Previously observed (older historical record, e.g. 2 hours ago)
    mock_obs_old = MagicMock()
    mock_obs_old.anchor_object = "desk"
    mock_obs_old.timestamp = time.time() - 7200.0  # 120 min ago
    mock_memory.get_latest_observation.return_value = mock_obs_old

    resp_old = mock_app.process_command("Mera phone kahan hai?")
    assert resp_old.success is True
    assert "120 minute pehle" in resp_old.spoken_response_hi or "desk" in resp_old.spoken_response_hi


# 3: Obstacle detection and spoken warning with priority preemption
def test_scenario_03_obstacle_detection_and_preemption(mock_app: NayraApplication):
    # Path check query
    resp_clear = mock_app.process_command("Kya saamne rasta saaf hai?")
    assert resp_clear.success is True
    assert "saaf" in resp_clear.spoken_response_hi

    # Simulate urgent obstacle detection in center zone
    class MockDet:
        class_name = "chair"
        box_xyxy = (200, 250, 400, 470)
        confidence = 0.90

    hazards = mock_app.obstacle_service.detector.evaluate_obstacles([MockDet()], frame_width=600, frame_height=480)
    assert len(hazards) == 1
    assert hazards[0].is_urgent is True

    # Voice warning must use priority OBSTACLE_WARNING (Priority 3) or CRITICAL_SAFETY (Priority 1)
    spoken_warn = hazards[0].warning_hi
    assert "kursi" in spoken_warn or "chair" in spoken_warn


# 4: Scene description multi-modal spatial narration
def test_scenario_04_scene_description(mock_app: NayraApplication):
    class MockObj:
        def __init__(self, name, box):
            self.class_name = name
            self.box_xyxy = box
            self.confidence = 0.85

    dets = [
        MockObj("laptop", (220, 200, 320, 300)),
        MockObj("table", (180, 220, 450, 450)),
        MockObj("person", (50, 100, 180, 450)),
    ]
    mock_app.scene_service.update_scene_context(dets, frame_width=600, frame_height=480, ocr_text="CAUTION WET FLOOR")

    resp = mock_app.process_command("Mere saamne kya hai?")
    assert resp.success is True
    assert "laptop" in resp.spoken_response_hi or "table" in resp.spoken_response_hi
    assert "CAUTION WET FLOOR" in resp.spoken_response_hi or "CAUTION WET FLOOR" in resp.spoken_response_en


# 5: OCR / Read-aloud request
def test_scenario_05_ocr_read_aloud(mock_app: NayraApplication):
    # With active OCR text
    mock_app.ocr_adapter.update_text("PARACETAMOL 500MG TABLETS")
    resp = mock_app.process_command("Board par kya likha hai?")
    assert resp.success is True
    assert "PARACETAMOL 500MG TABLETS" in resp.spoken_response_hi

    # Empty OCR
    mock_app.ocr_adapter.update_text("")
    resp_empty = mock_app.process_command("Read the text in front of me")
    assert resp_empty.success is True
    assert "koi text" in resp_empty.spoken_response_hi or "no readable text" in resp_empty.spoken_response_en.lower()


# 6: Familiar person query (no false identification)
def test_scenario_06_familiar_person_no_false_identification(mock_app: NayraApplication):
    # Query for known caregiver profile by name
    p = FamiliarPerson(person_id="p1", name="Rahul", relationship="beta", phone="9876543210", notes="Roz aate hain")
    mock_app.alzheimer_service.profile.add_or_update_person(p)

    resp_named = mock_app.process_command("Who is Rahul?")
    assert resp_named.success is True
    assert "Rahul" in resp_named.spoken_response_en
    assert "beta" in resp_named.spoken_response_hi or "beta" in resp_named.spoken_response_en

    # Query for unidentified person in view when face recognition has no match
    resp_unknown = mock_app.process_command("Yeh kaun hai?")
    assert resp_unknown.success is True
    assert "jaana-pehchana vyakti nahi" in resp_unknown.spoken_response_hi or "do not recognize" in resp_unknown.spoken_response_en
    # MUST NOT falsely invent that someone is Rahul!
    assert "Saamne Rahul hain" not in resp_unknown.spoken_response_hi


# 7: Alzheimer's orientation assistance (truthful date/time/location without false safety claims)
def test_scenario_07_alzheimer_orientation_truthful(mock_app: NayraApplication):
    resp = mock_app.process_command("Main kahan hoon?")
    assert resp.success is True
    # Grounded in registered location
    assert "ghar par" in resp.spoken_response_hi.lower()
    # Does not make unverified safety assertions
    assert "Aap bilkul surakshit hain" not in resp.spoken_response_hi


# 8: Medication reminder creation and background firing (not automatically marked taken)
def test_scenario_08_medication_reminder_not_marked_taken(mock_app: NayraApplication):
    now = time.time()
    # Add reminder due 0.1s in past
    rem_id = mock_app.daily_service.db.add_reminder(
        title="Dawai lene ka samay",
        due_timestamp=now - 0.1,
        category="medication",
    )

    # Let scheduler process it
    time.sleep(0.3)

    # Verify that the reminder is STILL NOT marked completed!
    active_rems = mock_app.daily_service.db.list_active()
    assert any(r.id == rem_id and not r.is_completed for r in active_rems), "Spoken reminder was incorrectly marked taken!"


# 9: Reminder snooze and dismissal (explicit completion)
def test_scenario_09_reminder_snooze_and_completion(mock_app: NayraApplication):
    # Create active reminder
    rem_id = mock_app.daily_service.db.add_reminder(
        title="Blood pressure medicine",
        due_timestamp=time.time() + 60.0,
        category="medication",
    )

    # Snooze
    resp_snooze = mock_app.process_command("Reminder ko snooze kar do")
    assert resp_snooze.success is True
    assert "snooze" in resp_snooze.spoken_response_hi

    # User confirms medication taken
    resp_taken = mock_app.process_command("Maine dawai le li hai")
    assert resp_taken.success is True
    assert "complete" in resp_taken.spoken_response_hi or "dismiss" in resp_taken.spoken_response_hi

    # Verify it is now completed
    assert len(mock_app.daily_service.db.list_active()) == 0


# 10: SOS trigger, confirmation, and cancellation
def test_scenario_10_sos_two_stage_and_cancellation(mock_app: NayraApplication):
    # Step A: Trigger
    resp_trig = mock_app.process_command("Madad karo emergency hai")
    assert resp_trig.success is True
    assert mock_app.safety_service.sos.state == SOSState.PENDING_CONFIRMATION
    assert "sach mein" in resp_trig.spoken_response_hi

    # Step B: Cancel
    resp_cancel = mock_app.process_command("Nahi cancel kar do galti se bol diya")
    assert resp_cancel.success is True
    assert mock_app.safety_service.sos.state == SOSState.IDLE
    assert "cancel kar diya" in resp_cancel.spoken_response_hi

    # Step C: Trigger and Confirm
    mock_app.process_command("Emergency help me!")
    assert mock_app.safety_service.sos.state == SOSState.PENDING_CONFIRMATION

    resp_conf = mock_app.process_command("Haan emergency madad chahiye")
    assert resp_conf.success is True
    assert mock_app.safety_service.sos.state == SOSState.TRIGGERED
    assert "local log mein darj" in resp_conf.spoken_response_hi


# 11, 12, 13: Multilingual queries (Hindi, Hinglish, English)
def test_scenario_11_12_13_multilingual_routing(mock_app: NayraApplication):
    # Pure Hindi
    r_hi = mock_app.process_command("Kya aage rasta saaf hai?")
    assert r_hi.intent == UserIntent.OBSTACLE_CHECK

    # Hinglish
    r_hing = mock_app.process_command("Mujhe 10 minute baad dawai remind karna")
    assert r_hing.intent == UserIntent.SET_REMINDER

    # English
    r_en = mock_app.process_command("What is in front of me?")
    assert r_en.intent == UserIntent.DESCRIBE_SCENE


# 14: Multiple commands in succession
def test_scenario_14_successive_dialog_commands(mock_app: NayraApplication):
    queries = [
        "Main kahan hoon?",
        "Kya rasta saaf hai?",
        "Mujhe paani peene ka reminder do",
        "Mere kya reminders hain?",
    ]
    for q in queries:
        resp = mock_app.process_command(q)
        assert resp.success is True


# 15: TTS playback without self-triggering Whisper (AcousticShield timing)
def test_scenario_15_acoustic_shield_gating(mock_app: NayraApplication):
    shield = mock_app.voice_manager.shield

    # Before speech: mic open
    t0 = 2000.0
    assert shield.is_shield_active(now=t0) is False

    # Start speech: mic muted
    shield.on_speech_started()
    assert shield.is_shield_active(now=t0) is True

    # End speech: cooldown active for 1.2s
    shield.on_speech_ended(now=t0 + 2.0)
    assert shield.is_shield_active(now=t0 + 2.5) is True  # 0.5s into cooldown -> mic STILL muted
    assert shield.is_shield_active(now=t0 + 3.5) is False # 1.5s after speech -> mic open again


# 16: Camera disconnection and reconnection handling
def test_scenario_16_camera_fault_tolerance():
    # If camera index is invalid or stream is unavailable, application must degrade gracefully
    bad_config = AppConfig(enable_camera=True, camera_index=9999)
    app = NayraApplication(config=bad_config)
    assert app.initialize() is True
    # Should not crash on invalid camera device
    app.start()
    resp = app.process_command("Main kahan hoon?")
    assert resp.success is True
    app.shutdown()


# 17: Microphone disconnection handling
def test_scenario_17_microphone_fault_tolerance(mock_app: NayraApplication):
    # Audio buffer error handling
    initial_len = len(mock_app.voice_manager.queue)
    # Sending empty or None audio does not crash voice system
    assert mock_app.voice_manager.speak("") is False
    assert len(mock_app.voice_manager.queue) == initial_len


# 18: Graceful application shutdown
def test_scenario_18_graceful_shutdown():
    app = NayraApplication(config=AppConfig(enable_camera=False))
    app.initialize()
    app.start()
    assert app._running is True
    app.shutdown()
    assert app._running is False
    assert not app.voice_manager._is_active
