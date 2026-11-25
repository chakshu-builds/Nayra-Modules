"""Phase 5 Product Integrity, Safety, and Real Workflow Regression Tests.

Validates:
1. Absence of fabricated reassurance, unverified names, and false safety claims.
2. Real module execution vs intent routing across Modules 10-15.
3. Obstacle awareness handling for stale frames and unavailable perception.
4. Object memory visibility state distinctions.
5. Reminder database isolation and deterministic behavior.
6. Speech preemption and voice interaction safety.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from unittest.mock import MagicMock
import pytest

from shared.schemas import IntentRequest, ObstacleHazard, SpeechPriority, UserIntent
from nayra_app import AppConfig, NayraApplication
from obstacle_awareness import ObstacleDetector, ObstacleService
from scene_understanding import SceneService
from alzheimer_assistance import AlzheimerService, OrientationGuide, PatientProfileManager
from daily_assistance import DailyService, ReminderDB
from safety_sos import SafetyService, SOSManager, SOSState
from object_memory import ObjectMemoryManager


def test_no_fabricated_reassurance_on_capability_query():
    """Verify that asking what NAYRA can do returns truthful capabilities and never fabricates names or safety."""
    app = NayraApplication(config=AppConfig(enable_camera=False, default_language="hi"))
    app.initialize()

    resp = app.process_command("Aap kya madad kar sakti ho?")
    assert resp.success is True
    assert resp.intent == UserIntent.HELP
    spoken_hi = resp.spoken_response_hi.lower()
    spoken_en = resp.spoken_response_en.lower()

    # MUST NOT contain unverified claims
    assert "rahul" not in spoken_hi
    assert "rahul" not in spoken_en
    assert "sab theek hai" not in spoken_hi
    assert "everything is fine" not in spoken_en
    assert "jald hi aayenge" not in spoken_hi
    assert "will be here soon" not in spoken_en

    # MUST describe actual capabilities
    assert "cheezein dhundhne" in spoken_hi or "dawai" in spoken_hi or "madad" in spoken_hi
    app.shutdown()


def test_sos_cancellation_honesty(tmp_path: Path):
    """Verify SOS cancellation never asserts the user is safe without evidence."""
    mock_voice = MagicMock()
    sos = SOSManager(
        alerts_log_file=tmp_path / "alerts.jsonl",
        contacts_file=tmp_path / "contacts.json",
        voice_output=mock_voice,
    )
    sos.request_sos(auto_confirm_on_timeout=False)
    spoken = sos.cancel_sos()

    assert ("cancel kar diya gaya" in spoken) or ("cancelled" in spoken.lower())
    # MUST NOT claim "Aap bilkul surakshit hain"
    assert "surakshit" not in spoken.lower()
    assert "safe" not in spoken.lower()


def test_orientation_guide_honesty():
    """Verify OrientationGuide reassurance does not default to fabricated names or unfounded safety claims."""
    guide = OrientationGuide()
    hi, en = guide.get_reassurance()

    assert "rahul" not in hi.lower()
    assert "rahul" not in en.lower()
    assert "sab theek hai" not in hi.lower()
    assert "everything is fine" not in en.lower()
    assert "jald hi aayenge" not in hi.lower()
    assert "will be here soon" not in en.lower()


def test_obstacle_service_stale_and_unavailable_handling():
    """Verify ObstacleService distinguishes active clear path from stale/unavailable perception."""
    service = ObstacleService()
    service.initialize()

    # Case 1: Initial state before any camera frames evaluated
    req = IntentRequest(raw_query="Kya rasta saaf hai?", intent=UserIntent.OBSTACLE_CHECK)
    resp_init = service.process_query(req)
    assert resp_init.success is True
    assert "Camera band" in resp_init.spoken_response_hi or "inactive" in resp_init.spoken_response_en
    # MUST NOT claim unqualified clear path
    assert "bilkul saaf hai" not in resp_init.spoken_response_hi

    # Case 2: Fresh frame evaluated with no obstacles
    service.evaluate_frame(detections=[], frame_width=640, frame_height=480, auto_speak=False)
    resp_fresh = service.process_query(req)
    assert resp_fresh.success is True
    assert resp_fresh.data["perception_available"] is True
    assert "saaf" in resp_fresh.spoken_response_hi

    # Case 3: Stale frame (> 5s ago)
    service._last_evaluated_time = time.time() - 10.0
    resp_stale = service.process_query(req)
    assert resp_stale.success is True
    assert resp_stale.data["stale"] is True
    assert "stale" in resp_stale.spoken_response_hi.lower() or "stale" in resp_stale.spoken_response_en.lower()


def test_obstacle_service_update_obstacles_alias():
    """Verify that update_obstacles alias functions correctly for camera loop integration."""
    service = ObstacleService()
    service.initialize()

    class MockDet:
        class_name = "chair"
        box_xyxy = (200, 250, 400, 470)
        confidence = 0.90

    hazards = service.update_obstacles([MockDet()], frame_width=600, frame_height=480, auto_speak=False)
    assert len(hazards) == 1
    assert hazards[0].class_name == "chair"


def test_reminder_db_isolation(tmp_path: Path):
    """Verify that using an isolated database does not touch data/reminders.db."""
    isolated_db_path = tmp_path / "isolated_reminders.db"
    prod_db_path = Path("data/reminders.db")

    prod_count_before = 0
    if prod_db_path.exists():
        prod_db = ReminderDB(db_path=prod_db_path)
        prod_count_before = len(prod_db.list_active())

    # Create isolated DB and add reminders
    iso_db = ReminderDB(db_path=isolated_db_path)
    r1 = iso_db.add_reminder(title="Test Isolated 1", due_timestamp=time.time() + 3600.0)
    r2 = iso_db.add_reminder(title="Test Isolated 2", due_timestamp=time.time() + 7200.0)
    assert r1 > 0
    assert r2 > 0
    assert len(iso_db.list_active()) == 2

    # Verify prod DB was not mutated
    if prod_db_path.exists():
        prod_db_after = ReminderDB(db_path=prod_db_path)
        assert len(prod_db_after.list_active()) == prod_count_before


def test_real_feature_execution_matrix(tmp_path: Path):
    """Verify actual feature execution across Modules 10, 11, 12, 13, 14, 15 without mocks."""
    test_db = tmp_path / "app_test_reminders.db"
    test_om_db = tmp_path / "app_test_object_memory.db"
    config = AppConfig(
        enable_camera=False,
        default_language="hi",
        reminders_db_path=test_db,
        object_memory_db_path=test_om_db,
    )
    app = NayraApplication(config=config)
    app.initialize()

    # 1. Module 10: Object Memory Execution
    resp_obj = app.process_command("Mera laptop kahan hai?")
    assert resp_obj.success is True
    assert resp_obj.intent == UserIntent.FIND_OBJECT
    assert "never_detected" in resp_obj.data.get("status", "") or "Camera band" in resp_obj.spoken_response_hi

    # 2. Module 11: Obstacle Awareness Execution
    resp_obs = app.process_command("Kya saamne rasta saaf hai?")
    assert resp_obs.success is True
    assert resp_obs.intent == UserIntent.OBSTACLE_CHECK

    # 3. Module 12: Scene Understanding Execution
    resp_scene = app.process_command("Mere aas paas kya hai?")
    assert resp_scene.success is True
    assert resp_scene.intent == UserIntent.DESCRIBE_SCENE

    # 4. Module 13: Alzheimer Assistance Execution
    resp_orient = app.process_command("Main kahan hoon?")
    assert resp_orient.success is True
    assert resp_orient.intent == UserIntent.HELP
    assert "ghar par" in resp_orient.spoken_response_hi

    # 5. Module 14: Daily Assistance Execution
    resp_set_rem = app.process_command("Mujhe 10 minute baad paani peene ka reminder do")
    assert resp_set_rem.success is True
    assert resp_set_rem.intent == UserIntent.SET_REMINDER
    assert "Paani peene" in resp_set_rem.spoken_response_hi

    resp_list_rem = app.process_command("Mere kya reminder hain?")
    assert resp_list_rem.success is True
    assert resp_list_rem.intent == UserIntent.LIST_REMINDERS
    assert resp_list_rem.data.get("count", 0) >= 1

    # 6. Module 15: Safety SOS Execution
    resp_sos = app.process_command("Help me emergency hai")
    assert resp_sos.success is True
    assert resp_sos.intent == UserIntent.TRIGGER_SOS
    assert "sach mein" in resp_sos.spoken_response_hi

    resp_cancel = app.process_command("Nahi galti se bol diya")
    assert resp_cancel.success is True
    assert resp_cancel.intent == UserIntent.CANCEL_SOS
    assert "cancel kar diya" in resp_cancel.spoken_response_hi

    app.shutdown()
