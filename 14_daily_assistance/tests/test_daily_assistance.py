"""Unit tests for Module 14: Daily Assistance & Reminders."""

from __future__ import annotations

from pathlib import Path
import time
from unittest.mock import MagicMock
import pytest

from shared.schemas import IntentRequest, SpeechPriority, UserIntent
from daily_assistance import (
    DailyService,
    ReminderDB,
    ReminderRecord,
    ReminderScheduler,
)


def test_reminder_db_crud(tmp_path: Path):
    db_file = tmp_path / "test_reminders.db"
    db = ReminderDB(db_path=db_file)

    now = time.time()
    # Add a reminder due immediately
    rem_id = db.add_reminder("Take blood pressure medicine", due_timestamp=now - 1.0, category="medication")
    assert rem_id > 0

    # Retrieve due
    due = db.get_due_reminders(now)
    assert len(due) == 1
    assert due[0].title == "Take blood pressure medicine"

    # Mark completed
    assert db.mark_completed(rem_id) is True
    assert len(db.get_due_reminders(now)) == 0


def test_reminder_scheduler_firing(tmp_path: Path):
    db_file = tmp_path / "test_reminders.db"
    db = ReminderDB(db_path=db_file)
    voice_mock = MagicMock()
    fired_callbacks = []

    def on_rem(rec: ReminderRecord):
        fired_callbacks.append(rec)

    scheduler = ReminderScheduler(
        db=db,
        voice_output=voice_mock,
        check_interval_sec=0.1,
        reminder_callback=on_rem,
    )
    scheduler.start()

    try:
        now = time.time()
        db.add_reminder("Drink water", due_timestamp=now - 0.5, category="hydration")
        time.sleep(0.3)

        assert len(fired_callbacks) == 1
        assert fired_callbacks[0].title == "Drink water"
        voice_mock.speak.assert_called()
    finally:
        scheduler.stop()


def test_daily_service_intent_flow(tmp_path: Path):
    db_file = tmp_path / "test_reminders.db"
    db = ReminderDB(db_path=db_file)
    voice_mock = MagicMock()

    service = DailyService(voice_output=voice_mock, db=db)
    service.initialize()

    try:
        # Set reminder
        req_set = IntentRequest(
            intent=UserIntent.SET_REMINDER,
            raw_query="Mujhe 10 minute baad dawai lene ki yaad dilana",
            target_entity="dawai",
            detected_language="hi",
        )
        resp_set = service.process_query(req_set)
        assert resp_set.success is True
        assert "reminder set" in resp_set.spoken_response_hi

        # List reminders
        req_list = IntentRequest(
            intent=UserIntent.LIST_REMINDERS,
            raw_query="Mere kya reminders hain?",
            detected_language="hi",
        )
        resp_list = service.process_query(req_list)
        assert resp_list.success is True
        assert "1 reminder" in resp_list.spoken_response_hi or "reminder" in resp_list.spoken_response_hi
    finally:
        service.shutdown()
