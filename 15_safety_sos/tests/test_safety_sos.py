"""Unit tests for Module 15: Safety & SOS Assistance."""

from __future__ import annotations

from pathlib import Path
import time
from unittest.mock import MagicMock
import pytest

from shared.schemas import IntentRequest, SpeechPriority, UserIntent
from safety_sos import (
    EmergencyContact,
    SafetyService,
    SOSManager,
    SOSState,
)


def test_sos_two_stage_confirmation(tmp_path: Path):
    contacts_file = tmp_path / "contacts.json"
    alerts_file = tmp_path / "alerts.jsonl"
    voice_mock = MagicMock()

    mgr = SOSManager(
        voice_output=voice_mock,
        contacts_file=contacts_file,
        alerts_log_file=alerts_file,
        countdown_sec=0.5,
    )

    # 1. Trigger
    spoken = mgr.request_sos("User fell down")
    assert mgr.state == SOSState.PENDING_CONFIRMATION
    assert "sach mein" in spoken
    voice_mock.speak.assert_called_with(spoken, priority=SpeechPriority.EMERGENCY_INTERACTION)

    # 2. Confirm actively
    spoken_conf = mgr.confirm_sos()
    assert mgr.state == SOSState.TRIGGERED
    assert "activate ho gaya" in spoken_conf
    assert alerts_file.exists()


def test_sos_cancel(tmp_path: Path):
    contacts_file = tmp_path / "contacts.json"
    alerts_file = tmp_path / "alerts.jsonl"
    mgr = SOSManager(
        contacts_file=contacts_file,
        alerts_log_file=alerts_file,
        countdown_sec=2.0,
    )

    mgr.request_sos("Mistaken button press")
    assert mgr.state == SOSState.PENDING_CONFIRMATION

    spoken_cancel = mgr.cancel_sos()
    assert "cancel kar diya" in spoken_cancel
    assert mgr.state == SOSState.IDLE


def test_safety_service_intents(tmp_path: Path):
    contacts_file = tmp_path / "contacts.json"
    alerts_file = tmp_path / "alerts.jsonl"
    mgr = SOSManager(
        contacts_file=contacts_file,
        alerts_log_file=alerts_file,
        countdown_sec=5.0,
    )
    service = SafetyService(sos_manager=mgr)
    service.initialize()

    # Trigger via intent
    req_trigger = IntentRequest(
        intent=UserIntent.TRIGGER_SOS,
        raw_query="Bachao mujhe madad chahiye",
        detected_language="hi",
    )
    resp_trig = service.process_query(req_trigger)
    assert resp_trig.success is True
    assert mgr.state == SOSState.PENDING_CONFIRMATION

    # Confirm via intent
    req_conf = IntentRequest(
        intent=UserIntent.CONFIRM_SOS,
        raw_query="Haan madad chahiye",
        detected_language="hi",
    )
    resp_conf = service.process_query(req_conf)
    assert resp_conf.success is True
    assert mgr.state == SOSState.TRIGGERED
