"""Unit tests for Module 13: Alzheimer's & Cognitive Assistance."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from shared.schemas import IntentRequest, SpeechPriority, UserIntent
from alzheimer_assistance import (
    AlzheimerService,
    FamiliarPerson,
    OrientationGuide,
    PatientProfileManager,
)


def test_orientation_guide():
    guide = OrientationGuide(home_location="apne ghar par")
    fixed_time = datetime.datetime(2026, 11, 23, 10, 30)
    hi, en = guide.get_orientation(fixed_time)

    assert "apne ghar par" in hi
    assert "Somwaar" in hi
    assert "10:30 AM" in hi
    assert "Monday" in en
    assert "November 23" in en


def test_patient_profile_manager(tmp_path: Path):
    db_file = tmp_path / "test_profile.json"
    mgr = PatientProfileManager(data_file=db_file)
    assert mgr.patient_name == "User"

    # Add a new person
    p = FamiliarPerson(person_id="p99", name="Aman", relationship="bhai", phone="12345")
    mgr.add_or_update_person(p)

    reloaded = PatientProfileManager(data_file=db_file)
    found = reloaded.get_person_by_name("Aman")
    assert found is not None
    assert found.relationship == "bhai"


def test_alzheimer_service_intent_flow(tmp_path: Path):
    voice_mock = MagicMock()
    db_file = tmp_path / "test_profile.json"
    mgr = PatientProfileManager(data_file=db_file)
    service = AlzheimerService(voice_output=voice_mock, profile_manager=mgr)
    service.initialize()

    # Orientation query
    req1 = IntentRequest(
        intent=UserIntent.HELP,
        raw_query="Main kahan hoon?",
        detected_language="hi",
    )
    resp1 = service.process_query(req1)
    assert resp1.success is True
    assert "ghar par" in resp1.spoken_response_hi.lower()
    voice_mock.speak.assert_called()

    # Person identification query
    req2 = IntentRequest(
        intent=UserIntent.IDENTIFY_PERSON,
        raw_query="Yeh kaun hai?",
        target_entity="Rahul",
        detected_language="hi",
    )
    resp2 = service.process_query(req2)
    assert resp2.success is True
    assert "Rahul" in resp2.spoken_response_hi
