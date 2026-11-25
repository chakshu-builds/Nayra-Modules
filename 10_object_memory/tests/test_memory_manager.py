"""NAYRA Module 10 — Unit tests for Central Object Location Memory Manager."""

from __future__ import annotations

from pathlib import Path
import time
import pytest

from object_memory.db.models import ObjectObservation
from object_memory.memory_manager import ObjectMemoryManager


@pytest.fixture
def temp_manager(tmp_path: Path) -> ObjectMemoryManager:
    db_file = tmp_path / "test_memory_mgr.db"
    return ObjectMemoryManager(db_path=db_file)


def test_manager_manual_record_and_query_flow(temp_manager: ObjectMemoryManager):
    t_obs = time.time() - 120  # 2 minutes ago
    obs = ObjectObservation(
        id=None,
        object_label="cell phone",
        canonical_name="phone",
        hindi_alias="फ़ोन",
        timestamp=t_obs,
        confidence=0.92,
        bbox=(200, 200, 260, 260),
        spatial_relation="on",
        anchor_object="dining table",
        horizontal_zone="CENTER",
    )
    temp_manager.record_manual_observation(obs)

    # Check known objects
    known = temp_manager.list_known_objects()
    assert len(known) == 1
    assert known[0]["canonical_name"] == "phone"

    # Query in Hindi
    res_hi = temp_manager.query("Mera phone kahan hai?")
    assert res_hi.found is True
    assert res_hi.canonical_name == "phone"
    assert "dining table par" in res_hi.spoken_response_hi
    assert "wahin hona chahiye" in res_hi.spoken_response_hi

    # Query in English
    res_en = temp_manager.query("Where is my phone?", preferred_language="en")
    assert res_en.found is True
    assert "on the dining table" in res_en.spoken_response_en
    assert "should still be there" in res_en.spoken_response_en


def test_manager_query_unsupported_object(temp_manager: ObjectMemoryManager):
    res = temp_manager.query("Mera chashma kahan hai?")
    assert res.found is False
    assert res.canonical_name == "glasses"
    assert "saksham nahi" in res.spoken_response_hi


def test_manager_query_not_found_object(temp_manager: ObjectMemoryManager):
    res = temp_manager.query("Where is the TV remote?")
    assert res.found is False
    assert res.canonical_name == "remote"
    assert "haven't seen your remote yet" in res.spoken_response_en


def test_manager_clear_and_prune(temp_manager: ObjectMemoryManager):
    obs = ObjectObservation(
        id=None,
        object_label="bottle",
        canonical_name="bottle",
        timestamp=time.time(),
        confidence=0.85,
        bbox=(50, 50, 80, 120),
    )
    temp_manager.record_manual_observation(obs)
    assert len(temp_manager.list_known_objects()) == 1

    temp_manager.clear()
    assert len(temp_manager.list_known_objects()) == 0
