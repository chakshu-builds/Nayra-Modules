"""NAYRA Module 10 — Unit tests for SQLite Object Memory Database."""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from object_memory.db.models import ObjectObservation, MemoryQueryResult, RoomAnchor
from object_memory.db.database import ObjectMemoryDB


@pytest.fixture
def temp_db(tmp_path: Path) -> ObjectMemoryDB:
    db_file = tmp_path / "test_memory.db"
    return ObjectMemoryDB(db_path=db_file)


def test_db_initialization(temp_db: ObjectMemoryDB):
    assert temp_db.db_path.exists()
    all_objs = temp_db.list_all_tracked_objects()
    assert len(all_objs) == 0


def test_record_and_get_latest_observation(temp_db: ObjectMemoryDB):
    t0 = time.time() - 100
    obs1 = ObjectObservation(
        id=None,
        object_label="cell phone",
        canonical_name="phone",
        hindi_alias="फ़ोन",
        timestamp=t0,
        confidence=0.88,
        bbox=(100, 100, 200, 200),
        spatial_relation="on",
        anchor_object="dining table",
        horizontal_zone="CENTER",
    )
    id1 = temp_db.record_observation(obs1)
    assert id1 is not None

    latest = temp_db.get_latest_observation("phone")
    assert latest is not None
    assert latest.id == id1
    assert latest.canonical_name == "phone"
    assert latest.anchor_object == "dining table"
    assert latest.spatial_relation == "on"
    assert latest.is_latest is True

    # Record newer observation
    t1 = time.time()
    obs2 = ObjectObservation(
        id=None,
        object_label="cell phone",
        canonical_name="phone",
        hindi_alias="फ़ोन",
        timestamp=t1,
        confidence=0.92,
        bbox=(150, 150, 250, 250),
        spatial_relation="near",
        anchor_object="couch",
        horizontal_zone="RIGHT",
    )
    id2 = temp_db.record_observation(obs2)
    assert id2 > id1

    latest_updated = temp_db.get_latest_observation("phone")
    assert latest_updated is not None
    assert latest_updated.id == id2
    assert latest_updated.anchor_object == "couch"
    assert latest_updated.is_latest is True

    # Check history
    history = temp_db.get_history("phone", limit=5)
    assert len(history) == 2
    assert history[0].id == id2
    assert history[1].id == id1
    assert history[1].is_latest is False


def test_list_all_tracked_objects(temp_db: ObjectMemoryDB):
    obs_phone = ObjectObservation(
        id=None,
        object_label="cell phone",
        canonical_name="phone",
        timestamp=time.time(),
        confidence=0.9,
        bbox=(50, 50, 100, 100),
        spatial_relation="on",
        anchor_object="dining table",
    )
    obs_remote = ObjectObservation(
        id=None,
        object_label="remote",
        canonical_name="remote",
        timestamp=time.time() - 50,
        confidence=0.85,
        bbox=(200, 200, 250, 230),
        spatial_relation="on",
        anchor_object="couch",
    )
    temp_db.record_observation(obs_phone)
    temp_db.record_observation(obs_remote)

    tracked = temp_db.list_all_tracked_objects()
    assert len(tracked) == 2
    canonical_names = [o["canonical_name"] for o in tracked]
    assert "phone" in canonical_names
    assert "remote" in canonical_names


def test_query_logging(temp_db: ObjectMemoryDB):
    temp_db.log_query(
        user_query="Mera phone kahan hai?",
        matched_object="phone",
        obs_id=1,
        response_text="Aapka phone dining table par hai.",
    )
    with temp_db._get_connection() as conn:
        row = conn.execute("SELECT * FROM memory_queries").fetchone()
        assert row is not None
        assert row["matched_object"] == "phone"
        assert "dining table" in row["response_text"]


def test_prune_old_observations(temp_db: ObjectMemoryDB):
    # Old historical record (15 days ago)
    old_time = time.time() - (15 * 86400)
    obs_old = ObjectObservation(
        id=None,
        object_label="bottle",
        canonical_name="bottle",
        timestamp=old_time,
        confidence=0.8,
        bbox=(10, 10, 40, 80),
    )
    temp_db.record_observation(obs_old)

    # Newer record for bottle (making old one is_latest = 0)
    obs_new = ObjectObservation(
        id=None,
        object_label="bottle",
        canonical_name="bottle",
        timestamp=time.time(),
        confidence=0.9,
        bbox=(20, 20, 50, 90),
    )
    temp_db.record_observation(obs_new)

    history = temp_db.get_history("bottle")
    assert len(history) == 2

    # Prune records older than 14 days
    pruned_count = temp_db.prune_old_observations(max_age_days=14)
    assert pruned_count == 1

    history_after = temp_db.get_history("bottle")
    assert len(history_after) == 1
    assert history_after[0].id == obs_new.id


def test_clear_all(temp_db: ObjectMemoryDB):
    obs = ObjectObservation(
        id=None,
        object_label="book",
        canonical_name="book",
        timestamp=time.time(),
        confidence=0.8,
        bbox=(10, 10, 50, 50),
    )
    temp_db.record_observation(obs)
    assert len(temp_db.list_all_tracked_objects()) == 1

    temp_db.clear_all()
    assert len(temp_db.list_all_tracked_objects()) == 0
