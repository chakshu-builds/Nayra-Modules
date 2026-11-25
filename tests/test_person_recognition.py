"""Unit and integration tests for Module 06: Consent-Based Familiar Person Recognition."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_person_rec = importlib.import_module("06_person_recognition")
sys.modules["person_recognition"] = _person_rec

FaceAligner = _person_rec.FaceAligner
FaceDatabase = _person_rec.FaceDatabase
FaceDetector = _person_rec.FaceDetector
FaceRecognizer = _person_rec.FaceRecognizer
FaceTracker = _person_rec.FaceTracker
PersonRecognitionPipeline = _person_rec.PersonRecognitionPipeline
PersonContextDatabase = _person_rec.PersonContextDatabase
FaceDetection = _person_rec.FaceDetection
RecognitionStatus = _person_rec.RecognitionStatus
RecognizedPerson = _person_rec.RecognizedPerson
PersonContext = _person_rec.PersonContext
FactItem = _person_rec.FactItem
PreferenceItem = _person_rec.PreferenceItem

from config.settings import PersonRecognitionSettings


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """Create a clean 320x320 synthetic face image with head and eye features."""
    img = np.ones((320, 320, 3), dtype=np.uint8) * 180
    # Head ellipse
    cv2.ellipse(img, (160, 160), (70, 90), 0, 0, 360, (210, 210, 210), -1)
    # Eyes
    cv2.circle(img, (135, 140), 12, (50, 50, 50), -1)
    cv2.circle(img, (185, 140), 12, (50, 50, 50), -1)
    # Nose
    cv2.line(img, (160, 145), (160, 175), (80, 80, 80), 3)
    # Mouth
    cv2.ellipse(img, (160, 200), (25, 10), 0, 0, 180, (60, 60, 60), 3)
    return img


@pytest.fixture
def unit_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic orthogonal/similar 128D unit embeddings."""
    np.random.seed(42)
    # Vector A
    v_a = np.random.randn(128).astype(np.float32)
    v_a /= np.linalg.norm(v_a)

    # Vector A' (similar to A, cosine sim ~ 0.93)
    v_noise = np.random.randn(128).astype(np.float32)
    v_noise -= np.dot(v_noise, v_a) * v_a
    v_noise /= np.linalg.norm(v_noise)
    v_a_sim = 0.93 * v_a + 0.36 * v_noise
    v_a_sim /= np.linalg.norm(v_a_sim)

    # Vector B (orthogonal to A, cosine sim ~ 0.0)
    v_b = np.random.randn(128).astype(np.float32)
    v_b -= np.dot(v_b, v_a) * v_a
    v_b /= np.linalg.norm(v_b)

    return v_a, v_a_sim, v_b


def test_face_detector_initialization():
    settings = PersonRecognitionSettings()
    detector = FaceDetector(settings)
    assert detector.model_path.exists()
    assert detector._detector is not None


def test_face_detector_invalid_and_empty_frames():
    detector = FaceDetector()
    assert detector.detect(None) == []
    assert detector.detect(np.empty((0, 0, 3), dtype=np.uint8)) == []
    assert detector.detect(np.zeros((5, 5, 3), dtype=np.uint8)) == []


def test_face_detector_on_frame(synthetic_face_image):
    detector = FaceDetector()
    detections = detector.detect(synthetic_face_image)
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, FaceDetection)
        assert len(det.landmarks_5) == 5
        assert 0.0 <= det.confidence <= 1.0


def test_face_aligner_geometry():
    aligner = FaceAligner(target_size=(112, 112))
    dummy_det = FaceDetection(
        box_xywh=(50, 50, 100, 100),
        box_xyxy=(50, 50, 150, 150),
        landmarks_5=[(80.0, 80.0), (120.0, 80.0), (100.0, 100.0), (85.0, 125.0), (115.0, 125.0)],
        confidence=0.95,
    )
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
    aligned = aligner.align(frame, dummy_det)
    assert aligned is not None
    assert aligned.shape == (112, 112, 3)


def test_face_aligner_quality_checks():
    aligner = FaceAligner()
    # Test valid crop
    np.random.seed(0)
    valid_crop = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    ok, _ = aligner.check_quality(valid_crop)
    assert ok is True

    # Test black/underexposed crop
    black_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    ok, reason = aligner.check_quality(black_crop)
    assert ok is False
    assert "dark" in reason

    # Test blurry uniform crop
    flat_crop = np.full((112, 112, 3), 120, dtype=np.uint8)
    ok, reason = aligner.check_quality(flat_crop)
    assert ok is False
    assert "blurry" in reason


def test_face_recognizer_embedding_dimensions_and_normalization():
    recognizer = FaceRecognizer()
    crop = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    emb = recognizer.extract_embedding(crop)
    assert emb is not None
    assert emb.shape == (128,)
    # Norm must be 1.0 (unit L2 length)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)


def test_similarity_calculation(unit_embeddings):
    v_a, v_a_sim, v_b = unit_embeddings
    sim_identical = FaceRecognizer.compute_similarity(v_a, v_a)
    assert np.isclose(sim_identical, 1.0, atol=1e-4)

    sim_high = FaceRecognizer.compute_similarity(v_a, v_a_sim)
    assert sim_high > 0.75

    sim_orthogonal = FaceRecognizer.compute_similarity(v_a, v_b)
    assert abs(sim_orthogonal) < 0.25


def test_face_database_encryption_and_persistence(temp_dir, unit_embeddings):
    db_file = temp_dir / "test_face_db.bin"
    key_file = temp_dir / "test_face_db.key"
    settings = PersonRecognitionSettings(
        db_path=str(db_file),
        key_path=str(key_file),
        min_samples_per_enrollment=2,
    )

    db = FaceDatabase(settings)
    assert key_file.exists()
    assert len(db.list_profiles()) == 0

    v_a, v_a_sim, _ = unit_embeddings
    profile = db.enroll_person("Chakshu", [v_a, v_a_sim])
    assert profile.display_name == "Chakshu"
    assert profile.sample_count == 2
    assert db_file.exists()

    # Verify encrypted on disk (not raw JSON plain text)
    raw_disk_bytes = db_file.read_bytes()
    assert b"Chakshu" not in raw_disk_bytes

    # Load in new database instance
    db2 = FaceDatabase(settings)
    assert len(db2.list_profiles()) == 1
    loaded = db2.get_profile_by_name("Chakshu")
    assert loaded is not None
    assert loaded.display_name == "Chakshu"
    assert len(loaded.embeddings) == 2


def test_face_database_duplicate_handling_and_deletion(temp_dir, unit_embeddings):
    db_file = temp_dir / "test_face_db.bin"
    key_file = temp_dir / "test_face_db.key"
    settings = PersonRecognitionSettings(
        db_path=str(db_file),
        key_path=str(key_file),
        min_samples_per_enrollment=2,
    )
    db = FaceDatabase(settings)
    v_a, v_a_sim, v_b = unit_embeddings

    # Enroll
    db.enroll_person("Bhavya", [v_a, v_a_sim])
    assert len(db.list_profiles()) == 1

    # Re-enroll with same name updates existing profile without duplicating ID
    db.enroll_person("bhavya", [v_b, v_a])
    assert len(db.list_profiles()) == 1

    # Delete profile
    deleted = db.delete_profile("Bhavya")
    assert deleted is True
    assert len(db.list_profiles()) == 0

    # Deleting non-existent profile returns False
    assert db.delete_profile("UnknownPerson") is False


def test_face_database_corrupted_file_handling(temp_dir):
    db_file = temp_dir / "corrupted_db.bin"
    key_file = temp_dir / "corrupted_key.key"
    db_file.write_bytes(b"garbage-corrupted-bytes-12345")
    key_file.write_bytes(b"invalid-key-bytes")

    settings = PersonRecognitionSettings(db_path=str(db_file), key_path=str(key_file))
    # Should not raise uncaught crash; safely initializes empty
    db = FaceDatabase(settings)
    assert len(db.list_profiles()) == 0


def test_matching_known_and_unknown_identities(temp_dir, unit_embeddings):
    db_file = temp_dir / "test_match.bin"
    key_file = temp_dir / "test_match.key"
    settings = PersonRecognitionSettings(
        db_path=str(db_file),
        key_path=str(key_file),
        recognition_threshold=0.50,
        min_samples_per_enrollment=2,
    )
    db = FaceDatabase(settings)
    v_a, v_a_sim, v_b = unit_embeddings

    db.enroll_person("Chakshu", [v_a, v_a_sim])

    pipeline = PersonRecognitionPipeline(settings=settings, database=db)

    # Probe with similar embedding -> Recognized
    is_known, pid, name, score, status = pipeline.match_embedding(v_a_sim)
    assert is_known is True
    assert name == "Chakshu"
    assert status == RecognitionStatus.RECOGNIZED
    assert score >= 0.50

    # Probe with orthogonal embedding -> Unknown
    is_known_b, pid_b, name_b, score_b, status_b = pipeline.match_embedding(v_b)
    assert is_known_b is False
    assert name_b == "Unknown"
    assert status_b == RecognitionStatus.UNKNOWN


def test_ambiguity_margin_rejection(temp_dir, unit_embeddings):
    db_file = temp_dir / "test_ambig.bin"
    key_file = temp_dir / "test_ambig.key"
    settings = PersonRecognitionSettings(
        db_path=str(db_file),
        key_path=str(key_file),
        recognition_threshold=0.40,
        ambiguity_margin=0.15,  # require 0.15 margin
        min_samples_per_enrollment=2,
    )
    db = FaceDatabase(settings)
    v_a, v_a_sim, _ = unit_embeddings

    # Enroll two people with almost identical reference embeddings
    db.enroll_person("PersonOne", [v_a, v_a])
    db.enroll_person("PersonTwo", [v_a_sim, v_a_sim])

    pipeline = PersonRecognitionPipeline(settings=settings, database=db)
    is_known, _, name, _, status = pipeline.match_embedding(v_a)
    # Because PersonOne and PersonTwo are both high matches, margin should be violated
    assert is_known is False
    assert status == RecognitionStatus.LOW_CONFIDENCE


def test_face_tracker_association_and_expiration():
    tracker = FaceTracker(iou_threshold=0.3, max_lost_frames=3, announcement_cooldown_sec=1.0)
    det1 = FaceDetection((20, 20, 50, 50), (20, 20, 70, 70), [(0, 0)] * 5, 0.9)

    # Frame 1: new track created
    persons, events = tracker.associate_and_stabilize([(det1, True, "p1", "Alice", 0.9, RecognitionStatus.RECOGNIZED)], timestamp=1.0)
    assert len(persons) == 1
    assert persons[0].face_track_id == 1

    # Frame 2: consecutive update
    det1_moved = FaceDetection((22, 22, 50, 50), (22, 22, 72, 72), [(0, 0)] * 5, 0.9)
    persons2, _ = tracker.associate_and_stabilize([(det1_moved, True, "p1", "Alice", 0.9, RecognitionStatus.RECOGNIZED)], timestamp=1.1)
    assert len(persons2) == 1
    assert persons2[0].face_track_id == 1

    # Frame 3-6: missed frames trigger expiration
    for t in [1.2, 1.3, 1.4]:
        tracker.associate_and_stabilize([], timestamp=t)
    assert len(tracker._tracks) == 0


def test_announcement_cooldown_suppression():
    tracker = FaceTracker(announcement_cooldown_sec=5.0)
    det = FaceDetection((10, 10, 40, 40), (10, 10, 50, 50), [(0, 0)] * 5, 0.9)

    # First event emitted
    _, events1 = tracker.associate_and_stabilize([(det, True, "id_1", "Bob", 0.85, RecognitionStatus.RECOGNIZED)], timestamp=10.0)
    # Note: requires stabilization (hits >= 2)
    _, events2 = tracker.associate_and_stabilize([(det, True, "id_1", "Bob", 0.85, RecognitionStatus.RECOGNIZED)], timestamp=10.1)
    assert len(events2) == 1
    assert "Bob" in events2[0].message

    # Repeated observation within 5s cooldown must NOT emit another event
    _, events3 = tracker.associate_and_stabilize([(det, True, "id_1", "Bob", 0.85, RecognitionStatus.RECOGNIZED)], timestamp=12.0)
    assert len(events3) == 0

    # After cooldown expires (timestamp 15.5 >= 10.1 + 5.0), new event is emitted
    _, events4 = tracker.associate_and_stabilize([(det, True, "id_1", "Bob", 0.85, RecognitionStatus.RECOGNIZED)], timestamp=15.5)
    assert len(events4) == 1


def test_pipeline_end_to_end_synthetic(synthetic_face_image):
    settings = PersonRecognitionSettings(recognition_interval=1)
    pipeline = PersonRecognitionPipeline(settings=settings)
    result = pipeline.process(synthetic_face_image, timestamp=1.0)
    assert isinstance(result.persons, list)
    assert "detection_ms" in result.metrics
    assert "total_ms" in result.metrics


def test_cpu_device_selection_enforced():
    settings = PersonRecognitionSettings(device="cpu")
    recognizer = FaceRecognizer(settings)
    assert "CPU" in recognizer.active_device


# ============================================================================
# Personal Context Database Tests
# ============================================================================


def test_context_database_initialization(temp_dir):
    db_path = temp_dir / "test_context.db"
    context_db = PersonContextDatabase(db_path=db_path)
    assert db_path.exists()
    assert context_db.list_all_contexts() == []


def test_context_creation_and_retrieval(temp_dir):
    db_path = temp_dir / "test_context.db"
    context_db = PersonContextDatabase(db_path=db_path)
    profile_id = "test-prof-1234"

    ctx = context_db.set_context(
        profile_id=profile_id,
        relationship="Sister",
        occupation="Software Engineer",
        description="Wears glasses",
        notes="Prefers morning meetings",
    )
    assert ctx.profile_id == profile_id
    assert ctx.relationship == "Sister"
    assert ctx.occupation == "Software Engineer"
    assert ctx.description == "Wears glasses"
    assert ctx.notes == "Prefers morning meetings"

    retrieved = context_db.get_context(profile_id)
    assert retrieved is not None
    assert retrieved.relationship == "Sister"
    assert retrieved.occupation == "Software Engineer"


def test_context_profile_id_validation(temp_dir):
    # Verify that attempting to link context to a non-existent biometric profile fails
    db_file = temp_dir / "val_face_db.bin"
    key_file = temp_dir / "val_face_db.key"
    ctx_file = temp_dir / "val_context.db"

    settings = PersonRecognitionSettings(
        db_path=str(db_file),
        key_path=str(key_file),
        context_db_path=str(ctx_file),
    )
    face_db = FaceDatabase(settings)
    context_db = PersonContextDatabase(settings=settings, db_path=ctx_file)

    # Valid profile exists in face_db
    v = np.random.randn(128).astype(np.float32)
    v /= np.linalg.norm(v)
    valid_profile = face_db.enroll_person("Alice", [v] * 3)

    # Context linked to valid profile succeeds
    ctx = context_db.set_context(valid_profile.profile_id, relationship="Friend", face_db=face_db)
    assert ctx.relationship == "Friend"

    # Context linked to fake profile ID raises ValueError
    with pytest.raises(ValueError, match="does not exist in FaceDatabase"):
        context_db.set_context("non-existent-uuid", relationship="Stranger", face_db=face_db)


def test_context_multiple_facts_and_preferences(temp_dir):
    db_path = temp_dir / "test_facts_prefs.db"
    context_db = PersonContextDatabase(db_path=db_path)
    pid = "prof-multi-item"

    context_db.set_context(pid, relationship="Colleague")

    f1 = context_db.add_fact(pid, "Leads frontend engineering team", category="work")
    f2 = context_db.add_fact(pid, "Allergic to peanuts", category="medical")

    p1 = context_db.add_preference(pid, "Prefers asynchronous chat", category="communication")
    p2 = context_db.add_preference(pid, "Drinks black coffee", category="food")

    ctx = context_db.get_context(pid)
    assert ctx is not None
    assert len(ctx.facts) == 2
    assert ctx.facts[0].fact == "Leads frontend engineering team"
    assert ctx.facts[0].category == "work"
    assert ctx.facts[1].category == "medical"

    assert len(ctx.preferences) == 2
    assert ctx.preferences[0].preference == "Prefers asynchronous chat"
    assert ctx.preferences[1].category == "food"

    # Test removing individual items
    assert context_db.remove_fact(f1.id) is True
    assert context_db.remove_preference(p2.id) is True

    updated_ctx = context_db.get_context(pid)
    assert len(updated_ctx.facts) == 1
    assert updated_ctx.facts[0].id == f2.id
    assert len(updated_ctx.preferences) == 1
    assert updated_ctx.preferences[0].id == p1.id


def test_context_partial_updates(temp_dir):
    db_path = temp_dir / "test_partial.db"
    context_db = PersonContextDatabase(db_path=db_path)
    pid = "prof-partial"

    context_db.set_context(pid, relationship="Friend", occupation="Designer", notes="Original note")

    # Update only relationship and occupation; notes and description should be preserved
    updated = context_db.update_profile_details(pid, relationship="Best Friend")
    assert updated.relationship == "Best Friend"
    assert updated.occupation == "Designer"
    assert updated.notes == "Original note"


def test_context_deletion_and_cascade(temp_dir):
    db_path = temp_dir / "test_cascade.db"
    context_db = PersonContextDatabase(db_path=db_path)
    pid = "prof-delete-cascade"

    context_db.set_context(pid, relationship="Neighbor")
    f = context_db.add_fact(pid, "Has a golden retriever")
    p = context_db.add_preference(pid, "Likes morning walks")

    assert context_db.get_context(pid) is not None

    # Delete context
    assert context_db.delete_context(pid) is True
    assert context_db.get_context(pid) is None

    # Verify foreign key cascade wiped facts and preferences
    with context_db._get_connection() as conn:
        f_count = conn.execute("SELECT COUNT(*) FROM person_facts WHERE profile_id = ?;", (pid,)).fetchone()[0]
        p_count = conn.execute("SELECT COUNT(*) FROM person_preferences WHERE profile_id = ?;", (pid,)).fetchone()[0]
        assert f_count == 0
        assert p_count == 0


def test_context_orphan_handling_and_sync(temp_dir):
    db_file = temp_dir / "sync_face_db.bin"
    key_file = temp_dir / "sync_face_db.key"
    ctx_file = temp_dir / "sync_context.db"

    settings = PersonRecognitionSettings(
        db_path=str(db_file),
        key_path=str(key_file),
        context_db_path=str(ctx_file),
    )
    face_db = FaceDatabase(settings)
    context_db = PersonContextDatabase(settings=settings, db_path=ctx_file)

    v1 = np.random.randn(128).astype(np.float32)
    v1 /= np.linalg.norm(v1)
    p1 = face_db.enroll_person("Person_A", [v1] * 3)

    v2 = np.random.randn(128).astype(np.float32)
    v2 /= np.linalg.norm(v2)
    p2 = face_db.enroll_person("Person_B", [v2] * 3)

    context_db.set_context(p1.profile_id, relationship="Sister", face_db=face_db)
    context_db.set_context(p2.profile_id, relationship="Doctor", face_db=face_db)

    assert len(context_db.list_all_contexts()) == 2

    # Delete Person_A from biometric database
    face_db.delete_profile(p1.profile_id)

    # Sync with FaceDatabase prunes Person_A's context record
    pruned = context_db.sync_with_face_db(face_db)
    assert pruned == 1
    assert context_db.get_context(p1.profile_id) is None
    assert context_db.get_context(p2.profile_id) is not None


def test_context_persistence_across_instances(temp_dir):
    db_path = temp_dir / "persist_context.db"
    pid = "persist-id"

    # Instance 1 writes
    db1 = PersonContextDatabase(db_path=db_path)
    db1.set_context(pid, relationship="Caregiver", occupation="Nurse")
    db1.add_fact(pid, "Available 8am to 4pm")

    # Instance 2 reads
    db2 = PersonContextDatabase(db_path=db_path)
    ctx = db2.get_context(pid)
    assert ctx is not None
    assert ctx.relationship == "Caregiver"
    assert ctx.occupation == "Nurse"
    assert len(ctx.facts) == 1
    assert ctx.facts[0].fact == "Available 8am to 4pm"


def test_context_corrupted_database_handling(temp_dir):
    db_path = temp_dir / "corrupted_context.db"
    # Write garbage bytes to simulate corrupted file
    db_path.write_bytes(b"THIS_IS_CORRUPTED_GARBAGE_DATA_NOT_SQLITE")

    # Initialization should detect corruption and gracefully create a clean database
    context_db = PersonContextDatabase(db_path=db_path)
    assert context_db.list_all_contexts() == []

    # Writing to the recovered database works
    ctx = context_db.set_context("recovered-prof", relationship="Friend")
    assert ctx.relationship == "Friend"


def test_pipeline_integration_with_context(temp_dir):
    db_file = temp_dir / "pipe_face_db.bin"
    key_file = temp_dir / "pipe_face_db.key"
    ctx_file = temp_dir / "pipe_context.db"

    settings = PersonRecognitionSettings(
        db_path=str(db_file),
        key_path=str(key_file),
        context_db_path=str(ctx_file),
        recognition_interval=1,
    )
    face_db = FaceDatabase(settings)
    context_db = PersonContextDatabase(settings=settings, db_path=ctx_file)

    # Enroll profile in face_db
    v = np.random.randn(128).astype(np.float32)
    v /= np.linalg.norm(v)
    profile = face_db.enroll_person("Sarah", [v] * 3)

    # Attach context in context_db
    context_db.set_context(profile.profile_id, relationship="Sister", face_db=face_db)

    # Run pipeline with both databases
    pipeline = PersonRecognitionPipeline(settings=settings, database=face_db, context_database=context_db)

    # Simulate detected face matches enrolled profile
    det = FaceDetection((10, 10, 50, 50), (10, 10, 60, 60), [(0, 0)] * 5, 0.95)
    # Feed detection to tracker
    persons, events = pipeline.tracker.associate_and_stabilize(
        [(det, True, profile.profile_id, "Sarah", 0.92, RecognitionStatus.RECOGNIZED)],
        timestamp=100.0,
    )
    # Frame 2 to stabilize
    persons, events = pipeline.tracker.associate_and_stabilize(
        [(det, True, profile.profile_id, "Sarah", 0.92, RecognitionStatus.RECOGNIZED)],
        timestamp=100.1,
    )

    # Manually attach context as done in pipeline.process()
    for p in persons:
        if p.is_known and p.identity_id:
            p.context = pipeline.context_database.get_context(p.identity_id, face_db=pipeline.database)

    for event in events:
        if event.person.context and event.person.context.relationship:
            event.message = f"{event.person.display_name} ({event.person.context.relationship}) is in front of you."

    assert len(persons) == 1
    assert persons[0].is_known is True
    assert persons[0].context is not None
    assert persons[0].context.relationship == "Sister"

    assert len(events) == 1
    assert "Sarah (Sister) is in front of you." in events[0].message

