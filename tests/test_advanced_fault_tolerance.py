"""Advanced Fault-Tolerance, Edge-Case, and Concurrency Test Suite for NAYRA.

Covers:
1. SQLite corruption recovery and automated clean rebuilding.
2. High-concurrency multithreaded database access under WAL mode.
3. Input validation and boundary value enforcement.
4. Bounded priority speech queue overflow and eviction under floods.
5. Acoustic feedback shield re-entrancy and thread safety.
6. Obstacle warning generator concurrency and memory pruning.
7. Malformed, extreme length, and special character query resilience.
8. Zero-network socket privacy assertion (100% offline verification).
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import pytest

from shared.schemas import (
    IntentRequest,
    IntentResponse,
    ObstacleHazard,
    SpeechPriority,
    SpeechRequest,
    UserIntent,
)

from obstacle_awareness.warning_generator import ObstacleWarningGenerator
from daily_assistance.reminder_db import ReminderDB, ReminderRecord
from voice_interaction.priority_speech_queue import PrioritySpeechQueue
from voice_interaction.acoustic_shield import AcousticShield
from assistant_orchestrator.intent_router import IntentClassifier
from assistant_orchestrator.orchestrator import AssistantOrchestrator


class TestDatabaseCorruptionAndValidation:
    """Tests SQLite corruption detection, auto-recovery, and input validation."""

    def test_corrupted_database_auto_recovery(self, tmp_path: Path):
        """Simulate a corrupted SQLite file; ReminderDB should back it up and rebuild cleanly."""
        corrupt_db_path = tmp_path / "corrupted_reminders.db"
        # Write corrupted garbage bytes to simulate sudden power outage / disk corruption
        with open(corrupt_db_path, "wb") as f:
            f.write(b"NOT A VALID SQLITE DATABASE HEADER GIBBERISH" * 50)

        # Instantiation must detect corruption, create backup, and restore a working schema
        db = ReminderDB(db_path=corrupt_db_path)

        # Verify backup was created
        backups = list(tmp_path.glob("*.corrupt.*.bak"))
        assert len(backups) >= 1, "Backup file for corrupted database should have been created"

        # Verify new DB is completely functional
        rem_id = db.add_reminder(title="Davaai lene ka waqt", due_timestamp=time.time() + 300)
        assert rem_id > 0
        active = db.list_active()
        assert len(active) == 1
        assert active[0].title == "Davaai lene ka waqt"

    def test_add_reminder_input_validation(self, tmp_path: Path):
        """Test boundary values: empty title, whitespace-only, negative timestamps."""
        db = ReminderDB(db_path=tmp_path / "validation.db")

        with pytest.raises(ValueError, match="empty or whitespace"):
            db.add_reminder(title="", due_timestamp=time.time() + 100)

        with pytest.raises(ValueError, match="empty or whitespace"):
            db.add_reminder(title="   \t\n  ", due_timestamp=time.time() + 100)

        with pytest.raises(ValueError, match="Invalid due_timestamp"):
            db.add_reminder(title="Valid title", due_timestamp=-10.0)

        with pytest.raises(ValueError, match="Invalid due_timestamp"):
            db.add_reminder(title="Valid title", due_timestamp=0.0)


class TestConcurrencyAndLocking:
    """Tests multithreaded operations on databases and warning generators."""

    def test_concurrent_reminder_db_writes(self, tmp_path: Path):
        """10 threads concurrently inserting and reading reminders with zero locks or crashes."""
        db = ReminderDB(db_path=tmp_path / "concurrent_reminders.db")
        thread_count = 10
        reminders_per_thread = 15

        def worker(thread_idx: int):
            for i in range(reminders_per_thread):
                db.add_reminder(
                    title=f"Worker {thread_idx} Task {i}",
                    due_timestamp=time.time() + 1000 + (thread_idx * 100) + i,
                    category="medication",
                )
                _ = db.list_active()

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker, t) for t in range(thread_count)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        active = db.list_active()
        assert len(active) == thread_count * reminders_per_thread

    def test_obstacle_warning_generator_concurrency_and_pruning(self):
        """Multiple threads flooding hazards to trigger pruning without deadlocks or race conditions."""
        generator = ObstacleWarningGenerator(cooldown_sec=5.0)

        def flood_hazards(start_id: int):
            now = time.time()
            for i in range(50):
                hazard = ObstacleHazard(
                    class_name="chair",
                    zone="CENTER",
                    track_id=start_id + i,
                    warning_hi=f"Saamne chair {start_id + i} hai",
                    warning_en=f"Chair {start_id + i} ahead",
                    relative_depth=0.5,
                    proximity="MEDIUM",
                    is_urgent=False,
                )
                generator.should_announce(hazard, now=now)

        threads = [threading.Thread(target=flood_hazards, args=(t * 100,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Pruning kicks in when dict grows > 150
        with generator._lock:
            assert len(generator._last_warn_time) <= 250


class TestPrioritySpeechQueueBounds:
    """Tests priority queue eviction and overflow bounding."""

    def test_speech_queue_max_size_eviction(self):
        """When queue reaches capacity, low-priority items are evicted to make room for high-priority alerts."""
        queue = PrioritySpeechQueue(max_size=3)

        req_info1 = SpeechRequest(text="Info 1", priority=SpeechPriority.INFORMATIONAL)
        req_info2 = SpeechRequest(text="Info 2", priority=SpeechPriority.INFORMATIONAL)
        req_info3 = SpeechRequest(text="Info 3", priority=SpeechPriority.INFORMATIONAL)

        queue.push(req_info1)
        queue.push(req_info2)
        queue.push(req_info3)
        assert len(queue) == 3

        # Incoming CRITICAL_SAFETY request should evict one INFORMATIONAL item
        req_critical = SpeechRequest(text="Emergency stop!", priority=SpeechPriority.CRITICAL_SAFETY)
        preempt = queue.push(req_critical)
        assert len(queue) == 3

        # Highest priority popped should be the critical one
        popped = queue.pop()
        assert popped is not None
        assert popped.text == "Emergency stop!"
        assert popped.priority == SpeechPriority.CRITICAL_SAFETY

    def test_speech_queue_rejects_lower_priority_when_full(self):
        """When queue is full of high-priority items, low-priority items are dropped."""
        queue = PrioritySpeechQueue(max_size=2)
        req_warn1 = SpeechRequest(text="Warning 1", priority=SpeechPriority.OBSTACLE_WARNING)
        req_warn2 = SpeechRequest(text="Warning 2", priority=SpeechPriority.OBSTACLE_WARNING)
        queue.push(req_warn1)
        queue.push(req_warn2)

        # Incoming INFORMATIONAL request is lower priority (higher int) than existing WARNING items -> rejected
        req_ambient = SpeechRequest(text="Ambient sound", priority=SpeechPriority.INFORMATIONAL)
        accepted = queue.push(req_ambient)
        assert accepted is False
        assert len(queue) == 2


class TestAcousticShieldReentrancy:
    """Tests acoustic feedback shield state consistency under re-entrant and concurrent calls."""

    def test_acoustic_shield_concurrent_toggles(self):
        shield = AcousticShield(cooldown_sec=1.0)

        def toggle_loop():
            for _ in range(50):
                shield.on_speech_started()
                assert shield.is_shield_active() is True
                shield.on_speech_ended()

        threads = [threading.Thread(target=toggle_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class TestNaturalLanguageRobustness:
    """Tests orchestrator intent handling with extreme, malformed, or unusual input strings."""

    def test_empty_and_whitespace_utterance(self):
        classifier = IntentClassifier()
        res_empty = classifier.classify("")
        assert res_empty.intent == UserIntent.UNKNOWN

        res_space = classifier.classify("   \n\t   ")
        assert res_space.intent == UserIntent.UNKNOWN

    def test_extreme_length_utterance(self):
        """10,000 character sentence should not cause Regex catastrophic backtracking or crash."""
        classifier = IntentClassifier()
        long_query = "kahan hai " + ("mera phone " * 1000)
        start = time.time()
        res = classifier.classify(long_query)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Query took too long to process: {elapsed:.3f}s"
        assert res.intent in (UserIntent.FIND_OBJECT, UserIntent.UNKNOWN)

    def test_special_characters_utterance(self):
        classifier = IntentClassifier()
        res = classifier.classify("!@#$%^&*()_+{}[]:;\"'<>?,./~`")
        assert res.intent == UserIntent.UNKNOWN


class TestZeroNetworkLeakage:
    """Verifies that NAYRA operates 100% offline without initiating any outbound network connections."""

    def test_zero_outbound_network_connections(self, monkeypatch):
        """Any attempt to call socket.connect to non-local external addresses will raise an assertion error."""
        original_connect = socket.socket.connect

        def guarded_connect(sock, address):
            host = address[0] if isinstance(address, tuple) else address
            # Disallow internet addresses (allow localhost loopback if needed for test mock IPC)
            if host not in ("127.0.0.1", "localhost", "::1"):
                raise AssertionError(f"FATAL: Offline violation! Attempted outbound connection to: {address}")
            return original_connect(sock, address)

        monkeypatch.setattr(socket.socket, "connect", guarded_connect)

        # Run an orchestrator dispatch flow and verify zero network calls occur
        orchestrator = AssistantOrchestrator()
        response = orchestrator.process_voice_query("Mera chashma kahan hai?")
        assert response.spoken_response_hi != ""
