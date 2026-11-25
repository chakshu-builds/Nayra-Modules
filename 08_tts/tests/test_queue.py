"""Unit tests for AudioQueue priority scheduling and preemption."""

import pytest
from tts import AudioQueue, PriorityLevel, RequestStatus, SpeechRequest


def test_queue_priority_ordering():
    q = AudioQueue(max_size=10)

    r_normal = SpeechRequest(text="Normal OCR reading", priority=PriorityLevel.NORMAL)
    r_critical = SpeechRequest(text="Critical drop-off ahead", priority=PriorityLevel.CRITICAL, interrupt=False)
    r_perception = SpeechRequest(text="Person identified: John", priority=PriorityLevel.PERCEPTION)

    # Enqueue in reverse order
    q.enqueue(r_normal)
    q.enqueue(r_perception)
    q.enqueue(r_critical)

    # Should pop Critical (P0) first, Perception (P1) second, Normal (P2) third
    p1 = q.get(timeout=0.1)
    p2 = q.get(timeout=0.1)
    p3 = q.get(timeout=0.1)

    assert p1.priority == PriorityLevel.CRITICAL
    assert p1.text == "Critical drop-off ahead"
    assert p2.priority == PriorityLevel.PERCEPTION
    assert p3.priority == PriorityLevel.NORMAL


def test_queue_fifo_within_same_priority():
    q = AudioQueue(max_size=10)

    r1 = SpeechRequest(text="Sentence 1", priority=PriorityLevel.NORMAL)
    r2 = SpeechRequest(text="Sentence 2", priority=PriorityLevel.NORMAL)
    r3 = SpeechRequest(text="Sentence 3", priority=PriorityLevel.NORMAL)

    q.enqueue(r1)
    q.enqueue(r2)
    q.enqueue(r3)

    assert q.get(timeout=0.1).text == "Sentence 1"
    assert q.get(timeout=0.1).text == "Sentence 2"
    assert q.get(timeout=0.1).text == "Sentence 3"


def test_critical_preemption_purges_lower_priority():
    q = AudioQueue(max_size=10)

    # Enqueue normal OCR items
    r_ocr1 = SpeechRequest(text="OCR Line 1", priority=PriorityLevel.NORMAL)
    r_ocr2 = SpeechRequest(text="OCR Line 2", priority=PriorityLevel.NORMAL)
    q.enqueue(r_ocr1)
    q.enqueue(r_ocr2)

    assert q.qsize() == 2

    # Enqueue Critical alert with interrupt=True
    r_alert = SpeechRequest(text="Stop! Obstacle immediately ahead.", priority=PriorityLevel.CRITICAL, interrupt=True)
    q.enqueue(r_alert)

    # Pending OCR items should have been purged from queue
    assert q.qsize() == 1
    popped = q.get(timeout=0.1)
    assert popped.priority == PriorityLevel.CRITICAL
    assert popped.text == "Stop! Obstacle immediately ahead."
    assert r_ocr1.status == RequestStatus.CANCELLED
    assert r_ocr2.status == RequestStatus.CANCELLED


def test_duplicate_suppression():
    q = AudioQueue(max_size=10, duplicate_window_sec=2.0)

    r1 = SpeechRequest(text="Repeated warning", priority=PriorityLevel.PERCEPTION)
    r2 = SpeechRequest(text="Repeated warning", priority=PriorityLevel.PERCEPTION)

    assert q.enqueue(r1) is True
    assert q.enqueue(r2) is False
    assert r2.status == RequestStatus.CANCELLED
    assert q.qsize() == 1


def test_queue_overflow_eviction():
    q = AudioQueue(max_size=2)

    r1 = SpeechRequest(text="Task 1", priority=PriorityLevel.NORMAL)
    r2 = SpeechRequest(text="Task 2", priority=PriorityLevel.NORMAL)
    assert q.enqueue(r1) is True
    assert q.enqueue(r2) is True

    # Attempt to insert high priority when full -> should evict lowest priority
    r_high = SpeechRequest(text="High Priority", priority=PriorityLevel.PERCEPTION)
    assert q.enqueue(r_high) is True
    assert q.qsize() == 2

    first = q.get(timeout=0.1)
    assert first.priority == PriorityLevel.PERCEPTION
