"""Unit tests for Module 16: Shared Voice Interaction Layer."""

from __future__ import annotations

import time
from unittest.mock import MagicMock
import pytest

from shared.schemas import SpeechPriority, SpeechRequest
from voice_interaction import (
    AcousticShield,
    PrioritySpeechQueue,
    VoiceManager,
)


def test_acoustic_shield_timing():
    shield = AcousticShield(cooldown_sec=0.2)
    t0 = 1000.0

    assert shield.is_shield_active(now=t0) is False

    # Start speaking
    shield.on_speech_started()
    assert shield.is_shield_active(now=t0) is True

    # End speaking -> cooldown active
    shield.on_speech_ended(now=t0)
    assert shield.is_shield_active(now=t0 + 0.1) is True
    # After cooldown expires
    assert shield.is_shield_active(now=t0 + 0.25) is False


def test_priority_speech_queue_order():
    pq = PrioritySpeechQueue()

    req_info = SpeechRequest(text="Ambient info", priority=SpeechPriority.INFORMATIONAL)
    req_warn = SpeechRequest(text="Obstacle left", priority=SpeechPriority.OBSTACLE_WARNING)
    req_crit = SpeechRequest(text="STOP! Hazard ahead", priority=SpeechPriority.CRITICAL_SAFETY)

    pq.push(req_info)
    pq.push(req_warn)
    pq.push(req_crit)

    # Highest priority (CRITICAL_SAFETY = 1) must be popped first
    pop1 = pq.pop()
    assert pop1 is not None and pop1.priority == SpeechPriority.CRITICAL_SAFETY

    # Second highest (OBSTACLE_WARNING = 3)
    pop2 = pq.pop()
    assert pop2 is not None and pop2.priority == SpeechPriority.OBSTACLE_WARNING

    # Third highest (INFORMATIONAL = 6)
    pop3 = pq.pop()
    assert pop3 is not None and pop3.priority == SpeechPriority.INFORMATIONAL


def test_voice_manager_lifecycle():
    played = []

    def mock_play(text: str, lang: str):
        played.append(text)

    vm = VoiceManager(playback_fn=mock_play)
    vm.start()

    try:
        vm.speak("Test spoken message", priority=SpeechPriority.USER_QUERY_RESPONSE)
        time.sleep(0.15)

        assert len(played) >= 1
        assert "Test spoken message" in played[0]
    finally:
        vm.shutdown()
