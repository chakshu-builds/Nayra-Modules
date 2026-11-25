"""Unit tests for Module 17: Assistant Orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from shared.schemas import IntentRequest, IntentResponse, SpeechPriority, UserIntent
from assistant_orchestrator import (
    AssistantOrchestrator,
    IntentClassifier,
)


def test_intent_classifier_multilingual():
    classifier = IntentClassifier()

    # Hindi object search
    req1 = classifier.classify("Mera phone kahan hai?")
    assert req1.intent == UserIntent.FIND_OBJECT
    assert req1.target_entity == "phone"
    assert req1.detected_language == "hi"

    # English scene description
    req2 = classifier.classify("What is in front of me?")
    assert req2.intent == UserIntent.DESCRIBE_SCENE
    assert req2.detected_language == "en"

    # Hindi obstacle check
    req3 = classifier.classify("Kya rasta saaf hai?")
    assert req3.intent == UserIntent.OBSTACLE_CHECK

    # SOS trigger
    req4 = classifier.classify("Madad karo emergency hai")
    assert req4.intent == UserIntent.TRIGGER_SOS

    # SOS confirmation dialog flow
    req5 = classifier.classify("Haan madad chahiye", pending_intent=UserIntent.TRIGGER_SOS)
    assert req5.intent == UserIntent.CONFIRM_SOS

    # Reminder
    req6 = classifier.classify("Mujhe 10 minute baad dawai lene ki yaad dilana")
    assert req6.intent == UserIntent.SET_REMINDER
    assert req6.target_entity == "dawai"


def test_orchestrator_routing():
    voice_mock = MagicMock()
    orchestrator = AssistantOrchestrator(voice_output=voice_mock)

    # Register mock module for reminders
    mock_reminder_mod = MagicMock()
    mock_reminder_mod.initialize.return_value = True
    mock_reminder_mod.process_query.return_value = IntentResponse(
        success=True,
        intent=UserIntent.SET_REMINDER,
        spoken_response_hi="Reminder set kar diya gaya hai.",
        spoken_response_en="Reminder has been set.",
    )
    orchestrator.register_module("14_daily_assistance", mock_reminder_mod)
    assert orchestrator.initialize_all() is True

    # Process query
    resp = orchestrator.process_voice_query("Mujhe dawai ki yaad dilana")
    assert resp.success is True
    assert "Reminder set" in resp.spoken_response_hi
    mock_reminder_mod.process_query.assert_called_once()
    voice_mock.speak.assert_called()


def test_orchestrator_fallback():
    orchestrator = AssistantOrchestrator()
    resp = orchestrator.process_voice_query("Abrakadabra unknown intent 1234")
    assert resp.success is True
    assert "NAYRA" in resp.spoken_response_hi
