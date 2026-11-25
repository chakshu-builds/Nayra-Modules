"""Unit tests for Module 12: Contextual Scene Understanding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from unittest.mock import MagicMock
import pytest

from shared.schemas import IntentRequest, SpeechPriority, UserIntent
from scene_understanding import (
    SceneAnalyzer,
    SceneDescription,
    SceneService,
)


@dataclass
class MockDetection:
    class_name: str
    box_xyxy: Tuple[int, int, int, int]
    confidence: float = 0.85
    track_id: int | None = None


def test_scene_analyzer_full_synthesis():
    analyzer = SceneAnalyzer()
    detections = [
        MockDetection("laptop", (220, 200, 320, 300)),
        MockDetection("table", (180, 250, 450, 450)),
        MockDetection("chair", (480, 200, 580, 400)),
        MockDetection("person", (50, 100, 180, 450)),
    ]

    desc = analyzer.analyze(
        detections=detections,
        frame_width=600,
        frame_height=480,
        ocr_text="DANGER EXIT",
    )

    assert isinstance(desc, SceneDescription)
    assert desc.people_count == 1
    assert "laptop" in desc.objects_by_zone["CENTER"]
    assert "chair" in desc.objects_by_zone["RIGHT"]
    assert "DANGER EXIT" in desc.ocr_text

    # Verify natural Hindi description elements
    assert "vyakti" in desc.spoken_hi or "Saamne" in desc.spoken_hi
    assert "DANGER EXIT" in desc.spoken_hi


def test_scene_analyzer_directional_query():
    analyzer = SceneAnalyzer()
    detections = [
        MockDetection("chair", (500, 200, 580, 400)), # RIGHT
        MockDetection("laptop", (220, 200, 320, 300)), # CENTER
    ]

    # Query right side
    desc_right = analyzer.analyze(detections, frame_width=600, frame_height=480, target_zone="RIGHT")
    assert "kursi" in desc_right.spoken_hi or "chair" in desc_right.spoken_en

    # Query left side (empty)
    desc_left = analyzer.analyze(detections, frame_width=600, frame_height=480, target_zone="LEFT")
    assert "nahi dikh rahi" in desc_left.spoken_hi or "nothing" in desc_left.spoken_en


def test_scene_service_intent_handling():
    voice_mock = MagicMock()
    service = SceneService(voice_output=voice_mock)
    service.initialize()

    detections = [
        MockDetection("chair", (250, 200, 350, 400)),
    ]
    service.update_scene_context(detections, frame_width=600, frame_height=480)

    req = IntentRequest(
        intent=UserIntent.DESCRIBE_SCENE,
        raw_query="Mere saamne kya hai?",
        detected_language="hi",
    )
    resp = service.process_query(req)

    assert resp.success is True
    assert "kursi" in resp.spoken_response_hi or "chair" in resp.spoken_response_en
    voice_mock.speak.assert_called_once()
    assert voice_mock.speak.call_args[1]["priority"] == SpeechPriority.USER_QUERY_RESPONSE
