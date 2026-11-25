"""Unit tests for Module 11: Obstacle Awareness & Guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from unittest.mock import MagicMock
import pytest

from shared.schemas import IntentRequest, SpeechPriority, UserIntent
from obstacle_awareness import (
    ObstacleDetector,
    ObstacleWarningGenerator,
    ObstacleService,
)


@dataclass
class MockDetection:
    class_name: str
    box_xyxy: Tuple[int, int, int, int]
    confidence: float = 0.85
    track_id: int | None = None


def test_zone_classification():
    detector = ObstacleDetector()
    # Frame width 600: left < 200, center 200-400, right > 400
    assert detector.classify_zone((50, 100, 150, 200), frame_width=600) == "LEFT"
    assert detector.classify_zone((250, 100, 350, 200), frame_width=600) == "CENTER"
    assert detector.classify_zone((450, 100, 550, 200), frame_width=600) == "RIGHT"


def test_obstacle_filtering_and_urgency():
    detector = ObstacleDetector()
    # Physical obstacle in center near the bottom
    chair = MockDetection(class_name="chair", box_xyxy=(220, 250, 380, 470), track_id=1)
    # Handheld non-obstacle
    phone = MockDetection(class_name="cell phone", box_xyxy=(250, 300, 290, 350), track_id=2)

    hazards = detector.evaluate_obstacles([chair, phone], frame_width=600, frame_height=480)
    # Phone should be excluded, chair retained
    assert len(hazards) == 1
    assert hazards[0].class_name == "chair"
    assert hazards[0].zone == "CENTER"
    assert hazards[0].proximity in ("NEAR", "MEDIUM")
    assert hazards[0].is_urgent is True


def test_warning_repetition_suppression():
    generator = ObstacleWarningGenerator(cooldown_sec=5.0)
    detector = ObstacleDetector()
    chair = MockDetection(class_name="chair", box_xyxy=(220, 250, 380, 470), track_id=1)
    hazards = detector.evaluate_obstacles([chair], frame_width=600, frame_height=480)
    top_hazard = hazards[0]

    # First announcement should pass
    t0 = 1000.0
    assert generator.should_announce(top_hazard, now=t0) is True

    # Immediate next check should be suppressed
    assert generator.should_announce(top_hazard, now=t0 + 1.0) is False

    # After cooldown, should pass again
    assert generator.should_announce(top_hazard, now=t0 + 6.0) is True


def test_clear_path_query_response():
    generator = ObstacleWarningGenerator()
    hi, en = generator.formulate_query_summary([])
    assert "saaf" in hi
    assert "clear" in en.lower()


def test_obstacle_service_lifecycle_and_query():
    mock_voice = MagicMock()
    service = ObstacleService(voice_output=mock_voice)
    assert service.initialize() is True

    chair = MockDetection(class_name="chair", box_xyxy=(220, 250, 380, 470), track_id=1)
    service.evaluate_frame([chair], frame_width=600, frame_height=480, auto_speak=True)

    # Voice output should have been invoked for urgent obstacle
    assert mock_voice.speak.called

    # User query
    req = IntentRequest(
        raw_query="Mere saamne koi obstacle hai?",
        intent=UserIntent.OBSTACLE_CHECK,
    )
    resp = service.process_query(req)
    assert resp.success is True
    assert "chair" in resp.spoken_response_hi or "Savdhaan" in resp.spoken_response_hi
    assert "chair" in resp.spoken_response_en

    service.shutdown()
    assert len(service._latest_hazards) == 0
