"""Unit tests for Module 18: Unified Application."""

from __future__ import annotations

import time
from unittest.mock import MagicMock
import pytest

from shared.schemas import UserIntent
from nayra_app import AppConfig, NayraApplication


def test_app_lifecycle():
    config = AppConfig(enable_camera=False, default_language="hi")
    app = NayraApplication(config=config)

    assert app.initialize() is True
    app.start()

    try:
        # 1. Orientation query -> 13_alzheimer_assistance
        resp_orient = app.process_command("Main kahan hoon?")
        assert resp_orient.success is True
        assert "ghar par" in resp_orient.spoken_response_hi.lower()

        # 2. Obstacle query -> 11_obstacle_awareness
        resp_obs = app.process_command("Kya saamne rasta saaf hai?")
        assert resp_obs.success is True
        assert "saaf" in resp_obs.spoken_response_hi.lower() or "clear" in resp_obs.spoken_response_en.lower()

        # 3. Scene query -> 12_scene_understanding
        resp_scene = app.process_command("Mere saamne kya hai?")
        assert resp_scene.success is True

        # 4. Reminder query -> 14_daily_assistance
        resp_rem = app.process_command("Mujhe 5 minute baad dawai lene ki yaad dilana")
        assert resp_rem.success is True
        assert "reminder" in resp_rem.spoken_response_hi.lower()

        # 5. SOS query -> 15_safety_sos
        resp_sos = app.process_command("Madad karo emergency hai")
        assert resp_sos.success is True
        assert "emergency" in resp_sos.spoken_response_hi.lower() or "sach mein" in resp_sos.spoken_response_hi.lower()
    finally:
        app.shutdown()
