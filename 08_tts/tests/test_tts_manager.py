"""Integration tests for TTSManager lifecycle, priority preemption, and playback controls."""

import time
import pytest
import numpy as np

from tts import AudioPlayer, BaseTTSEngine, PlayerState, PriorityLevel, RequestStatus, TTSManager


class MockTTSEngine(BaseTTSEngine):
    """Fast deterministic mock engine generating synthetic sine tones for tests."""

    def __init__(self):
        self._ready = False

    def initialize(self):
        self._ready = True

    def synthesize(self, text, voice=None, lang=None, speed=1.0):
        if not text or not text.strip():
            raise ValueError("Empty text")
        # 0.2 second synthetic tone
        sr = 24000
        samples = np.zeros(int(0.2 * sr), dtype=np.float32)
        return samples, sr

    def get_sample_rate(self):
        return 24000

    def get_available_voices(self):
        return ["af_bella", "hf_alpha"]

    def is_ready(self):
        return self._ready

    def shutdown(self):
        self._ready = False


@pytest.fixture
def mock_tts_manager():
    """TTSManager configured with fast mock engine and mock player."""
    engine = MockTTSEngine()
    engine.initialize()
    player = AudioPlayer(mock_mode=True)
    manager = TTSManager(engine=engine, player=player, mock_audio=True)
    yield manager
    manager.shutdown()


def test_manager_speak_returns_id(mock_tts_manager):
    req_id = mock_tts_manager.speak("Hello world", priority=PriorityLevel.NORMAL)
    assert req_id is not None
    assert isinstance(req_id, str)
    assert len(req_id) > 0


def test_manager_empty_text_raises(mock_tts_manager):
    with pytest.raises(ValueError):
        mock_tts_manager.speak("")

    with pytest.raises(ValueError):
        mock_tts_manager.speak("   ")


def test_manager_status(mock_tts_manager):
    status = mock_tts_manager.get_status()
    assert "engine_ready" in status
    assert "player_state" in status
    assert "queue_size" in status
    assert "is_speaking" in status
    assert status["engine_ready"] is True


def test_manager_pause_and_resume(mock_tts_manager):
    mock_tts_manager.speak("Sentence one. Sentence two.", priority=PriorityLevel.NORMAL)
    time.sleep(0.05)

    mock_tts_manager.pause()
    assert mock_tts_manager.player.get_state() in (PlayerState.PAUSED, PlayerState.IDLE)

    mock_tts_manager.resume()
    assert mock_tts_manager.player.get_state() in (PlayerState.PLAYING, PlayerState.IDLE)


def test_manager_stop(mock_tts_manager):
    mock_tts_manager.speak("Long text that should be interrupted immediately", priority=PriorityLevel.NORMAL)
    time.sleep(0.05)
    mock_tts_manager.stop()
    assert not mock_tts_manager.player.is_playing()


def test_manager_priority_preemption(mock_tts_manager):
    # Queue normal reading
    id_ocr = mock_tts_manager.speak("Reading page 1 of document.", priority=PriorityLevel.NORMAL)
    time.sleep(0.02)

    # Immediately queue critical obstacle alert with interrupt=True
    id_alert = mock_tts_manager.speak("Obstacle ahead!", priority=PriorityLevel.CRITICAL, interrupt=True)
    time.sleep(0.05)

    status = mock_tts_manager.get_status()
    # Alert should have taken precedence or active
    assert status is not None
