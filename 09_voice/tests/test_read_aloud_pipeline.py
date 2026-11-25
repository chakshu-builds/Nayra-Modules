"""Unit & Integration tests for NAYRA Phase 3 End-to-End Read Aloud Pipeline."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import ocr
from voice import CommandIntent, PhoneAudioReceiver, ReadAloudPipeline, VoiceCommandRecognizer
from tts import KokoroEngine, PriorityLevel, TTSManager


class MockVoiceRecognizer:
    """Mock recognizer returning controlled intents without neural inference overhead."""

    def __init__(self, default_intent: CommandIntent = CommandIntent.READ, default_text: str = "Read this") -> None:
        self.default_intent = default_intent
        self.default_text = default_text
        self.call_count = 0
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def recognize(self, samples: np.ndarray, sample_rate: int = 16000):
        self.call_count += 1
        return self.default_intent, self.default_text, 15.0


@pytest.fixture
def dummy_audio_samples() -> np.ndarray:
    """Generate 1.0s of synthetic 16kHz sine tone."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    samples = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    return samples


def test_pipeline_initialization():
    """Verify clean initialization, default parameters, and mock mode."""
    rec = MockVoiceRecognizer()
    pipeline = ReadAloudPipeline(voice_recognizer=rec, mock_mode=True, debounce_sec=1.5)

    status = pipeline.get_status()
    assert not status["is_running"]
    assert status["total_reads"] == 0
    assert status["total_stops"] == 0
    assert status["total_utterances"] == 0

    assert pipeline.start()
    assert rec.initialized
    status_started = pipeline.get_status()
    assert status_started["is_running"]

    pipeline.stop()
    assert not pipeline.get_status()["is_running"]


def test_read_command_with_text(dummy_audio_samples):
    """Verify that a READ command captures frame, runs OCR, and enqueues TTS."""
    events = []
    def on_event(name, data):
        events.append((name, data))

    rec = MockVoiceRecognizer(default_intent=CommandIntent.READ, default_text="NAYRA, read the text")
    tts_mock = MagicMock(spec=TTSManager)
    tts_mock.speak.return_value = "req_123"

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.combined_text = "Exit to Platform 2"
    mock_result.primary_language.value = "ENGLISH"
    mock_ocr_fn = MagicMock(return_value=mock_result)

    test_frame = np.zeros((200, 400, 3), dtype=np.uint8)

    pipeline = ReadAloudPipeline(
        voice_recognizer=rec,
        tts_manager=tts_mock,
        frame_provider=lambda: test_frame,
        ocr_fn=mock_ocr_fn,
        mock_mode=True,
        debounce_sec=1.0,
        on_event=on_event,
    )

    pipeline.start()
    try:
        pipeline.trigger_mock_command(dummy_audio_samples)
        time.sleep(0.3)

        assert rec.call_count == 1
        assert mock_ocr_fn.called
        assert tts_mock.speak.called
        args, kwargs = tts_mock.speak.call_args
        spoken_text = kwargs.get("text") or (args[0] if args else "")
        assert spoken_text == "Exit to Platform 2"
        assert kwargs.get("priority") == PriorityLevel.NORMAL
        assert kwargs.get("language_hint") == "en-us"

        status = pipeline.get_status()
        assert status["total_reads"] == 1
        assert status["last_intent"] == "READ"
    finally:
        pipeline.stop()


def test_read_command_empty_scene(dummy_audio_samples):
    """Verify fallback verbal prompt when no text is found in frame."""
    rec = MockVoiceRecognizer(default_intent=CommandIntent.READ, default_text="Read this")
    tts_mock = MagicMock(spec=TTSManager)

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.combined_text = ""
    mock_ocr_fn = MagicMock(return_value=mock_result)

    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    pipeline = ReadAloudPipeline(
        voice_recognizer=rec,
        tts_manager=tts_mock,
        frame_provider=lambda: test_frame,
        ocr_fn=mock_ocr_fn,
        mock_mode=True,
        debounce_sec=0.5,
    )

    pipeline.start()
    try:
        pipeline.trigger_mock_command(dummy_audio_samples)
        time.sleep(0.3)

        assert tts_mock.speak.called
        args, kwargs = tts_mock.speak.call_args
        spoken_text = kwargs.get("text") or (args[0] if args else "")
        assert spoken_text == "No text detected."
    finally:
        pipeline.stop()


def test_stop_command_halts_tts(dummy_audio_samples):
    """Verify that a STOP command calls tts_manager.stop()."""
    rec = MockVoiceRecognizer(default_intent=CommandIntent.STOP, default_text="Stop reading")
    tts_mock = MagicMock(spec=TTSManager)

    pipeline = ReadAloudPipeline(
        voice_recognizer=rec,
        tts_manager=tts_mock,
        mock_mode=True,
    )

    pipeline.start()
    try:
        pipeline.trigger_mock_command(dummy_audio_samples)
        time.sleep(0.3)

        assert tts_mock.stop.called
        status = pipeline.get_status()
        assert status["total_stops"] == 1
        assert status["last_intent"] == "STOP"
    finally:
        pipeline.stop()


def test_read_command_debounce(dummy_audio_samples):
    """Verify debounce rejects duplicate triggers within cooldown window."""
    events = []
    def on_event(name, data):
        events.append(name)

    rec = MockVoiceRecognizer(default_intent=CommandIntent.READ, default_text="Read this")
    tts_mock = MagicMock(spec=TTSManager)
    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.combined_text = "Pharmacy"
    mock_result.primary_language.value = "ENGLISH"
    mock_ocr_fn = MagicMock(return_value=mock_result)

    pipeline = ReadAloudPipeline(
        voice_recognizer=rec,
        tts_manager=tts_mock,
        frame_provider=lambda: test_frame,
        ocr_fn=mock_ocr_fn,
        mock_mode=True,
        debounce_sec=2.0,
        on_event=on_event,
    )

    pipeline.start()
    try:
        # First command
        pipeline.trigger_mock_command(dummy_audio_samples)
        time.sleep(0.2)
        # Immediate second command (should debounce)
        pipeline.trigger_mock_command(dummy_audio_samples)
        time.sleep(0.2)

        # OCR should only have been called ONCE
        assert mock_ocr_fn.call_count == 1
        assert "command_read_debounced" in events
        status = pipeline.get_status()
        assert status["total_reads"] == 1
    finally:
        pipeline.stop()


def test_missing_frame_provider_notification(dummy_audio_samples):
    """Verify verbal error message when camera returns None."""
    rec = MockVoiceRecognizer(default_intent=CommandIntent.READ, default_text="Read this")
    tts_mock = MagicMock(spec=TTSManager)

    pipeline = ReadAloudPipeline(
        voice_recognizer=rec,
        tts_manager=tts_mock,
        frame_provider=lambda: None,  # Camera unavailable
        mock_mode=True,
        debounce_sec=0.5,
    )

    pipeline.start()
    try:
        pipeline.trigger_mock_command(dummy_audio_samples)
        time.sleep(0.3)

        assert tts_mock.speak.called
        args, kwargs = tts_mock.speak.call_args
        spoken_text = kwargs.get("text") or (args[0] if args else "")
        assert "Camera frame not available." in spoken_text
    finally:
        pipeline.stop()


def test_unknown_command_ignored(dummy_audio_samples):
    """Verify unknown speech commands do not trigger OCR or TTS."""
    rec = MockVoiceRecognizer(default_intent=CommandIntent.UNKNOWN, default_text="Hello how are you")
    tts_mock = MagicMock(spec=TTSManager)

    pipeline = ReadAloudPipeline(
        voice_recognizer=rec,
        tts_manager=tts_mock,
        mock_mode=True,
    )

    pipeline.start()
    try:
        pipeline.trigger_mock_command(dummy_audio_samples)
        time.sleep(0.3)

        assert not tts_mock.speak.called
        assert not tts_mock.stop.called
        status = pipeline.get_status()
        assert status["last_intent"] == "UNKNOWN"
        assert status["total_reads"] == 0
        assert status["total_stops"] == 0
    finally:
        pipeline.stop()


def test_real_models_end_to_end():
    """Integration test executing real Whisper STT + real OCR on image1.jpg + real Kokoro TTS."""
    img_path = Path(__file__).resolve().parent.parent.parent / "image1.jpg"
    assert img_path.exists(), "image1.jpg not found"
    frame = cv2.imread(str(img_path))
    assert frame is not None and frame.size > 0

    # Initialize real Kokoro engine to synthesize synthetic speech command
    kokoro = KokoroEngine(use_gpu=False)
    kokoro.initialize()
    try:
        raw_audio, sr = kokoro.synthesize("NAYRA, read the text.", voice="af_bella", lang="en-us")
        num_out = int(len(raw_audio) * 16000 / sr)
        audio_16k = np.interp(
            np.linspace(0, len(raw_audio), num_out, endpoint=False),
            np.arange(len(raw_audio)),
            raw_audio,
        ).astype(np.float32)

        stop_audio, stop_sr = kokoro.synthesize("Stop reading.", voice="af_bella", lang="en-us")
        stop_num_out = int(len(stop_audio) * 16000 / stop_sr)
        stop_audio_16k = np.interp(
            np.linspace(0, len(stop_audio), stop_num_out, endpoint=False),
            np.arange(len(stop_audio)),
            stop_audio,
        ).astype(np.float32)
    finally:
        kokoro.shutdown()

    events = []
    def on_event(name, data):
        events.append((name, data))

    # Initialize real pipeline with mock audio player (so CI/test runs without physical speaker lock)
    real_tts = TTSManager(mock_audio=True)
    real_recognizer = VoiceCommandRecognizer()

    pipeline = ReadAloudPipeline(
        voice_recognizer=real_recognizer,
        tts_manager=real_tts,
        frame_provider=lambda: frame,
        preferred_ocr_provider="CPU",
        mock_mode=True,
        debounce_sec=1.0,
        on_event=on_event,
    )

    # Pre-warm OCR on CPU so cold initialization doesn't timeout the test
    _ = ocr.read_text(frame, preferred_provider="CPU")

    pipeline.start()
    try:
        # Trigger READ
        pipeline.trigger_mock_command(audio_16k)

        # Allow time for Whisper transcription and OCR DBNet/PP-OCR execution
        deadline = time.time() + 25.0
        while time.time() < deadline:
            st = pipeline.get_status()
            if st["total_reads"] > 0 and any(e[0] == "tts_enqueued" for e in events):
                break
            time.sleep(0.2)

        st = pipeline.get_status()
        assert st["total_reads"] == 1
        assert st["last_intent"] == "READ"

        # Verify OCR output was enqueued for TTS
        tts_events = [e[1] for e in events if e[0] == "tts_enqueued"]
        assert len(tts_events) > 0
        spoken_text = tts_events[0]["text"]
        assert "राजीव" in spoken_text or "Rajiv" in spoken_text

        # Now test STOP command
        pipeline.trigger_mock_command(stop_audio_16k)
        stop_deadline = time.time() + 25.0
        while time.time() < stop_deadline:
            st_after_stop = pipeline.get_status()
            if st_after_stop["total_stops"] > 0:
                break
            time.sleep(0.2)
        st_after_stop = pipeline.get_status()
        assert st_after_stop["total_stops"] == 1
        assert st_after_stop["last_intent"] == "STOP"

    finally:
        pipeline.close()
        real_recognizer.shutdown()


def test_read_aloud_pipeline_with_network_audio_player():
    """Verify ReadAloudPipeline seamlessly coordinates with NetworkAudioPlayer."""
    from tts import NetworkAudioPlayer
    import socket

    def _get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    port = _get_free_port()
    net_player = NetworkAudioPlayer(host="127.0.0.1", port=port, mock_mode=True)
    tts_mgr = TTSManager(player=net_player, mock_audio=True)
    mock_rec = MockVoiceRecognizer(default_intent=CommandIntent.READ, default_text="Read this")

    dummy_frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    mock_ocr = MagicMock()
    mock_ocr.return_value = MagicMock(
        success=True,
        combined_text="Network audio player test output",
        primary_language="ENGLISH",
        execution_providers={"CPUExecutionProvider": 1},
    )

    pipeline = ReadAloudPipeline(
        voice_recognizer=mock_rec,
        tts_manager=tts_mgr,
        frame_provider=lambda: dummy_frame,
        ocr_fn=mock_ocr,
        debounce_sec=0.5,
        mock_mode=True,
    )

    try:
        pipeline.start()
        dummy_audio = np.zeros(16000, dtype=np.float32)
        pipeline.trigger_mock_command(dummy_audio)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            st = pipeline.get_status()
            if st["total_reads"] > 0:
                break
            time.sleep(0.05)

        st = pipeline.get_status()
        assert st["total_reads"] == 1
        assert st["last_intent"] == "READ"
        assert mock_ocr.call_count == 1

        # Test STOP command triggers net_player stop
        pipeline.trigger_mock_command(dummy_audio)  # Intent can be changed or stop triggered
        pipeline._handle_stop_command()
        assert net_player.get_state().value in ("IDLE", "STOPPED")

    finally:
        pipeline.close()

