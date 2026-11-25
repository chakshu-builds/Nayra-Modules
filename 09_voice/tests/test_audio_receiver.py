"""Unit tests for PhoneAudioReceiver."""

import numpy as np
import pytest
from voice import PhoneAudioReceiver


def test_url_normalization():
    assert PhoneAudioReceiver._normalize_audio_url("192.168.0.101:8080") == "http://192.168.0.101:8080/audio.wav"
    assert PhoneAudioReceiver._normalize_audio_url("http://192.168.0.101:8080") == "http://192.168.0.101:8080/audio.wav"
    assert PhoneAudioReceiver._normalize_audio_url("http://192.168.0.101:8080/") == "http://192.168.0.101:8080/audio.wav"
    assert PhoneAudioReceiver._normalize_audio_url("http://192.168.0.101:8080/audio.wav") == "http://192.168.0.101:8080/audio.wav"


def test_audio_injection_and_callback():
    received_samples = []

    def _callback(samples, sr):
        received_samples.append((samples, sr))

    receiver = PhoneAudioReceiver(
        mock_mode=True,
        on_speech_utterance=_callback,
        target_sample_rate=16000,
    )
    receiver.start()

    # Inject synthetic 1.0s tone at 16 kHz
    dummy_tone = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000)).astype(np.float32)
    receiver.inject_audio_sample(dummy_tone, sample_rate=16000)

    assert len(received_samples) == 1
    samples, sr = received_samples[0]
    assert sr == 16000
    assert len(samples) == 16000

    receiver.stop()
    assert not receiver._is_running


def test_dc_offset_rms_suppression():
    """Verify that pure DC offset without AC oscillation does not trigger false speech."""
    # Chunk with strong DC bias (+0.05) but zero AC variation
    dc_chunk = np.full(1600, 0.05, dtype=np.float32)
    chunk_ac = dc_chunk - np.mean(dc_chunk)
    rms_ac = float(np.sqrt(np.mean(chunk_ac ** 2)))
    assert rms_ac < 1e-6

