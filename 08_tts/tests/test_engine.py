"""Integration tests for KokoroEngine ONNX model."""

import pytest
import numpy as np

from tts import KokoroEngine, ModelNotFoundError, SynthesisError


@pytest.fixture(scope="module")
def kokoro_engine():
    """Shared KokoroEngine instance on CPU."""
    engine = KokoroEngine(use_gpu=False)
    engine.initialize()
    yield engine
    engine.shutdown()


def test_engine_initialization(kokoro_engine):
    assert kokoro_engine.is_ready()
    assert kokoro_engine.get_sample_rate() == 24000
    voices = kokoro_engine.get_available_voices()
    assert "af_bella" in voices
    assert "hf_alpha" in voices


def test_english_synthesis(kokoro_engine):
    samples, sr = kokoro_engine.synthesize("Hello from NAYRA.", voice="af_bella", lang="en-us")
    assert sr == 24000
    assert isinstance(samples, np.ndarray)
    assert len(samples) > 0
    assert samples.dtype == np.float32


def test_hindi_synthesis(kokoro_engine):
    samples, sr = kokoro_engine.synthesize("नमस्ते, आपका स्वागत है।", voice="hf_alpha", lang="hi")
    assert sr == 24000
    assert isinstance(samples, np.ndarray)
    assert len(samples) > 0


def test_hinglish_synthesis(kokoro_engine):
    samples, sr = kokoro_engine.synthesize("यह दवाई twice a day लेनी है।", voice="hf_alpha", lang="hi")
    assert sr == 24000
    assert isinstance(samples, np.ndarray)
    assert len(samples) > 0


def test_empty_input_validation(kokoro_engine):
    with pytest.raises(ValueError):
        kokoro_engine.synthesize("")

    with pytest.raises(ValueError):
        kokoro_engine.synthesize("   ")


def test_invalid_voice_validation(kokoro_engine):
    with pytest.raises(ValueError, match="not found in available voices"):
        kokoro_engine.synthesize("Hello", voice="non_existent_voice_123")
