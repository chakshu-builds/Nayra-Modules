"""Integration test for VoiceCommandRecognizer on target commands."""

import numpy as np
import pytest

import tts
from tts import KokoroEngine
from voice import CommandIntent, VoiceCommandRecognizer


@pytest.fixture(scope="module")
def recognizer():
    rec = VoiceCommandRecognizer()
    rec.initialize()
    yield rec
    rec.shutdown()


@pytest.fixture(scope="module")
def kokoro_engine():
    engine = KokoroEngine(use_gpu=False)
    engine.initialize()
    yield engine
    engine.shutdown()


def _synthesize_16k(engine, text, voice, lang):
    samples, sr = engine.synthesize(text, voice=voice, lang=lang)
    num_out = int(len(samples) * 16000 / sr)
    return np.interp(
        np.linspace(0, len(samples), num_out, endpoint=False),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)


def test_recognize_read_english_01(recognizer, kokoro_engine):
    audio_16k = _synthesize_16k(kokoro_engine, "NAYRA, read the text.", "af_bella", "en-us")
    intent, transcript, latency = recognizer.recognize(audio_16k)
    assert intent == CommandIntent.READ
    assert latency > 0


def test_recognize_read_english_02(recognizer, kokoro_engine):
    audio_16k = _synthesize_16k(kokoro_engine, "Read this.", "af_bella", "en-us")
    intent, transcript, latency = recognizer.recognize(audio_16k)
    assert intent == CommandIntent.READ


def test_recognize_read_hindi(recognizer, kokoro_engine):
    audio_16k = _synthesize_16k(kokoro_engine, "मेरे लिए पढ़ो।", "hf_alpha", "hi")
    intent, transcript, latency = recognizer.recognize(audio_16k)
    assert intent == CommandIntent.READ


def test_recognize_stop_english(recognizer, kokoro_engine):
    audio_16k = _synthesize_16k(kokoro_engine, "Stop reading.", "af_bella", "en-us")
    intent, transcript, latency = recognizer.recognize(audio_16k)
    assert intent == CommandIntent.STOP


def test_recognize_stop_hinglish(recognizer, kokoro_engine):
    audio_16k = _synthesize_16k(kokoro_engine, "Text reading band karo.", "hf_alpha", "hi")
    intent, transcript, latency = recognizer.recognize(audio_16k)
    assert intent == CommandIntent.STOP


def test_recognize_empty(recognizer):
    intent, transcript, latency = recognizer.recognize(np.array([], dtype=np.float32))
    assert intent == CommandIntent.UNKNOWN
    assert transcript == ""
