"""Unit tests for TextNormalizer and Script Detection."""

import pytest
from tts import TextNormalizer


def test_empty_and_whitespace():
    assert TextNormalizer.normalize("") == ""
    assert TextNormalizer.normalize("   \n\t  ") == ""
    assert TextNormalizer.detect_script("") == "UNKNOWN"


def test_hindi_danda_normalization():
    raw = "नमस्ते॥ आपका स्वागत है।।"
    norm = TextNormalizer.normalize(raw)
    assert "॥" not in norm
    assert "नमस्ते।" in norm
    assert "स्वागत है।" in norm


def test_medicine_and_dosage_preservation():
    med_text = "Paracetamol 500mg Tab. Take 1 tablet twice a day."
    norm = TextNormalizer.normalize(med_text)
    assert "500mg" in norm
    assert "Paracetamol" in norm
    assert "twice a day" in norm


def test_ocr_linebreak_handling():
    raw_ocr = "Amoxicillin\n500mg\n\nTake after food."
    norm = TextNormalizer.normalize(raw_ocr)
    assert "Amoxicillin" in norm
    assert "500mg" in norm


def test_currency_preservation():
    raw = "कुल मूल्य ₹ 150.50 है।"
    norm = TextNormalizer.normalize(raw)
    assert "150.50" in norm
    assert "रुपये" in norm


def test_script_detection():
    # Pure Hindi
    assert TextNormalizer.detect_script("नमस्ते आप कैसे हैं") == "HINDI"

    # Pure English
    assert TextNormalizer.detect_script("Hello welcome to NAYRA") == "ENGLISH"

    # Mixed Hinglish
    assert TextNormalizer.detect_script("यह दवाई twice a day लेनी है") == "MIXED"
    assert TextNormalizer.detect_script("कृपया अपना ID card साथ रखें") == "MIXED"

    # Pure numbers/punctuation
    assert TextNormalizer.detect_script("12345 67890 !@#$") == "UNKNOWN"


def test_voice_resolution():
    voice, lang = TextNormalizer.resolve_voice_and_lang("नमस्ते")
    assert voice == "hf_alpha"
    assert lang == "hi"

    voice, lang = TextNormalizer.resolve_voice_and_lang("Hello NAYRA")
    assert voice == "af_bella"
    assert lang == "en-us"

    voice, lang = TextNormalizer.resolve_voice_and_lang("यह दवाई twice a day लेनी है")
    assert voice == "hf_alpha"
    assert lang == "hi"

    # Explicit hint override
    voice, lang = TextNormalizer.resolve_voice_and_lang("123", language_hint="hi")
    assert voice == "hf_alpha"
    assert lang == "hi"
