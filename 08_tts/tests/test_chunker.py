"""Unit tests for TextChunker."""

import pytest
from tts import TextChunker


def test_chunker_empty():
    chunker = TextChunker()
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_hindi_danda_splitting():
    chunker = TextChunker(max_chunk_chars=100)
    text = "नमस्ते। आपका स्वागत है। कृपया आगे बढ़िए।"
    chunks = chunker.chunk(text)
    assert len(chunks) == 3
    assert chunks[0] == "नमस्ते।"
    assert chunks[1] == "आपका स्वागत है।"
    assert chunks[2] == "कृपया आगे बढ़िए।"


def test_abbreviation_and_decimal_protection():
    chunker = TextChunker(max_chunk_chars=150)
    text = "Dr. Sharma prescribed 0.5% eye drops. Price is Rs. 150 per bottle."
    chunks = chunker.chunk(text)
    # Should split into exactly 2 sentences on the true period, not on Dr., 0.5, or Rs.
    assert len(chunks) == 2
    assert "Dr. Sharma prescribed 0.5% eye drops." in chunks[0]
    assert "Rs. 150" in chunks[1]


def test_long_sentence_clause_splitting():
    chunker = TextChunker(max_chunk_chars=60)
    long_sentence = (
        "This is the first clause of the statement, "
        "and here is the second extended clause that should be split cleanly, "
        "followed by the concluding remark."
    )
    chunks = chunker.chunk(long_sentence)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 65  # Slight tolerance for word boundaries


def test_medicine_instruction_chunking():
    chunker = TextChunker(max_chunk_chars=100)
    med_text = "Take Paracetamol 500mg twice daily after meals. If symptoms persist, consult your physician."
    chunks = chunker.chunk(med_text)
    assert len(chunks) == 2
    assert "Paracetamol 500mg" in chunks[0]
    assert "physician" in chunks[1]
