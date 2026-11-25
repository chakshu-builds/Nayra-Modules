"""NAYRA Module 09 — Voice Command Intent Definitions & Keyword Matcher.

Classifies transcribed speech into discrete NAYRA command intents:
- READ: "NAYRA, read the text", "Read this", "Mere liye padho"
- STOP: "Stop reading", "Text reading band karo"
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Optional, Tuple
import unicodedata


class CommandIntent(str, Enum):
    """Discrete user voice command intents."""
    READ = "READ"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"


class CommandMatcher:
    """Robust keyword and phrase matcher for English, Hindi, and Hinglish voice commands."""

    # Pre-compiled regular expressions for READ intent
    _RE_READ = [
        # English commands
        re.compile(r"\b(?:nayra\s*,?\s*)?read\s+(?:the\s+)?text\b", re.IGNORECASE),
        re.compile(r"\b(?:nayra\s*,?\s*)?read\s+this\b", re.IGNORECASE),
        re.compile(r"\b(?:please\s+)?read\s+(?:for\s+me|it)\b", re.IGNORECASE),
        # Hinglish / Romanized Hindi
        re.compile(r"\bmere\s+liye\s+(?:padho|parho|padhiye)\b", re.IGNORECASE),
        re.compile(r"\bmera?\s+liye\s+par?ho\b", re.IGNORECASE),
        re.compile(r"\bmy\s+lee\s+par?ho\b", re.IGNORECASE),  # Whisper phonetic variant
        # Devanagari Hindi
        re.compile(r"मेरे\s+लिए\s+पढ़ो"),
        re.compile(r"(?:यह\s+)?पढ़ो"),
        re.compile(r"पढ़िये"),
    ]

    # Pre-compiled regular expressions for STOP intent
    _RE_STOP = [
        # English commands
        re.compile(r"\b(?:nayra\s*,?\s*)?stop\s+reading\b", re.IGNORECASE),
        re.compile(r"\b(?:nayra\s*,?\s*)?stop\b", re.IGNORECASE),
        re.compile(r"\bpause\s+reading\b", re.IGNORECASE),
        # Hinglish / Romanized Hindi
        re.compile(r"\b(?:text\s+)?reading\s+band\s+karo\b", re.IGNORECASE),
        re.compile(r"\b(?:text\s+)?reading\s+banned\s+karo\b", re.IGNORECASE),  # Whisper phonetic variant
        re.compile(r"\bband\s+karo\b", re.IGNORECASE),
        re.compile(r"\brok\s+do\b", re.IGNORECASE),
        # Devanagari Hindi
        re.compile(r"(?:पढ़ना\s+)?बंद\s+करो"),
        re.compile(r"रोक\s+दो"),
    ]

    @classmethod
    def match_intent(cls, text: str) -> Tuple[CommandIntent, Optional[str]]:
        """Classify input text into CommandIntent.

        Args:
            text: Transcribed text string from STT.

        Returns:
            Tuple of (CommandIntent, matched_rule_description).
        """
        if not text or not text.strip():
            return CommandIntent.UNKNOWN, None

        cleaned = unicodedata.normalize("NFC", text.strip())

        # Check STOP patterns first for safety preemption
        for pattern in cls._RE_STOP:
            match = pattern.search(cleaned)
            if match:
                return CommandIntent.STOP, match.group(0)

        # Check READ patterns
        for pattern in cls._RE_READ:
            match = pattern.search(cleaned)
            if match:
                return CommandIntent.READ, match.group(0)

        return CommandIntent.UNKNOWN, None
