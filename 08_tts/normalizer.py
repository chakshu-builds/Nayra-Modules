"""NAYRA Module 08 — Conservative Text Normalizer & Language Router.

Performs conservative text normalization preserving assistive domain semantics
(medicine names, dosages, serial identifiers, currency) and routes text to
verified Kokoro voices and language codes based on Unicode script analysis.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
import unicodedata


class TextNormalizer:
    """Conservative normalizer for assistive OCR and system speech output.

    Guarantees:
    1. Preserves original words, brand names, and technical terms.
    2. Preserves dosages (e.g., '500mg', '250 mg', '0.5% w/v').
    3. Handles Hindi danda ('।', '॥') and standard sentence punctuation.
    4. Cleans OCR line breaks and excessive whitespace without corrupting layout.
    5. Detects Devanagari, Latin, and mixed (Hinglish) text for language routing.
    """

    # Unicode ranges
    DEVANAGARI_RANGE = (0x0900, 0x097F)
    LATIN_RANGES = [
        (0x0041, 0x005A),  # A-Z
        (0x0061, 0x007A),  # a-z
        (0x00C0, 0x00FF),  # Latin-1 Supplement letters
    ]

    # Pre-compiled regex patterns
    _RE_MULTISPACE = re.compile(r"[ \t]+")
    _RE_MULTILINE = re.compile(r"\n\s*\n+")
    _RE_OCR_LINEBREAK = re.compile(r"(?<=[^\n])\n(?=[^\n])")
    _RE_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF\u00AD]")
    _RE_DOUBLE_DANDA = re.compile(r"॥+")
    _RE_DANDA_SPACING = re.compile(r"([।!?])(?=[^\s।!?])")

    # Common currency mappings that should be read safely without altering values
    _RE_RUPEE_SYMBOL = re.compile(r"₹\s*([0-9]+(?:\.[0-9]+)?)")

    @classmethod
    def normalize(cls, text: str) -> str:
        """Conservatively normalize text for TTS synthesis.

        Args:
            text: Raw input text from OCR or system alert.

        Returns:
            Cleaned and normalized text preserving all semantic tokens.
        """
        if not text or not text.strip():
            return ""

        # 1. Unicode NFC normalization
        normalized = unicodedata.normalize("NFC", text)

        # 2. Strip non-printable / zero-width characters and soft hyphens
        normalized = cls._RE_ZERO_WIDTH.sub("", normalized)

        # 3. Replace non-breaking spaces with standard space
        normalized = normalized.replace("\u00A0", " ")

        # 4. Standardize Hindi danda
        # Replace double danda ॥ with single danda ।
        normalized = cls._RE_DOUBLE_DANDA.sub("।", normalized)

        # 5. Expand rupee symbol safely to 'रुपये ' (Hindi) or 'Rs. ' depending on context
        # We replace bare ₹ with 'रुपये ' if followed by digits for clear pronunciation
        normalized = cls._RE_RUPEE_SYMBOL.sub(r"रुपये \1", normalized)

        # 6. Normalize paragraph and line breaks:
        # Multiple blank lines become double danda / sentence pause
        normalized = cls._RE_MULTILINE.sub(" ।\n", normalized)

        # Single OCR line breaks: if a line ends without punctuation, treat it as a natural pause
        # e.g., 'Paracetamol\n500mg' -> 'Paracetamol. 500mg'
        def _line_break_replacer(match: re.Match) -> str:
            char_before = match.string[match.start() - 1]
            if char_before in ".।!?":
                return " "
            return ". "

        normalized = cls._RE_OCR_LINEBREAK.sub(_line_break_replacer, normalized)

        # 7. Ensure space after punctuation (।!?.) if immediately followed by word characters
        normalized = cls._RE_DANDA_SPACING.sub(r"\1 ", normalized)

        # 8. Collapse consecutive spaces
        normalized = cls._RE_MULTISPACE.sub(" ", normalized).strip()

        return normalized

    @classmethod
    def detect_script(cls, text: str) -> str:
        """Determine primary script of the input text via character codepoint inspection.

        Returns:
            'HINDI' if text is predominantly Devanagari.
            'ENGLISH' if text is predominantly Latin.
            'MIXED' if significant portions of both Devanagari and Latin exist (Hinglish).
            'UNKNOWN' if neither (e.g. pure digits, symbols).
        """
        if not text:
            return "UNKNOWN"

        devanagari_count = 0
        latin_count = 0

        for char in text:
            cp = ord(char)
            if cls.DEVANAGARI_RANGE[0] <= cp <= cls.DEVANAGARI_RANGE[1]:
                devanagari_count += 1
            elif any(start <= cp <= end for start, end in cls.LATIN_RANGES):
                latin_count += 1

        total_letters = devanagari_count + latin_count
        if total_letters == 0:
            return "UNKNOWN"

        dev_ratio = devanagari_count / total_letters
        lat_ratio = latin_count / total_letters

        # If both are present with at least 10% representation, classify as MIXED (Hinglish)
        if devanagari_count > 0 and latin_count > 0:
            if dev_ratio >= 0.10 and lat_ratio >= 0.10:
                return "MIXED"

        if devanagari_count > latin_count:
            return "HINDI"
        return "ENGLISH"

    @classmethod
    def resolve_voice_and_lang(
        cls, text: str, language_hint: Optional[str] = None
    ) -> Tuple[str, str]:
        """Resolve the optimal Kokoro voice ID and language identifier.

        Args:
            text: Normalized text to synthesize.
            language_hint: Optional manual language hint ('hi', 'en-us', 'auto').

        Returns:
            Tuple of (voice_name, language_code).
        """
        hint = (language_hint or "").lower().strip()

        if hint in ("hi", "hindi"):
            return "hf_alpha", "hi"
        if hint in ("en", "en-us", "english"):
            return "af_bella", "en-us"

        script = cls.detect_script(text)

        if script == "HINDI":
            return "hf_alpha", "hi"
        elif script == "MIXED":
            # For Hinglish, eSpeak-NG handles Latin words inside Hindi mode smoothly
            return "hf_alpha", "hi"
        else:
            # Default to English (af_bella) for Latin or unknown/numeric
            return "af_bella", "en-us"
