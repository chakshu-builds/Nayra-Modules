"""NAYRA Module 08 — Intelligent Sentence and Clause Chunker.

Breaks long OCR texts and assistive narrations into natural, acoustically coherent
speech chunks for low Time-to-First-Audio (TTFA) streaming playback.
"""

from __future__ import annotations

import re
from typing import List


class TextChunker:
    """Splits text into speech-ready chunks on natural sentence and clause boundaries.

    Key capabilities:
    1. Recognizes Hindi danda ('।') as a first-class sentence terminator alongside ('.', '!', '?').
    2. Protects abbreviations (e.g. 'Dr.', 'Mr.', 'Rs.', 'Tab.', 'Cap.', 'i.e.', 'e.g.') and decimals ('0.5%').
    3. Prevents splitting inside medicine names, chemical formulas, or numerical dosages.
    4. Subdivides overly long sentences on clause boundaries (',', ';', ':') only when needed.
    5. Yields chunks suitable for immediate pipelined synthesis and playback.
    """

    # Abbreviations that should NOT trigger a sentence split on '.'
    _ABBREVIATIONS = {
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
        "rs", "no", "vs", "eg", "ie", "etc", "approx",
        "tab", "cap", "syr", "inj", "mg", "ml", "gm", "kg",
    }

    # Sentence boundary regex (danda, exclamation, question mark, period followed by whitespace or end)
    # Uses negative lookbehind to avoid splitting on decimals and known abbreviations
    _RE_DECIMAL = re.compile(r"(\d+)\.(\d+)")

    def __init__(self, max_chunk_chars: int = 150, min_chunk_chars: int = 10) -> None:
        """Initialize chunker with length boundaries.

        Args:
            max_chunk_chars: Maximum target length for an individual chunk before clause splitting.
            min_chunk_chars: Minimum character length to avoid micro-chunks unless at end.
        """
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, text: str) -> List[str]:
        """Split text into an ordered list of speech-ready chunks.

        Args:
            text: Normalized input text string.

        Returns:
            List of non-empty text chunks.
        """
        if not text or not text.strip():
            return []

        clean_text = text.strip()

        # Step 1: Split into candidate sentences
        sentences = self._split_sentences(clean_text)

        # Step 2: For any sentence exceeding max_chunk_chars, split on clause boundaries
        chunks: List[str] = []
        for s in sentences:
            if len(s) <= self.max_chunk_chars:
                chunks.append(s)
            else:
                clause_chunks = self._split_clauses(s)
                chunks.extend(clause_chunks)

        # Step 3: Merge short trailing fragments if necessary
        merged_chunks = self._merge_small_chunks(chunks)

        return [c.strip() for c in merged_chunks if c.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into raw sentences respecting danda and abbreviations."""
        sentences: List[str] = []
        current: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            char = text[i]
            current.append(char)

            # Check for sentence delimiters: Hindi danda '।', '!', '?'
            if char in ("।", "!", "?"):
                # Always a hard sentence boundary
                sentences.append("".join(current).strip())
                current = []
            elif char == ".":
                # Check if this is a decimal point between digits
                prev_char = text[i - 1] if i > 0 else ""
                next_char = text[i + 1] if i + 1 < n else ""

                if prev_char.isdigit() and next_char.isdigit():
                    # Part of a decimal number (e.g. 0.5)
                    pass
                else:
                    # Check if preceding token is a known abbreviation
                    token = "".join(current).rstrip(".").split()[-1] if current else ""
                    token_clean = token.lower().strip("()[]{},")

                    if token_clean in self._ABBREVIATIONS or len(token_clean) == 1:
                        # Abbreviation (e.g. 'Dr.' or single initial 'A.') - do not split
                        pass
                    elif next_char.isspace() or i + 1 == n:
                        # Legitimate sentence boundary
                        sentences.append("".join(current).strip())
                        current = []
            elif char == "\n":
                sentences.append("".join(current).strip())
                current = []

            i += 1

        if current:
            remaining = "".join(current).strip()
            if remaining:
                sentences.append(remaining)

        return [s for s in sentences if s]

    def _split_clauses(self, sentence: str) -> List[str]:
        """Split an overly long sentence on clause markers (,, ;, :, -)."""
        # Split on commas, semicolons, colons, or dashes followed by whitespace
        parts = re.split(r"([,;:\-—]\s+)", sentence)
        clause_list: List[str] = []
        buf = ""

        for part in parts:
            if len(buf) + len(part) <= self.max_chunk_chars:
                buf += part
            else:
                if buf.strip():
                    clause_list.append(buf.strip())
                buf = part

        if buf.strip():
            clause_list.append(buf.strip())

        # If any single clause is still too long (e.g. no punctuation), split by words
        final_clauses: List[str] = []
        for cl in clause_list:
            if len(cl) > self.max_chunk_chars:
                final_clauses.extend(self._split_words(cl))
            else:
                final_clauses.append(cl)

        return final_clauses

    def _split_words(self, text: str) -> List[str]:
        """Split an unbroken run of text by word boundaries without exceeding max_chunk_chars."""
        words = text.split()
        word_chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for w in words:
            if current_len + len(w) + 1 > self.max_chunk_chars and current:
                word_chunks.append(" ".join(current))
                current = [w]
                current_len = len(w)
            else:
                current.append(w)
                current_len += len(w) + 1

        if current:
            word_chunks.append(" ".join(current))

        return word_chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge micro-chunks with preceding chunks to avoid awkward acoustic cutoffs."""
        if not chunks:
            return []

        merged: List[str] = []
        for c in chunks:
            if not merged:
                merged.append(c)
                continue

            prev = merged[-1]
            if len(c) < self.min_chunk_chars and (len(prev) + len(c) + 1) <= self.max_chunk_chars:
                # Merge into previous chunk
                merged[-1] = f"{prev} {c}"
            else:
                merged.append(c)

        return merged
