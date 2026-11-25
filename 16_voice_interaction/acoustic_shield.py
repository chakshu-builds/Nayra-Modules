"""Acoustic Feedback Shield preventing Kokoro TTS playback from echoing into Whisper STT."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("NAYRA-AcousticShield")


class AcousticShield:
    """Manages audio gating during and immediately after TTS output."""

    def __init__(self, cooldown_sec: float = 1.2) -> None:
        self.cooldown_sec = cooldown_sec
        self._is_speaking = False
        self._speech_end_time = 0.0
        self._lock = threading.RLock()

    def on_speech_started(self) -> None:
        """Mark start of TTS audio playback."""
        with self._lock:
            self._is_speaking = True
            logger.debug("AcousticShield: TTS playback started (mic gated).")

    def on_speech_ended(self, now: float | None = None) -> None:
        """Mark end of TTS audio playback, triggering cooldown."""
        with self._lock:
            self._is_speaking = False
            self._speech_end_time = time.time() if now is None else now
            logger.debug(f"AcousticShield: TTS playback ended. Cooldown active for {self.cooldown_sec}s.")

    def is_shield_active(self, now: float | None = None) -> bool:
        """Check if microphone audio should currently be dropped."""
        with self._lock:
            if self._is_speaking:
                return True
            if now is None:
                now = time.time()
            return (now - self._speech_end_time) < self.cooldown_sec

    def allow_microphone_input(self, now: float | None = None) -> bool:
        """Return True if microphone is open (shield inactive), False if dropped."""
        return not self.is_shield_active(now=now)

    def should_drop_audio(self, now: float | None = None) -> bool:
        """Alias for is_shield_active."""
        return self.is_shield_active(now=now)
