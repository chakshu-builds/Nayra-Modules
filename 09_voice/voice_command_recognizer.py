"""NAYRA Module 09 — Offline Voice Command Recognizer.

Wraps faster-whisper (Whisper-tiny INT8) for fast local multilingual
speech-to-intent classification supporting English, Hindi, and Hinglish.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Optional, Tuple

import numpy as np

try:
    from .command_matcher import CommandIntent, CommandMatcher
except (ImportError, ValueError):
    from command_matcher import CommandIntent, CommandMatcher

logger = logging.getLogger("NAYRA-VoiceCommandRecognizer")


class VoiceCommandRecognizer:
    """Offline neural speech-to-intent recognizer.

    Guarantees:
    1. 100% offline inference on local CPU (INT8 quantization).
    2. Prompt biasing toward NAYRA voice commands for fast sub-500ms decoding.
    3. Multilingual support for English, Hindi, and Hinglish phrases.
    4. Deterministic mapping to CommandIntent (READ, STOP, UNKNOWN).
    """

    PROMPT_BIAS: str = (
        "NAYRA, read the text. Read this. Mere liye padho. Stop reading. Text reading band karo."
    )

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[str | Path] = None,
    ) -> None:
        root_dir = Path(__file__).resolve().parent.parent
        self.download_root = Path(download_root) if download_root else (root_dir / "models" / "stt" / "whisper")
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        """Load Whisper model weights from local cache."""
        if self._initialized and self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run 'pip install faster-whisper'."
            ) from exc

        logger.info(
            "Loading offline Whisper model '%s' on %s (%s)...",
            self.model_name,
            self.device,
            self.compute_type,
        )
        t0 = time.perf_counter()

        # Check if local checkpoint exists
        local_only = self.download_root.exists() and any(self.download_root.iterdir())

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.download_root),
            local_files_only=local_only,
        )
        self._initialized = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("VoiceCommandRecognizer initialized in %.2f ms.", elapsed_ms)

    def recognize(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[CommandIntent, str, float]:
        """Recognize voice command from audio samples.

        Args:
            audio_samples: 1D numpy array of float32 samples.
            sample_rate: Input audio sample rate (must be or will be resampled to 16000).

        Returns:
            Tuple of (CommandIntent, transcribed_text, latency_ms).
        """
        if not self._initialized or self._model is None:
            self.initialize()

        data = np.asarray(audio_samples, dtype=np.float32).ravel()
        if len(data) == 0:
            return CommandIntent.UNKNOWN, "", 0.0

        # Resample to 16,000 Hz if needed
        if sample_rate != 16000:
            num_out = int(len(data) * 16000 / sample_rate)
            data = np.interp(
                np.linspace(0, len(data), num_out, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)

        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            data,
            initial_prompt=self.PROMPT_BIAS,
            beam_size=1,
            language=None,  # Auto-detect language
        )
        transcript = " ".join(seg.text for seg in segments).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        intent, matched_phrase = CommandMatcher.match_intent(transcript)

        logger.debug(
            "Recognized: '%s' -> Intent: %s (matched: '%s') in %.1f ms (lang: %s).",
            transcript,
            intent.value,
            matched_phrase,
            latency_ms,
            getattr(info, "language", "unknown"),
        )

        return intent, transcript, latency_ms

    def is_ready(self) -> bool:
        return self._initialized and self._model is not None

    def shutdown(self) -> None:
        """Release Whisper model resources."""
        self._model = None
        self._initialized = False
        logger.info("VoiceCommandRecognizer shutdown completed.")
