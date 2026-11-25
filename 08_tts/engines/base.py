"""NAYRA Module 08 — Abstract TTS Engine Base Interface & Exceptions.

Defines the contract for offline text-to-speech backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np


class TTSEngineError(Exception):
    """Base exception for all TTS engine failures."""
    pass


class ModelNotFoundError(TTSEngineError):
    """Raised when model weights or voice vector assets cannot be located."""
    pass


class SynthesisError(TTSEngineError):
    """Raised when phonemization, tokenization, or neural inference fails."""
    pass


class BaseTTSEngine(ABC):
    """Abstract interface for local offline TTS engines."""

    @abstractmethod
    def initialize(self) -> None:
        """Load neural network weights, session graphs, and voice vectors."""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        lang: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text into raw PCM audio samples.

        Args:
            text: Normalized text to synthesize.
            voice: Voice identifier (e.g. 'af_bella', 'hf_alpha').
            lang: Language code (e.g. 'en-us', 'hi').
            speed: Speech rate multiplier in [0.5, 2.0].

        Returns:
            Tuple of (audio_samples_float32, sample_rate_hz).
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return native output audio sample rate in Hz."""
        pass

    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Return list of supported voice identifiers."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the engine is initialized and ready for inference."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release ONNX sessions, tensors, and memory allocations."""
        pass
