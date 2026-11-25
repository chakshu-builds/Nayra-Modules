"""NAYRA Shared Interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import numpy as np

from shared.schemas import IntentRequest, IntentResponse, SpeechPriority, SpeechRequest


class IVoiceOutput(ABC):
    """Abstract interface for speech synthesis and playback."""

    @abstractmethod
    def speak(
        self,
        text: str,
        language: str = "hi",
        priority: SpeechPriority = SpeechPriority.USER_QUERY_RESPONSE,
        block: bool = False,
        spoken_callback: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Enqueue or immediately speak text with specified priority."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Halt playback and purge pending speech queue."""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """Check if audio output is currently playing."""
        pass


class IAssistiveModule(ABC):
    """Standard lifecycle and query interface implemented by all assistive domain modules."""

    @property
    @abstractmethod
    def module_name(self) -> str:
        """Name of the module."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize models, database connections, and background threads."""
        pass

    @abstractmethod
    def process_query(self, request: IntentRequest) -> IntentResponse:
        """Execute domain logic for an incoming user query."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanly flush databases and release system resources."""
        pass
