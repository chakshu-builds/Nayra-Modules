"""NAYRA Module 08 — Speech Request Schema & Data Contracts.

Defines the core data models for speech requests, priority levels,
and lifecycle states within the NAYRA Text-to-Speech (TTS) subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum
import time
from typing import Optional
import unicodedata
import uuid


class PriorityLevel(IntEnum):
    """Priority hierarchy for speech requests in NAYRA.

    Lower integer values represent strictly higher preemption priority:
    - CRITICAL (0): Obstacle alerts, drop-offs, stairs, safety hazards.
    - PERCEPTION (1): Facial recognition, object identification, user interactions.
    - NORMAL (2): OCR text reading, long form descriptions, background narration.
    """
    CRITICAL = 0
    PERCEPTION = 1
    NORMAL = 2


class RequestStatus(str, Enum):
    """Lifecycle state of an individual SpeechRequest."""
    PENDING = "PENDING"
    SYNTHESIZING = "SYNTHESIZING"
    PLAYING = "PLAYING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class SpeechRequest:
    """Represents a discrete speech synthesis and playback request.

    Attributes:
        request_id: Unique UUID or sequential identifier for tracking.
        text: Raw text string to synthesize and speak.
        language_hint: Optional manual language hint ('hi', 'en-us', 'auto').
        priority: PriorityLevel integer (0=CRITICAL, 1=PERCEPTION, 2=NORMAL).
        source: Component originating the request (e.g. 'ocr', 'safety', 'manual').
        interrupt: Whether this request immediately preempts active lower-priority playback.
        created_at: Timestamp (epoch seconds) when request was instantiated.
        status: Current processing state of the request.
        error_message: Optional diagnostic string if processing fails.
    """

    text: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    language_hint: Optional[str] = None
    priority: int = PriorityLevel.NORMAL
    source: str = "manual"
    interrupt: bool = False
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields and enforce Unicode NFC normalization."""
        # Gracefully handle SpeechPriority enum or out-of-range integer priorities
        if hasattr(self.priority, "value"):
            self.priority = self.priority.value
        if isinstance(self.priority, int) and self.priority not in (0, 1, 2):
            # Map high-level SpeechPriority: 1/2 -> 0 (urgent), 3/5 -> 1 (alerts), 4/6 -> 2 (normal)
            if self.priority in (1, 2):
                self.priority = 0
            elif self.priority in (3, 5):
                self.priority = 1
            else:
                self.priority = 2

        self.validate()
        self.text = unicodedata.normalize("NFC", self.text.strip())

    def validate(self) -> None:
        """Validate request integrity; raise ValueError if invalid."""
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("SpeechRequest 'text' must be a non-empty, non-whitespace string.")

        if not isinstance(self.priority, int) or self.priority not in (0, 1, 2):
            raise ValueError(f"SpeechRequest 'priority' must be 0, 1, or 2; got {self.priority}.")
