"""NAYRA Shared Schemas and Data Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import time
from typing import Any, Dict, List, Optional, Tuple


class SpeechPriority(IntEnum):
    """Priority levels for spoken voice output. Lower value = higher priority."""
    CRITICAL_SAFETY = 1         # Immediate physical hazard stop warnings
    EMERGENCY_INTERACTION = 2   # SOS confirmation prompts and cancellation
    OBSTACLE_WARNING = 3        # Directional obstacle alerts (left, ahead, right)
    USER_QUERY_RESPONSE = 4     # Direct answers to user speech inquiries
    REMINDER = 5                # Scheduled medication and daily reminders
    INFORMATIONAL = 6           # Scene descriptions, ambient tips, startup banners


@dataclass
class SpeechRequest:
    """Request for speech synthesis and playback."""
    text: str
    language: str = "hi"         # 'hi' or 'en'
    priority: SpeechPriority = SpeechPriority.USER_QUERY_RESPONSE
    utterance_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    spoken_callback: Optional[Any] = None


class UserIntent(str, Enum):
    """Recognized user intent categories."""
    FIND_OBJECT = "FIND_OBJECT"
    LIST_OBJECTS = "LIST_OBJECTS"
    OBSTACLE_CHECK = "OBSTACLE_CHECK"
    DESCRIBE_SCENE = "DESCRIBE_SCENE"
    READ_TEXT = "READ_TEXT"
    IDENTIFY_PERSON = "IDENTIFY_PERSON"
    SET_REMINDER = "SET_REMINDER"
    LIST_REMINDERS = "LIST_REMINDERS"
    DELETE_REMINDER = "DELETE_REMINDER"
    SNOOZE_REMINDER = "SNOOZE_REMINDER"
    TRIGGER_SOS = "TRIGGER_SOS"
    CONFIRM_SOS = "CONFIRM_SOS"
    CANCEL_SOS = "CANCEL_SOS"
    GENERAL_CHAT = "GENERAL_CHAT"
    WAKE_GREETING = "WAKE_GREETING"
    STOP = "STOP"
    SILENT_IGNORE = "SILENT_IGNORE"
    STATUS_QUERY = "STATUS_QUERY"
    HELP = "HELP"
    UNKNOWN = "UNKNOWN"


@dataclass
class IntentRequest:
    """Incoming user command parsed by the assistant orchestrator."""
    raw_query: str
    intent: UserIntent
    target_entity: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    detected_language: str = "en"
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntentResponse:
    """Result returned by an assistive module back to the orchestrator."""
    success: bool
    intent: UserIntent
    spoken_response_hi: str
    spoken_response_en: str
    priority: SpeechPriority = SpeechPriority.USER_QUERY_RESPONSE
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ObstacleHazard:
    """Detected physical obstacle in user path."""
    track_id: Optional[int]
    class_name: str
    zone: str                   # 'LEFT', 'CENTER', 'RIGHT'
    proximity: str              # 'NEAR', 'MEDIUM', 'FAR', 'UNKNOWN'
    relative_depth: float       # 0.0 (far) to 1.0 (very close)
    is_urgent: bool             # True if directly ahead and close
    warning_hi: str
    warning_en: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReminderItem:
    """Caregiver or user configured reminder."""
    id: Optional[int]
    title: str
    reminder_time_epoch: float
    time_str: str               # e.g. "20:00" or "8:00 PM"
    recurrence: str = "ONCE"    # 'ONCE', 'DAILY', 'HOURLY'
    is_completed: bool = False
    is_snoozed: bool = False
    snooze_until: Optional[float] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class EmergencyContact:
    """Emergency contact for Safety SOS."""
    name: str
    phone: str
    relationship: str = "Caregiver"
    is_primary: bool = True
