"""Typed data structures for face detection, embeddings, profiles, and recognition results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class RecognitionStatus(str, Enum):
    """Recognition decision and diagnostic status."""

    RECOGNIZED = "RECOGNIZED"
    UNKNOWN = "UNKNOWN"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FACE_NOT_VISIBLE = "FACE_NOT_VISIBLE"
    LOW_QUALITY = "LOW_QUALITY"
    PROCESSING_ERROR = "PROCESSING_ERROR"


@dataclass
class FaceDetection:
    """Detected face bounding box, 5 facial landmarks, and confidence score."""

    box_xywh: tuple[int, int, int, int]
    box_xyxy: tuple[int, int, int, int]
    landmarks_5: list[tuple[float, float]]  # right eye, left eye, nose, right mouth, left mouth
    confidence: float
    raw_array: list[float] = field(default_factory=list)


@dataclass
class FaceProfile:
    """Consent-enrolled familiar person profile with reference embedding vectors."""

    profile_id: str
    display_name: str
    embeddings: list[list[float]]  # list of 128-dimensional L2-normalized float embeddings
    enrolled_at: float = field(default_factory=time.time)
    sample_count: int = 1
    model_name: str = "sface_2021dec"
    embedding_dim: int = 128


@dataclass
class FactItem:
    """Individual personal fact about an enrolled person."""

    id: int
    fact: str
    category: str = "general"
    created_at: float = field(default_factory=time.time)


@dataclass
class PreferenceItem:
    """Individual personal preference of an enrolled person."""

    id: int
    preference: str
    category: str = "general"
    created_at: float = field(default_factory=time.time)


@dataclass
class PersonContext:
    """User-provided contextual metadata linked to an enrolled biometric profile."""

    profile_id: str
    relationship: str = ""
    occupation: str = ""
    description: str = ""
    notes: str = ""
    facts: list[FactItem] = field(default_factory=list)
    preferences: list[PreferenceItem] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class RecognizedPerson:
    """Per-face recognition outcome with temporal tracking association."""

    face_track_id: Optional[int]
    identity_id: Optional[str]
    display_name: str
    is_known: bool
    similarity_score: float
    detection_confidence: float
    recognition_status: RecognitionStatus
    bounding_box: tuple[int, int, int, int]
    landmarks: Optional[list[tuple[float, float]]] = None
    context: Optional[PersonContext] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecognitionEvent:
    """Discrete recognition event suitable for text-to-speech / audio announcement."""

    event_type: str  # "PERSON_RECOGNIZED", "UNKNOWN_PERSON_DETECTED"
    person: RecognizedPerson
    timestamp: float
    message: str


@dataclass
class PersonRecognitionResult:
    """Frame-level aggregated result containing recognized persons and new events."""

    persons: list[RecognizedPerson] = field(default_factory=list)
    events: list[RecognitionEvent] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
