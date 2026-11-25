"""Module 06: Consent-Based Familiar Person Recognition."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from context_database import PersonContextDatabase
from enroll import enroll_from_images
from face_aligner import FaceAligner
from face_database import FaceDatabase
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_tracker import FaceTracker
from person_recognition import PersonRecognitionPipeline
from recognition_types import (
    FaceDetection,
    FaceProfile,
    FactItem,
    PersonContext,
    PersonRecognitionResult,
    PreferenceItem,
    RecognitionEvent,
    RecognitionStatus,
    RecognizedPerson,
)
from visualizer import PersonRecognitionVisualizer

__all__ = [
    "FaceDetector",
    "FaceAligner",
    "FaceRecognizer",
    "FaceDatabase",
    "PersonContextDatabase",
    "FaceTracker",
    "PersonRecognitionPipeline",
    "PersonRecognitionVisualizer",
    "FaceDetection",
    "FaceProfile",
    "PersonContext",
    "FactItem",
    "PreferenceItem",
    "RecognizedPerson",
    "RecognitionStatus",
    "RecognitionEvent",
    "PersonRecognitionResult",
    "enroll_from_images",
]
