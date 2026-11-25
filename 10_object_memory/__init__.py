"""NAYRA Module 10 — Local-First Object Location Memory Subsystem.

Provides continuous object tracking, spatial relational context reasoning,
SQLite observation persistence, multilingual query parsing, empathetic
spoken responses, and voice-guided object finding for Alzheimer's and dementia patient assistance.
"""

from __future__ import annotations

from pathlib import Path
import sys

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from .db.models import MemoryQueryResult, ObjectObservation, RoomAnchor
    from .db.database import ObjectMemoryDB
    from .perception.spatial_context import ANCHOR_CLASSES, SpatialContextEngine
    from .perception.observation_filter import ObservationFilter
    from .perception.live_camera_stream import LiveCameraStream
    from .nlp.query_parser import MemoryQueryParser, ParsedMemoryQuery
    from .nlp.response_generator import ResponseGenerator
    from .memory_manager import ObjectMemoryManager
    from .voice_guidance import VoiceGuidedObjectFinder
except (ImportError, ValueError):
    from db.models import MemoryQueryResult, ObjectObservation, RoomAnchor
    from db.database import ObjectMemoryDB
    from perception.spatial_context import ANCHOR_CLASSES, SpatialContextEngine
    from perception.observation_filter import ObservationFilter
    from perception.live_camera_stream import LiveCameraStream
    from nlp.query_parser import MemoryQueryParser, ParsedMemoryQuery
    from nlp.response_generator import ResponseGenerator
    from memory_manager import ObjectMemoryManager
    from voice_guidance import VoiceGuidedObjectFinder

if "object_memory" not in sys.modules:
    sys.modules["object_memory"] = sys.modules[__name__]

__all__ = [
    "ObjectObservation",
    "RoomAnchor",
    "MemoryQueryResult",
    "ObjectMemoryDB",
    "ANCHOR_CLASSES",
    "SpatialContextEngine",
    "ObservationFilter",
    "LiveCameraStream",
    "MemoryQueryParser",
    "ParsedMemoryQuery",
    "ResponseGenerator",
    "ObjectMemoryManager",
    "VoiceGuidedObjectFinder",
]
