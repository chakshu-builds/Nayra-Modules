"""NAYRA — Public Object Location Memory Module Entrypoint.

Provides a standard Python identifier alias for '10_object_memory':
    from object_memory import (
        ObjectMemoryManager,
        ObjectMemoryDB,
        ObjectObservation,
        MemoryQueryResult,
        MemoryQueryParser,
        ResponseGenerator,
        SpatialContextEngine,
        ObservationFilter,
        LiveCameraStream,
        VoiceGuidedObjectFinder,
    )
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "10_object_memory"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("10_object_memory")

ObjectObservation = _pkg.ObjectObservation
RoomAnchor = _pkg.RoomAnchor
MemoryQueryResult = _pkg.MemoryQueryResult
ObjectMemoryDB = _pkg.ObjectMemoryDB
ANCHOR_CLASSES = _pkg.ANCHOR_CLASSES
SpatialContextEngine = _pkg.SpatialContextEngine
ObservationFilter = _pkg.ObservationFilter
LiveCameraStream = _pkg.LiveCameraStream
MemoryQueryParser = _pkg.MemoryQueryParser
ParsedMemoryQuery = _pkg.ParsedMemoryQuery
ResponseGenerator = _pkg.ResponseGenerator
ObjectMemoryManager = _pkg.ObjectMemoryManager
VoiceGuidedObjectFinder = _pkg.VoiceGuidedObjectFinder

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
