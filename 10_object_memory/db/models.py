"""NAYRA Module 10 — Typed Models for Object Location Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ObjectObservation:
    """Represents a recorded object observation in memory."""

    id: Optional[int]
    object_label: str
    canonical_name: str
    timestamp: float
    confidence: float
    bbox: Tuple[int, int, int, int]
    hindi_alias: Optional[str] = None
    iso_time: Optional[str] = None
    track_id: Optional[int] = None
    spatial_relation: Optional[str] = None
    anchor_object: Optional[str] = None
    horizontal_zone: Optional[str] = None
    vertical_zone: Optional[str] = None
    frame_width: Optional[int] = 640
    frame_height: Optional[int] = 480
    room_label: Optional[str] = None
    snapshot_path: Optional[str] = None
    is_latest: bool = True

    def __post_init__(self):
        if not self.iso_time:
            self.iso_time = datetime.fromtimestamp(self.timestamp).isoformat()

    @property
    def bbox_json(self) -> str:
        return json.dumps(list(self.bbox))

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.timestamp)

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60.0

    @property
    def elapsed_hours(self) -> float:
        return self.elapsed_seconds / 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object_label": self.object_label,
            "canonical_name": self.canonical_name,
            "hindi_alias": self.hindi_alias,
            "timestamp": self.timestamp,
            "iso_time": self.iso_time,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "spatial_relation": self.spatial_relation,
            "anchor_object": self.anchor_object,
            "horizontal_zone": self.horizontal_zone,
            "vertical_zone": self.vertical_zone,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "room_label": self.room_label,
            "bbox": list(self.bbox),
            "snapshot_path": self.snapshot_path,
            "is_latest": self.is_latest,
            "elapsed_minutes": round(self.elapsed_minutes, 1),
        }


@dataclass
class RoomAnchor:
    """Represents a signature furniture association with a named room."""

    id: Optional[int]
    room_name: str
    signature_objects: List[str]
    confirmed_by_user: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryQueryResult:
    """Result of an object location query."""

    found: bool
    canonical_name: str
    observation: Optional[ObjectObservation] = None
    spoken_response_hi: str = ""
    spoken_response_en: str = ""
    confidence_level: str = "HIGH"  # 'HIGH', 'UNCERTAIN', 'OLD', 'NONE'
    time_category: str = "RECENT"   # 'NOW', 'RECENT', 'EARLIER', 'OLD'
    is_currently_visible: bool = False
