"""NAYRA Module 10 — SQLite Object Location Memory Database Manager."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from .models import MemoryQueryResult, ObjectObservation, RoomAnchor
except (ImportError, ValueError):
    from db.models import MemoryQueryResult, ObjectObservation, RoomAnchor

logger = logging.getLogger("NAYRA-ObjectMemoryDB")


class ObjectMemoryDB:
    """Thread-safe SQLite storage engine for object location observations."""

    DEFAULT_DB_PATH = Path("data/memory/object_memory.db")

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        schema_path = Path(__file__).resolve().parent / "schema.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Missing database schema at {schema_path}")

        schema_sql = schema_path.read_text(encoding="utf-8")
        with self._lock:
            with self._get_connection() as conn:
                conn.executescript(schema_sql)
                # Auto-migrate optional columns for backward compatibility
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(object_observations)")
                existing_cols = {row["name"] for row in cur.fetchall()}
                if "vertical_zone" not in existing_cols:
                    conn.execute("ALTER TABLE object_observations ADD COLUMN vertical_zone TEXT DEFAULT NULL")
                if "frame_width" not in existing_cols:
                    conn.execute("ALTER TABLE object_observations ADD COLUMN frame_width INTEGER DEFAULT 640")
                if "frame_height" not in existing_cols:
                    conn.execute("ALTER TABLE object_observations ADD COLUMN frame_height INTEGER DEFAULT 480")
                conn.commit()
        logger.info("ObjectMemoryDB initialized at %s", self.db_path)

    def record_observation(self, obs: ObjectObservation) -> int:
        """Store a new object observation and maintain the 'is_latest' pointer."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                # Mark previous latest records for this canonical object as not latest
                cur.execute(
                    "UPDATE object_observations SET is_latest = 0 WHERE canonical_name = ?",
                    (obs.canonical_name,),
                )
                cur.execute(
                    """
                    INSERT INTO object_observations (
                        object_label, canonical_name, hindi_alias, timestamp, iso_time,
                        confidence, track_id, spatial_relation, anchor_object, horizontal_zone,
                        vertical_zone, frame_width, frame_height, room_label, bbox_json,
                        snapshot_path, is_latest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        obs.object_label,
                        obs.canonical_name,
                        obs.hindi_alias,
                        obs.timestamp,
                        obs.iso_time or datetime.fromtimestamp(obs.timestamp).isoformat(),
                        obs.confidence,
                        obs.track_id,
                        obs.spatial_relation,
                        obs.anchor_object,
                        obs.horizontal_zone,
                        obs.vertical_zone,
                        obs.frame_width or 640,
                        obs.frame_height or 480,
                        obs.room_label,
                        obs.bbox_json,
                        obs.snapshot_path,
                    ),
                )
                inserted_id = cur.lastrowid
                conn.commit()
                obs.id = inserted_id
                logger.debug("Recorded observation #%d for '%s'", inserted_id, obs.canonical_name)
                return inserted_id

    def get_latest_observation(self, canonical_name: str) -> Optional[ObjectObservation]:
        """Retrieve the single most recent valid observation for a given object."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT * FROM object_observations
                    WHERE canonical_name = ? AND is_latest = 1
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (canonical_name,),
                )
                row = cur.fetchone()
                if row:
                    return self._row_to_observation(row)
                return None

    def get_history(self, canonical_name: str, limit: int = 10) -> List[ObjectObservation]:
        """Retrieve historical observations for an object ordered newest to oldest."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT * FROM object_observations
                    WHERE canonical_name = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (canonical_name, limit),
                )
                return [self._row_to_observation(r) for r in cur.fetchall()]

    def list_all_tracked_objects(self) -> List[Dict[str, Any]]:
        """Return a list of all distinct objects currently remembered."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT canonical_name, object_label, hindi_alias, timestamp, anchor_object, spatial_relation
                    FROM object_observations
                    WHERE is_latest = 1
                    ORDER BY timestamp DESC
                    """
                )
                return [dict(r) for r in cur.fetchall()]

    def log_query(
        self,
        user_query: str,
        matched_object: Optional[str],
        obs_id: Optional[int],
        response_text: str,
    ) -> None:
        """Log a user query and the system's generated verbal answer."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO memory_queries (
                        user_query, matched_object, retrieved_observation_id,
                        response_text, query_timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_query, matched_object, obs_id, response_text, time.time()),
                )
                conn.commit()

    def prune_old_observations(self, max_age_days: int = 14) -> int:
        """Delete historical records older than max_age_days while preserving the latest observation."""
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM object_observations WHERE timestamp < ? AND is_latest = 0",
                    (cutoff,),
                )
                deleted = cur.rowcount
                conn.commit()
                logger.info("Pruned %d stale observation records older than %d days.", deleted, max_age_days)
                return deleted

    def clear_all(self) -> None:
        """Clear all memory database records (used for test teardown or privacy reset)."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM memory_queries")
                conn.execute("DELETE FROM object_observations")
                conn.execute("DELETE FROM room_anchors")
                conn.commit()

    @staticmethod
    def _row_to_observation(row: sqlite3.Row) -> ObjectObservation:
        bbox_list = json.loads(row["bbox_json"]) if row["bbox_json"] else [0, 0, 0, 0]
        keys = row.keys()
        return ObjectObservation(
            id=row["id"],
            object_label=row["object_label"],
            canonical_name=row["canonical_name"],
            hindi_alias=row["hindi_alias"],
            timestamp=row["timestamp"],
            iso_time=row["iso_time"],
            confidence=row["confidence"],
            track_id=row["track_id"],
            spatial_relation=row["spatial_relation"],
            anchor_object=row["anchor_object"],
            horizontal_zone=row["horizontal_zone"],
            vertical_zone=row["vertical_zone"] if "vertical_zone" in keys else None,
            frame_width=row["frame_width"] if "frame_width" in keys else 640,
            frame_height=row["frame_height"] if "frame_height" in keys else 480,
            room_label=row["room_label"],
            bbox=(int(bbox_list[0]), int(bbox_list[1]), int(bbox_list[2]), int(bbox_list[3])),
            snapshot_path=row["snapshot_path"],
            is_latest=bool(row["is_latest"]),
        )
