"""NAYRA Module 10 — Central Object Location Memory Manager.

Unified orchestrator combining perception (YOLOv10 + ByteTrack), relational spatial
reasoning, SQLite persistence, multilingual query parsing, and empathetic TTS verbal responses.
Includes non-blocking asynchronous database writes and motion-gated inference optimization.
"""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .db.database import ObjectMemoryDB
    from .db.models import MemoryQueryResult, ObjectObservation
    from .nlp.query_parser import MemoryQueryParser, ParsedMemoryQuery
    from .nlp.response_generator import ResponseGenerator
    from .perception.observation_filter import ObservationFilter
    from .perception.spatial_context import SpatialContextEngine
except (ImportError, ValueError):
    from db.database import ObjectMemoryDB
    from db.models import MemoryQueryResult, ObjectObservation
    from nlp.query_parser import MemoryQueryParser, ParsedMemoryQuery
    from nlp.response_generator import ResponseGenerator
    from perception.observation_filter import ObservationFilter
    from perception.spatial_context import SpatialContextEngine

logger = logging.getLogger("NAYRA-MemoryManager")


class ObjectMemoryManager:
    """Coordinates camera frame observation, memory persistence, query parsing, and verbal recall."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        detector: Optional[Any] = None,
        tracker: Optional[Any] = None,
        tts_manager: Optional[Any] = None,
        camera_stream: Optional[Any] = None,
        min_hits: int = 2,
        movement_threshold_ratio: float = 0.15,
        heartbeat_interval_sec: float = 60.0,
        save_snapshots: bool = False,
        snapshot_dir: Optional[str | Path] = None,
        async_db_writes: bool = False,
        enable_motion_gating: bool = False,
        detection_heartbeat_sec: float = 1.0,
        live_freshness_threshold_sec: float = 2.0,
        device: str = "cpu",
    ) -> None:
        self.device = str(device).lower()
        self.live_freshness_threshold_sec = float(live_freshness_threshold_sec)
        self.db = ObjectMemoryDB(db_path=db_path)
        self.spatial_engine = SpatialContextEngine()
        self.filter = ObservationFilter(
            spatial_engine=self.spatial_engine,
            min_hits=min_hits,
            movement_threshold_ratio=movement_threshold_ratio,
            heartbeat_interval_sec=heartbeat_interval_sec,
            save_snapshots=save_snapshots,
            snapshot_dir=snapshot_dir,
        )
        self.parser = MemoryQueryParser()
        self.response_generator = ResponseGenerator()

        self.detector = detector
        self.tracker = tracker
        self.tts = tts_manager
        self.camera_stream = camera_stream
        self._lock = threading.RLock()

        # Cache of currently visible live observations
        self._active_live_observations: Dict[str, ObjectObservation] = {}

        # Optimization knobs
        self.async_db_writes = async_db_writes
        self.enable_motion_gating = enable_motion_gating
        self.detection_heartbeat_sec = detection_heartbeat_sec

        # Telemetry
        self._last_detection_time: float = 0.0
        self._last_inference_latency_ms: float = 0.0
        self._inference_fps: float = 0.0
        self._skipped_frames: int = 0
        self._processed_frames: int = 0
        self._ema_alpha = 0.15

        # Conversational context for follow-up questions
        self._last_queried_target: Optional[str] = None

        # Asynchronous DB writer
        self._db_write_queue: queue.Queue[Optional[ObjectObservation]] = queue.Queue()
        self._db_writer_thread: Optional[threading.Thread] = None
        self._db_writer_running = False
        if self.async_db_writes:
            self._start_db_writer()

        logger.info("ObjectMemoryManager initialized with database: %s (async_writes=%s, motion_gating=%s)",
                    self.db.db_path, self.async_db_writes, self.enable_motion_gating)

    def _start_db_writer(self) -> None:
        """Start asynchronous SQLite background writer thread."""
        self._db_writer_running = True
        self._db_writer_thread = threading.Thread(
            target=self._db_writer_loop,
            name="AsyncDBWriter",
            daemon=True,
        )
        self._db_writer_thread.start()

    def _db_writer_loop(self) -> None:
        """Worker thread executing SQLite record insertion off the video path."""
        while self._db_writer_running:
            try:
                obs = self._db_write_queue.get(timeout=0.1)
                if obs is None:
                    self._db_write_queue.task_done()
                    break
                try:
                    self.db.record_observation(obs)
                except Exception as exc:
                    logger.error("Async DB write failed for %s: %s", obs.canonical_name, exc)
                finally:
                    self._db_write_queue.task_done()
            except queue.Empty:
                continue

    def flush_db_writes(self) -> None:
        """Wait until all queued database writes are committed to disk."""
        if self.async_db_writes and not self._db_write_queue.empty():
            self._db_write_queue.join()

    def shutdown(self) -> None:
        """Flush background queues and terminate helper threads cleanly."""
        if self._db_writer_running:
            self._db_writer_running = False
            self._db_write_queue.put(None)
            if self._db_writer_thread and self._db_writer_thread.is_alive():
                self._db_writer_thread.join(timeout=2.0)
            self._db_writer_thread = None

    def _ensure_vision_models(self) -> None:
        """Lazily initialize YOLOv10 detector and ByteTracker if not provided."""
        if self.detector is None:
            try:
                import importlib
                det_mod = importlib.import_module("03_object_detection.detector")
                ObjectDetector = det_mod.ObjectDetector
                from config.settings import ObjectDetectionSettings

                # Allow full 80 COCO classes for personal belongings with safe device selection
                try:
                    settings = ObjectDetectionSettings(
                        filter_assistive_only=False,
                        confidence_threshold=0.30,
                        device=self.device,
                    )
                    self.detector = ObjectDetector(settings=settings)
                except Exception as dev_exc:
                    if self.device != "cpu":
                        logger.warning(
                            "Failed to initialize ObjectDetector on '%s' (%s). Falling back to 'cpu'.",
                            self.device, dev_exc,
                        )
                        settings = ObjectDetectionSettings(
                            filter_assistive_only=False,
                            confidence_threshold=0.30,
                            device="cpu",
                        )
                        self.detector = ObjectDetector(settings=settings)
                    else:
                        raise dev_exc
                logger.info("Initialized ObjectDetector with full 80 COCO classes on device: %s.", self.detector._active_device)
            except Exception as exc:
                logger.warning("Could not auto-initialize ObjectDetector: %s", exc)

        if self.tracker is None:
            try:
                import importlib
                track_mod = importlib.import_module("04_object_tracking.tracker")
                ByteTracker = track_mod.ByteTracker
                from config.settings import ObjectTrackingSettings

                settings = ObjectTrackingSettings(
                    min_hits=1,
                    track_thresh=0.35,
                    low_thresh=0.15,
                )
                self.tracker = ByteTracker(settings=settings)
                logger.info("Initialized ByteTracker (min_hits=1 for responsive live object finding).")
            except Exception as exc:
                logger.warning("Could not auto-initialize ByteTracker: %s", exc)

    def observe_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        force_detection: bool = False,
    ) -> List[ObjectObservation]:
        """Ingest a camera frame, run detection and tracking, and persist updated observations.

        Args:
            frame: BGR numpy image from phone camera or test stream.
            timestamp: Optional epoch timestamp.
            force_detection: If True, bypasses motion gating.

        Returns:
            List of newly committed ObjectObservation records.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        self._ensure_vision_models()
        now = timestamp if timestamp is not None else time.time()
        self._processed_frames += 1

        if self.detector is None or self.tracker is None:
            logger.warning("Detector or Tracker unavailable; skipping frame observation.")
            return []

        with self._lock:
            # Motion gating check: skip redundant YOLO forward pass if scene is static
            if self.enable_motion_gating and not force_detection:
                has_motion = self.filter.detect_scene_motion(frame)
                time_since_last_det = now - self._last_detection_time
                if not has_motion and time_since_last_det < self.detection_heartbeat_sec:
                    self._skipped_frames += 1
                    return []

            # 1. Detection
            t_det0 = time.perf_counter()
            detections, _ = self.detector.detect(frame, timestamp=now)
            det_lat_ms = (time.perf_counter() - t_det0) * 1000.0
            self._last_inference_latency_ms = det_lat_ms
            self._last_detection_time = now
            self._last_frame_detections = detections

            inst_fps = 1000.0 / det_lat_ms if det_lat_ms > 0 else 0.0
            self._inference_fps = (
                inst_fps if self._inference_fps == 0.0
                else (self._ema_alpha * inst_fps) + ((1.0 - self._ema_alpha) * self._inference_fps)
            )

            # 2. Tracking
            tracks, _ = self.tracker.update(detections, timestamp=now)

            # Refresh active live observation cache for ALL currently active tracks on this frame
            from .perception.spatial_context import ANCHOR_CLASSES
            anchors: List[Tuple[str, Tuple[int, int, int, int]]] = [
                (d.class_name.lower(), d.box_xyxy)
                for d in detections
                if d.class_name.lower() in ANCHOR_CLASSES
            ]

            # 2A. Update live cache from all candidate tracks (confirmed + tentative)
            h, w = (frame.shape[:2] if frame is not None else (480, 640))
            all_candidate_tracks = getattr(self.tracker, "tracked_tracks", []) + getattr(self.tracker, "tentative_tracks", [])
            for t in all_candidate_tracks:
                if getattr(t, "consecutive_missed_frames", 0) <= 2:
                    cname, hi_alias = self.spatial_engine.get_canonical_info(t.class_name.lower())
                    box = t.box_xyxy
                    zone_code, _, _ = self.spatial_engine.compute_zone(box, width=w)
                    vzone_code, _, _ = self.spatial_engine.compute_vertical_zone(box, height=h)
                    anchor_res = self.spatial_engine.find_best_anchor(box, anchors) if cname not in ANCHOR_CLASSES else None
                    anc_name, rel_type = (anchor_res[0], anchor_res[1]) if anchor_res else (None, None)
                    live_obs = ObjectObservation(
                        id=None,
                        object_label=t.class_name,
                        canonical_name=cname,
                        hindi_alias=hi_alias,
                        timestamp=now,
                        confidence=float(t.confidence),
                        bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                        track_id=t.track_id,
                        spatial_relation=rel_type,
                        anchor_object=anc_name,
                        horizontal_zone=zone_code,
                        vertical_zone=vzone_code,
                        frame_width=w,
                        frame_height=h,
                        is_latest=True,
                    )
                    self._active_live_observations[cname] = live_obs

            # 2B. Direct raw detections fallback into live cache for instant zero-delay visibility
            for d in detections:
                if d.confidence >= 0.30:
                    cname, hi_alias = self.spatial_engine.get_canonical_info(d.class_name.lower())
                    if cname not in self._active_live_observations or self._active_live_observations[cname].confidence < d.confidence:
                        box = d.box_xyxy
                        zone_code, _, _ = self.spatial_engine.compute_zone(box, width=w)
                        vzone_code, _, _ = self.spatial_engine.compute_vertical_zone(box, height=h)
                        anchor_res = self.spatial_engine.find_best_anchor(box, anchors) if cname not in ANCHOR_CLASSES else None
                        anc_name, rel_type = (anchor_res[0], anchor_res[1]) if anchor_res else (None, None)
                        live_obs = ObjectObservation(
                            id=None,
                            object_label=d.class_name,
                            canonical_name=cname,
                            hindi_alias=hi_alias,
                            timestamp=now,
                            confidence=float(d.confidence),
                            bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                            track_id=None,
                            spatial_relation=rel_type,
                            anchor_object=anc_name,
                            horizontal_zone=zone_code,
                            vertical_zone=vzone_code,
                            frame_width=w,
                            frame_height=h,
                            is_latest=True,
                        )
                        self._active_live_observations[cname] = live_obs

            # Diagnostic logging
            det_summary = [f"{d.class_name}:{d.confidence:.2f}" for d in detections]
            track_summary = [f"ID{t.track_id}:{t.class_name}(h={getattr(t, 'hits', 1)})" for t in all_candidate_tracks]
            logger.info(
                "[DIAGNOSTIC OBSERVE] Frame #%d | Detections: %d %s | Active Tracks: %d %s | Latency: %.1fms",
                self._processed_frames,
                len(detections),
                det_summary[:5],
                len(all_candidate_tracks),
                track_summary[:5],
                det_lat_ms,
            )

            # 3. Filter and deduplicate (including high-confidence candidate tracks)
            new_obs_list = self.filter.process_frame(
                tracks=all_candidate_tracks,
                detections=detections,
                frame=frame,
                timestamp=now,
            )

            # 4. Save to database (asynchronously or synchronously)
            committed: List[ObjectObservation] = []
            for obs in new_obs_list:
                self._active_live_observations[obs.canonical_name] = obs
                if self.async_db_writes:
                    self._db_write_queue.put(obs)
                    committed.append(obs)
                else:
                    inserted_id = self.db.record_observation(obs)
                    obs.id = inserted_id
                    committed.append(obs)

            return committed

    def record_manual_observation(self, obs: ObjectObservation) -> int:
        """Directly insert an observation into memory (used in testing or pipeline feeds)."""
        return self.db.record_observation(obs)

    def get_currently_visible_observation(
        self,
        canonical_name: str,
        current_frame: Optional[np.ndarray] = None,
    ) -> Optional[ObjectObservation]:
        """Check whether the target object is actively in live camera view right now."""
        with self._lock:
            from .perception.spatial_context import ANCHOR_CLASSES
            now = time.time()

            # Strategy 1: Check fresh active live observation cache updated by stream
            cached = self._active_live_observations.get(canonical_name)
            if cached is not None:
                age_sec = now - cached.timestamp
                if age_sec <= self.live_freshness_threshold_sec:
                    logger.info(
                        "[DIAGNOSTIC VISIBILITY] Target '%s' IS VISIBLE in fresh live cache (Age: %.2fs, Conf: %.2f, Zone: %s)",
                        canonical_name, age_sec, cached.confidence, cached.horizontal_zone,
                    )
                    return cached

            # Strategy 2: Check current confirmed or tentative tracks in tracker (only if fresh)
            is_det_fresh = (self._last_detection_time == 0.0) or ((now - self._last_detection_time) <= self.live_freshness_threshold_sec)
            if self.tracker is not None and is_det_fresh:
                confirmed_tracks = [t for t in getattr(self.tracker, "tracked_tracks", []) if getattr(t, "consecutive_missed_frames", 0) <= 2]
                tentative_tracks = [t for t in getattr(self.tracker, "tentative_tracks", []) if getattr(t, "consecutive_missed_frames", 0) <= 1]
                candidate_tracks = confirmed_tracks + tentative_tracks

                anchors: List[Tuple[str, Tuple[int, int, int, int]]] = [
                    (t.class_name.lower(), t.box_xyxy)
                    for t in candidate_tracks
                    if t.class_name.lower() in ANCHOR_CLASSES
                ]

                h_frame, w_frame = (current_frame.shape[:2] if (current_frame is not None and isinstance(current_frame, np.ndarray)) else (480, 640))
                for t in candidate_tracks:
                    cname, hi_alias = self.spatial_engine.get_canonical_info(t.class_name.lower())
                    if cname == canonical_name:
                        box = t.box_xyxy
                        zone_code, _, _ = self.spatial_engine.compute_zone(box, width=w_frame)
                        vzone_code, _, _ = self.spatial_engine.compute_vertical_zone(box, height=h_frame)
                        anchor_res = self.spatial_engine.find_best_anchor(box, anchors)
                        anc_name, rel_type = (anchor_res[0], anchor_res[1]) if anchor_res else (None, None)

                        live_obs = ObjectObservation(
                            id=None,
                            object_label=t.class_name,
                            canonical_name=cname,
                            hindi_alias=hi_alias,
                            timestamp=now,
                            confidence=float(t.confidence),
                            bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                            track_id=t.track_id,
                            spatial_relation=rel_type,
                            anchor_object=anc_name,
                            horizontal_zone=zone_code,
                            vertical_zone=vzone_code,
                            frame_width=w_frame,
                            frame_height=h_frame,
                            is_latest=True,
                        )
                        self._active_live_observations[canonical_name] = live_obs
                        logger.info(
                            "[DIAGNOSTIC VISIBILITY] Target '%s' IS VISIBLE in tracker candidate tracks (Track ID: %s, Conf: %.2f)",
                            canonical_name, t.track_id, t.confidence,
                        )
                        return live_obs

            # Strategy 3: Check fresh camera frame if available
            fresh_frame = None
            if current_frame is not None and isinstance(current_frame, np.ndarray) and current_frame.size > 0:
                fresh_frame = current_frame
            elif self.camera_stream is not None and getattr(self.camera_stream, "is_running", False):
                has_frame, live_frame, _ = self.camera_stream.read_latest()
                if has_frame and live_frame is not None:
                    fresh_frame = live_frame

            if fresh_frame is not None:
                self.observe_frame(fresh_frame, force_detection=True)
                fresh_cached = self._active_live_observations.get(canonical_name)
                if fresh_cached is not None and (time.time() - fresh_cached.timestamp) <= self.live_freshness_threshold_sec:
                    logger.info(
                        "[DIAGNOSTIC VISIBILITY] Target '%s' IS VISIBLE on fresh camera frame (Conf: %.2f)",
                        canonical_name, fresh_cached.confidence,
                    )
                    return fresh_cached

            # Strategy 4: Raw detections on latest frame ONLY IF FRESH
            if (
                hasattr(self, "_last_frame_detections")
                and self._last_frame_detections
                and (now - self._last_detection_time) <= self.live_freshness_threshold_sec
            ):
                anchors_from_dets = [
                    (d.class_name.lower(), d.box_xyxy)
                    for d in self._last_frame_detections
                    if d.class_name.lower() in ANCHOR_CLASSES
                ]
                h_frame, w_frame = (current_frame.shape[:2] if (current_frame is not None and isinstance(current_frame, np.ndarray)) else (480, 640))
                for d in self._last_frame_detections:
                    cname, hi_alias = self.spatial_engine.get_canonical_info(d.class_name.lower())
                    if cname == canonical_name and d.confidence >= 0.30:
                        box = d.box_xyxy
                        zone_code, _, _ = self.spatial_engine.compute_zone(box, width=w_frame)
                        vzone_code, _, _ = self.spatial_engine.compute_vertical_zone(box, height=h_frame)
                        anchor_res = self.spatial_engine.find_best_anchor(box, anchors_from_dets)
                        anc_name, rel_type = (anchor_res[0], anchor_res[1]) if anchor_res else (None, None)

                        live_obs = ObjectObservation(
                            id=None,
                            object_label=d.class_name,
                            canonical_name=cname,
                            hindi_alias=hi_alias,
                            timestamp=self._last_detection_time,
                            confidence=float(d.confidence),
                            bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                            track_id=None,
                            spatial_relation=rel_type,
                            anchor_object=anc_name,
                            horizontal_zone=zone_code,
                            vertical_zone=vzone_code,
                            frame_width=w_frame,
                            frame_height=h_frame,
                            is_latest=True,
                        )
                        self._active_live_observations[canonical_name] = live_obs
                        logger.info(
                            "[DIAGNOSTIC VISIBILITY] Target '%s' IS VISIBLE in raw frame detections (Conf: %.2f)",
                            canonical_name, d.confidence,
                        )
                        return live_obs

            # Not visible: Provide exact diagnostic failure reason
            active_tracks_summary = [
                f"ID{t.track_id}:{t.class_name}(missed={getattr(t, 'consecutive_missed_frames', 0)})"
                for t in getattr(self.tracker, "tracked_tracks", [])
            ] if self.tracker else []
            cache_age = (now - cached.timestamp) if cached else None
            persisted_obs = self.db.get_latest_observation(canonical_name)
            persisted_age = (now - persisted_obs.timestamp) if persisted_obs else None

            logger.info(
                "[DIAGNOSTIC VISIBILITY] Target '%s' is NOT visible in current camera view. "
                "Reason: Not in active tracks (tracks: %s), memory cache age: %s, persisted DB age: %s",
                canonical_name,
                active_tracks_summary[:4],
                f"{cache_age:.1f}s" if cache_age is not None else "None",
                f"{persisted_age:.1f}s" if persisted_age is not None else "None",
            )
            return None

    def _handle_scene_enumeration_query(
        self,
        parsed_q: ParsedMemoryQuery,
        current_frame: Optional[np.ndarray] = None,
        preferred_language: Optional[str] = None,
        speak: bool = False,
    ) -> MemoryQueryResult:
        """Enumerate all currently visible objects in camera view for general scene queries."""
        now = time.time()
        fresh_frame = None
        if current_frame is not None and isinstance(current_frame, np.ndarray) and current_frame.size > 0:
            fresh_frame = current_frame
        elif self.camera_stream is not None and getattr(self.camera_stream, "is_running", False):
            has_frame, live_frame, _ = self.camera_stream.read_latest()
            if has_frame and live_frame is not None:
                fresh_frame = live_frame

        if fresh_frame is not None:
            self.observe_frame(fresh_frame, force_detection=True)

        visible_set: Set[str] = set()

        # From live observation cache
        for cname, obs in self._active_live_observations.items():
            if (now - obs.timestamp) <= self.live_freshness_threshold_sec:
                visible_set.add(cname)

        # From active tracks (only if fresh)
        is_det_fresh = (self._last_detection_time == 0.0) or ((now - self._last_detection_time) <= self.live_freshness_threshold_sec)
        if self.tracker is not None and is_det_fresh:
            for t in getattr(self.tracker, "tracked_tracks", []):
                if getattr(t, "consecutive_missed_frames", 0) <= 2:
                    cname, _ = self.spatial_engine.get_canonical_info(t.class_name.lower())
                    visible_set.add(cname)
            for t in getattr(self.tracker, "tentative_tracks", []):
                if getattr(t, "consecutive_missed_frames", 0) <= 1:
                    cname, _ = self.spatial_engine.get_canonical_info(t.class_name.lower())
                    visible_set.add(cname)

        # From raw detections if fresh
        if (
            hasattr(self, "_last_frame_detections")
            and self._last_frame_detections
            and (now - self._last_detection_time) <= self.live_freshness_threshold_sec
        ):
            for d in self._last_frame_detections:
                if d.confidence >= 0.30:
                    cname, _ = self.spatial_engine.get_canonical_info(d.class_name.lower())
                    visible_set.add(cname)

        result = self.response_generator.generate_scene_list_response(parsed_q, sorted(list(visible_set)))

        # Log query to database
        chosen_response = result.spoken_response_en if preferred_language == "en" else result.spoken_response_hi
        self.db.log_query(
            user_query=parsed_q.raw_query,
            matched_object="all",
            obs_id=None,
            response_text=chosen_response,
        )

        if speak and self.tts is not None:
            speak_lang = "en" if preferred_language == "en" else ("hi" if preferred_language == "hi" else parsed_q.detected_language)
            self._speak_response(result, speak_lang)

        return result

    def query(
        self,
        user_query: str,
        current_frame: Optional[np.ndarray] = None,
        check_live_view: Optional[bool] = None,
        speak: bool = False,
        preferred_language: Optional[str] = None,
    ) -> MemoryQueryResult:
        """Parse natural language query, retrieve location from live view or memory, and answer.

        Args:
            user_query: Question in Hindi, English, or Hinglish (e.g. 'Mera phone kahan hai?')
            current_frame: Optional live camera frame to inspect current visibility.
            check_live_view: If True, evaluates live camera view. If None, auto-detected.
            speak: If True and TTS is configured, speaks response aloud through laptop speakers.
            preferred_language: 'hi', 'en', or None (auto-detected from query).

        Returns:
            MemoryQueryResult with bilingual verbal responses and observation metadata.
        """
        # Ensure any pending asynchronous observation writes are committed
        self.flush_db_writes()

        parsed_q = self.parser.parse(user_query)

        # Handle follow-up query with pronoun when target_object was not explicit in this utterance
        if getattr(parsed_q, "is_follow_up", False) and not parsed_q.target_object:
            if self._last_queried_target:
                parsed_q.target_object = self._last_queried_target
                parsed_q.canonical_object = self._last_queried_target
                parsed_q.display_name_en = self._last_queried_target
                parsed_q.is_supported = True
                logger.info(
                    "[FOLLOW-UP RESOLUTION] Resolved follow-up pronoun to last queried target: '%s'",
                    self._last_queried_target,
                )
            else:
                return self.response_generator.generate_unknown_followup_response(parsed_q)

        # 1. Handle scene enumeration / list query ('What can you see?')
        if parsed_q.query_type == "LIST":
            return self._handle_scene_enumeration_query(
                parsed_q, current_frame=current_frame, preferred_language=preferred_language, speak=speak
            )

        target_canonical = parsed_q.target_object
        obs = None
        is_currently_visible: Optional[bool] = None

        now = time.time()
        active_tracks_str = [
            f"ID{t.track_id}:{t.class_name}"
            for t in getattr(self.tracker, "tracked_tracks", [])
        ] if self.tracker else []
        mem_obs = self._active_live_observations.get(target_canonical) if target_canonical else None
        mem_age_str = f"{(now - mem_obs.timestamp):.1f}s" if mem_obs else "None"
        db_obs = self.db.get_latest_observation(target_canonical) if target_canonical else None
        db_age_str = f"{(now - db_obs.timestamp):.1f}s" if db_obs else "None"

        logger.info(
            "[DIAGNOSTIC QUERY] Recognized: '%s' | Parsed Target: '%s' (Supported=%s, Lang=%s, Intent=%s) | "
            "Active Tracks: %s | Memory Cache: %s (Age: %s) | Persisted DB: %s (Age: %s)",
            user_query,
            target_canonical,
            parsed_q.is_supported,
            parsed_q.detected_language,
            parsed_q.query_type,
            active_tracks_str[:5],
            mem_obs.object_label if mem_obs else "None",
            mem_age_str,
            db_obs.object_label if db_obs else "None",
            db_age_str,
        )

        if check_live_view is None:
            # Auto-detect if live camera / tracker is actively engaged
            has_tracker = self.tracker is not None
            has_stream = bool(self.camera_stream and getattr(self.camera_stream, "is_running", False))
            has_frame = current_frame is not None
            check_live_view = has_tracker or has_stream or has_frame

        if target_canonical and parsed_q.is_supported:
            self._last_queried_target = target_canonical
            if check_live_view:
                # 1. Check if object is currently visible in live view
                live_obs = self.get_currently_visible_observation(target_canonical, current_frame=current_frame)
                if live_obs is not None:
                    obs = live_obs
                    is_currently_visible = True
                else:
                    # 2. If not visible, retrieve last reliable observation from SQLite DB strictly for target_canonical
                    obs = self.db.get_latest_observation(target_canonical)
                    is_currently_visible = False
            else:
                # Pure database recall
                obs = self.db.get_latest_observation(target_canonical)
                is_currently_visible = None

        # STRICT GROUNDING: Ensure retrieved observation strictly matches the requested canonical target
        if obs is not None and target_canonical is not None:
            if obs.canonical_name != target_canonical:
                logger.error(
                    "[GROUNDING VIOLATION BLOCKED] Observation canonical '%s' does not match target '%s'",
                    obs.canonical_name, target_canonical,
                )
                obs = None

        result = self.response_generator.generate_response(
            parsed_q, obs, is_currently_visible=is_currently_visible
        )

        # Log query to database
        if preferred_language == "en":
            chosen_response = result.spoken_response_en
            speak_lang = "en"
        elif preferred_language == "hi":
            chosen_response = result.spoken_response_hi
            speak_lang = "hi"
        elif parsed_q.detected_language in ("hi", "hinglish"):
            chosen_response = result.spoken_response_hi
            speak_lang = "hi"
        else:
            chosen_response = result.spoken_response_en
            speak_lang = "en"

        self.db.log_query(
            user_query=user_query,
            matched_object=target_canonical,
            obs_id=obs.id if obs else None,
            response_text=chosen_response,
        )

        # Verbal output via TTS if requested
        if speak and self.tts is not None:
            self._speak_response(result, speak_lang)

        return result

    def _speak_response(self, result: MemoryQueryResult, detected_lang: str) -> None:
        """Dispatch spoken audio to TTS manager."""
        try:
            if detected_lang in ("hi", "hinglish"):
                spoken_text = result.spoken_response_hi
                lang_code = "hi"
            else:
                spoken_text = result.spoken_response_en
                lang_code = "en"

            logger.info("Speaking memory response [%s]: %s", lang_code, spoken_text)
            import inspect
            sig = inspect.signature(self.tts.speak)
            kwargs = {}
            if "language_hint" in sig.parameters:
                kwargs["language_hint"] = lang_code
            elif "lang" in sig.parameters:
                kwargs["lang"] = lang_code
            self.tts.speak(spoken_text, **kwargs)
        except Exception as exc:
            logger.warning("TTS speech dispatch failed: %s", exc)

    def list_known_objects(self) -> List[Dict[str, Any]]:
        """List all distinct objects currently remembered."""
        self.flush_db_writes()
        return self.db.list_all_tracked_objects()

    def get_latest_observation(self, canonical_name: str) -> Optional[ObjectObservation]:
        """Directly retrieve the latest observation for a canonical object name."""
        self.flush_db_writes()
        return self.db.get_latest_observation(canonical_name)

    def prune_stale_observations(self, max_age_days: int = 14) -> int:
        """Delete historical observations older than max_age_days while preserving latest."""
        self.flush_db_writes()
        return self.db.prune_old_observations(max_age_days=max_age_days)

    def clear(self) -> None:
        """Clear database records and reset perception tracking filters."""
        self.flush_db_writes()
        self.db.clear_all()
        self.filter.reset()
        if self.tracker is not None and hasattr(self.tracker, "reset"):
            self.tracker.reset()
        logger.info("ObjectMemoryManager cleared.")

    def get_telemetry(self) -> Dict[str, Any]:
        """Return real-time performance and pipeline metrics."""
        return {
            "inference_latency_ms": round(self._last_inference_latency_ms, 1),
            "inference_fps": round(self._inference_fps, 1),
            "skipped_static_frames": self._skipped_frames,
            "processed_frames": self._processed_frames,
            "pending_db_writes": self._db_write_queue.qsize() if self.async_db_writes else 0,
        }
