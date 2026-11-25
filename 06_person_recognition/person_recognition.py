"""End-to-end consent-based familiar person recognition coordinator."""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np

from config.settings import PersonRecognitionSettings, get_default_person_recognition_settings
from context_database import PersonContextDatabase
from face_aligner import FaceAligner
from face_database import FaceDatabase
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_tracker import FaceTracker
from recognition_types import (
    FaceDetection,
    PersonRecognitionResult,
    RecognitionEvent,
    RecognitionStatus,
    RecognizedPerson,
)

logger = logging.getLogger(__name__)


class PersonRecognitionPipeline:
    """Coordinates detection, alignment, embedding extraction, database matching, and face tracking."""

    def __init__(
        self,
        settings: PersonRecognitionSettings | None = None,
        database: FaceDatabase | None = None,
        context_database: PersonContextDatabase | None = None,
    ) -> None:
        self.settings = settings or get_default_person_recognition_settings()
        self.database = database or FaceDatabase(self.settings)
        self.context_database = context_database or PersonContextDatabase(self.settings)

        self.detector = FaceDetector(self.settings)
        self.aligner = FaceAligner(target_size=(112, 112))
        self.recognizer = FaceRecognizer(self.settings)
        self.tracker = FaceTracker(
            announcement_cooldown_sec=self.settings.announcement_cooldown_sec
        )

        self._frame_count: int = 0
        self._last_result: PersonRecognitionResult | None = None

        logger.info(
            "PersonRecognitionPipeline initialized (threshold: %.2f, interval: %d, enrolled: %d)",
            self.settings.recognition_threshold,
            self.settings.recognition_interval,
            len(self.database.list_profiles()),
        )

    def match_embedding(
        self,
        probe_emb: np.ndarray,
    ) -> tuple[bool, str | None, str, float, RecognitionStatus]:
        """Compare probe embedding against all enrolled profiles using cosine similarity."""
        profiles = self.database.list_profiles()
        if not profiles:
            return False, None, "Unknown", 0.0, RecognitionStatus.UNKNOWN

        candidate_scores: list[tuple[str, str, float]] = []

        for p in profiles:
            best_sample_sim = -1.0
            for ref_emb_list in p.embeddings:
                ref_emb = np.asarray(ref_emb_list, dtype=np.float32)
                sim = self.recognizer.compute_similarity(probe_emb, ref_emb)
                if sim > best_sample_sim:
                    best_sample_sim = sim
            candidate_scores.append((p.profile_id, p.display_name, best_sample_sim))

        candidate_scores.sort(key=lambda x: x[2], reverse=True)
        best_id, best_name, best_score = candidate_scores[0]
        runner_up_score = candidate_scores[1][2] if len(candidate_scores) > 1 else -1.0

        if best_score < self.settings.recognition_threshold:
            return False, None, "Unknown", round(best_score, 4), RecognitionStatus.UNKNOWN

        # Ambiguity check
        if len(candidate_scores) > 1 and (best_score - runner_up_score) < self.settings.ambiguity_margin:
            logger.debug(
                "Ambiguous match between '%s' (%.3f) and '%s' (%.3f)",
                best_name,
                best_score,
                candidate_scores[1][1],
                runner_up_score,
            )
            return False, None, "Unknown", round(best_score, 4), RecognitionStatus.LOW_CONFIDENCE

        return True, best_id, best_name, round(best_score, 4), RecognitionStatus.RECOGNIZED

    def process(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> PersonRecognitionResult:
        """Execute full person recognition pipeline on video frame."""
        if timestamp is None:
            timestamp = time.time()

        t0 = time.perf_counter()
        self._frame_count += 1

        # Execution stride/interval check
        if (
            self.settings.recognition_interval > 1
            and (self._frame_count % self.settings.recognition_interval != 1)
            and self._last_result is not None
        ):
            return self._last_result

        t_det_start = time.perf_counter()
        detections = self.detector.detect(frame)
        t_det_ms = (time.perf_counter() - t_det_start) * 1000.0

        detections_with_rec: list[tuple[FaceDetection, bool, str | None, str, float, RecognitionStatus]] = []
        t_rec_total = 0.0

        for det in detections:
            aligned = self.aligner.align(frame, det)
            if aligned is None:
                detections_with_rec.append((det, False, None, "Unknown", 0.0, RecognitionStatus.FACE_NOT_VISIBLE))
                continue

            valid_qual, _ = self.aligner.check_quality(aligned)
            if not valid_qual:
                detections_with_rec.append((det, False, None, "Unknown", 0.0, RecognitionStatus.LOW_QUALITY))
                continue

            t_rec_start = time.perf_counter()
            emb = self.recognizer.extract_embedding(aligned)
            t_rec_total += (time.perf_counter() - t_rec_start) * 1000.0

            if emb is None:
                detections_with_rec.append((det, False, None, "Unknown", 0.0, RecognitionStatus.PROCESSING_ERROR))
                continue

            is_known, identity_id, name, score, status = self.match_embedding(emb)
            detections_with_rec.append((det, is_known, identity_id, name, score, status))

        persons, events = self.tracker.associate_and_stabilize(detections_with_rec, timestamp)

        # Retrieve personal context for recognized profiles
        for person in persons:
            if person.is_known and person.identity_id:
                person.context = self.context_database.get_context(person.identity_id, face_db=self.database)

        # Refine announcement message if relationship is known
        for event in events:
            if event.person.context and event.person.context.relationship:
                rel = event.person.context.relationship
                event.message = f"{event.person.display_name} ({rel}) is in front of you."

        total_ms = (time.perf_counter() - t0) * 1000.0
        metrics = {
            "detection_ms": round(t_det_ms, 2),
            "recognition_ms": round(t_rec_total, 2),
            "total_ms": round(total_ms, 2),
            "active_faces": len(persons),
            "known_faces": sum(1 for p in persons if p.is_known),
        }

        result = PersonRecognitionResult(
            persons=persons,
            events=events,
            metrics=metrics,
            timestamp=timestamp,
        )
        self._last_result = result
        return result
