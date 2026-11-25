"""Obstacle Service coordinating detection, warnings, and user query processing."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, List, Optional, Sequence
import numpy as np

from shared.interfaces import IAssistiveModule, IVoiceOutput
from shared.schemas import (
    IntentRequest,
    IntentResponse,
    ObstacleHazard,
    SpeechPriority,
    UserIntent,
)
try:
    from .obstacle_detector import ObstacleDetector
    from .warning_generator import ObstacleWarningGenerator
except (ImportError, ValueError):
    from obstacle_detector import ObstacleDetector
    from warning_generator import ObstacleWarningGenerator

logger = logging.getLogger("NAYRA-ObstacleService")


class ObstacleService(IAssistiveModule):
    """Module 11 service providing obstacle awareness, spoken warnings, and query handling."""

    def __init__(
        self,
        voice_output: Optional[IVoiceOutput] = None,
        detector: Optional[ObstacleDetector] = None,
        warning_generator: Optional[ObstacleWarningGenerator] = None,
        preferred_language: str = "en",
    ) -> None:
        self.voice_output = voice_output
        self.detector = detector or ObstacleDetector()
        self.generator = warning_generator or ObstacleWarningGenerator()
        self.preferred_language = preferred_language

        self._lock = threading.RLock()
        self._latest_hazards: List[ObstacleHazard] = []
        self._last_evaluated_time: float = 0.0
        self._is_active = False

    @property
    def module_name(self) -> str:
        return "11_obstacle_awareness"

    def initialize(self) -> bool:
        with self._lock:
            self._is_active = True
            logger.info("Module 11 ObstacleService initialized.")
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._is_active = False
            self._latest_hazards.clear()
            self._last_evaluated_time = 0.0
            logger.info("Module 11 ObstacleService shutdown.")

    def evaluate_frame(
        self,
        detections: Sequence[Any],
        tracks: Optional[Sequence[Any]] = None,
        frame_width: int = 640,
        frame_height: int = 480,
        depth_map: Optional[np.ndarray] = None,
        auto_speak: bool = True,
    ) -> List[ObstacleHazard]:
        """Analyze incoming video frame for navigational obstacles."""
        hazards = self.detector.evaluate_obstacles(
            detections=detections,
            tracks=tracks,
            frame_width=frame_width,
            frame_height=frame_height,
            depth_map=depth_map,
        )

        with self._lock:
            self._latest_hazards = hazards
            self._last_evaluated_time = time.time()

        # Check if an announcement should be made
        if auto_speak and self.voice_output is not None:
            announcement = self.generator.generate_scene_warning(hazards)
            if announcement is not None:
                warn_hi, warn_en, is_urgent = announcement
                prio = SpeechPriority.CRITICAL_SAFETY if is_urgent else SpeechPriority.OBSTACLE_WARNING
                chosen_warn = warn_en if self.preferred_language == "en" else warn_hi
                chosen_lang = "en" if self.preferred_language == "en" else "hi"
                # Deliver spoken warning via shared voice output
                self.voice_output.speak(chosen_warn, language=chosen_lang, priority=prio)

        return hazards

    def update_obstacles(
        self,
        detections: Sequence[Any],
        tracks: Optional[Sequence[Any]] = None,
        frame_width: int = 640,
        frame_height: int = 480,
        depth_map: Optional[np.ndarray] = None,
        auto_speak: bool = True,
    ) -> List[ObstacleHazard]:
        """Alias for evaluate_frame to support camera perception loops."""
        return self.evaluate_frame(
            detections=detections,
            tracks=tracks,
            frame_width=frame_width,
            frame_height=frame_height,
            depth_map=depth_map,
            auto_speak=auto_speak,
        )

    def process_query(self, request: IntentRequest) -> IntentResponse:
        """Handle user questions such as 'Mere saamne koi obstacle hai?'"""
        with self._lock:
            hazards = list(self._latest_hazards)
            last_eval = self._last_evaluated_time

        now = time.time()
        # Stale frame handling: if perception previously evaluated but no frames for > 5.0s
        stale = (last_eval > 0.0) and ((now - last_eval) > 5.0)
        if stale:
            spoken_hi = "Camera perception stale hai. Saamne raste ki jaanch nahi ki ja saki."
            spoken_en = "Camera perception is stale. Cannot verify obstacle-free path."
            return IntentResponse(
                success=True,
                intent=request.intent,
                spoken_response_hi=spoken_hi,
                spoken_response_en=spoken_en,
                priority=SpeechPriority.USER_QUERY_RESPONSE,
                data={"perception_available": False, "stale": True, "hazard_count": len(hazards)},
            )

        if not hazards:
            # Distinguish camera unavailable vs active no-obstacle detection
            if last_eval == 0.0:
                spoken_hi = "Camera band ya inactive hai. Live perception ke bina rasta saaf hone ki pushti nahi ki ja sakti."
                spoken_en = "Camera is inactive. Cannot confirm path is clear without active perception."
            else:
                spoken_hi, spoken_en = self.generator.formulate_query_summary([])

            return IntentResponse(
                success=True,
                intent=request.intent,
                spoken_response_hi=spoken_hi,
                spoken_response_en=spoken_en,
                priority=SpeechPriority.USER_QUERY_RESPONSE,
                data={"perception_available": last_eval > 0.0, "stale": False, "hazard_count": 0},
            )

        hi_text, en_text = self.generator.formulate_query_summary(hazards)
        urgent = any(h.is_urgent for h in hazards)
        prio = SpeechPriority.OBSTACLE_WARNING if urgent else SpeechPriority.USER_QUERY_RESPONSE

        return IntentResponse(
            success=True,
            intent=request.intent,
            spoken_response_hi=hi_text,
            spoken_response_en=en_text,
            priority=prio,
            data={
                "hazard_count": len(hazards),
                "urgent": urgent,
                "perception_available": True,
                "hazards": [
                    {
                        "class": h.class_name,
                        "zone": h.zone,
                        "proximity": h.proximity,
                        "relative_depth": h.relative_depth,
                    }
                    for h in hazards
                ],
            },
        )
