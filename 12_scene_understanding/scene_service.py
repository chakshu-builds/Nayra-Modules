"""Scene Service coordinating scene analysis and user queries."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Sequence

from shared.interfaces import IAssistiveModule, IVoiceOutput
from shared.schemas import (
    IntentRequest,
    IntentResponse,
    SpeechPriority,
    UserIntent,
)

try:
    from .scene_analyzer import SceneAnalyzer, SceneDescription
except (ImportError, ValueError):
    from scene_analyzer import SceneAnalyzer, SceneDescription

logger = logging.getLogger("NAYRA-SceneService")


class SceneService(IAssistiveModule):
    """Module 12 service providing contextual scene understanding and spoken scene narration."""

    def __init__(
        self,
        voice_output: Optional[IVoiceOutput] = None,
        analyzer: Optional[SceneAnalyzer] = None,
        preferred_language: str = "en",
    ) -> None:
        self.voice_output = voice_output
        self.analyzer = analyzer or SceneAnalyzer()
        self.preferred_language = preferred_language
        self._lock = threading.RLock()
        self._is_active = False
        self._latest_detections: Sequence[Any] = []
        self._latest_frame_size = (640, 480)
        self._latest_ocr_text: Optional[str] = None
        self._last_scene_description: Optional[SceneDescription] = None

    @property
    def module_name(self) -> str:
        return "12_scene_understanding"

    def initialize(self) -> bool:
        with self._lock:
            self._is_active = True
            logger.info("Module 12 SceneService initialized.")
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._is_active = False
            logger.info("Module 12 SceneService stopped.")

    def update_scene_context(
        self,
        detections: Sequence[Any],
        frame_width: int = 640,
        frame_height: int = 480,
        ocr_text: Optional[str] = None,
    ) -> SceneDescription:
        """Update live perception state and synthesize current scene description."""
        with self._lock:
            self._latest_detections = detections
            self._latest_frame_size = (frame_width, frame_height)
            if ocr_text is not None:
                self._latest_ocr_text = ocr_text

            desc = self.analyzer.analyze(
                detections=self._latest_detections,
                frame_width=self._latest_frame_size[0],
                frame_height=self._latest_frame_size[1],
                ocr_text=self._latest_ocr_text,
            )
            self._last_scene_description = desc
            return desc

    def process_query(self, request: IntentRequest) -> IntentResponse:
        """Process an incoming user intent query."""
        return self.handle_intent(request)

    def handle_intent(self, request: IntentRequest) -> IntentResponse:
        """Handle scene understanding intents."""
        if not self._is_active:
            return IntentResponse(
                success=False,
                intent=request.intent,
                spoken_response_hi="Scene understanding module abhi inactive hai.",
                spoken_response_en="Scene understanding module is currently inactive.",
            )

        with self._lock:
            # Check target zone in entities or text
            tz: Optional[str] = None
            query_lower = (request.raw_query or "").lower()
            if "saamne" in query_lower or "front" in query_lower or "ahead" in query_lower:
                tz = "CENTER"
            elif "left" in query_lower or "bayein" in query_lower or "bayi" in query_lower:
                tz = "LEFT"
            elif "right" in query_lower or "dayein" in query_lower or "dayi" in query_lower:
                tz = "RIGHT"

            desc = self.analyzer.analyze(
                detections=self._latest_detections,
                frame_width=self._latest_frame_size[0],
                frame_height=self._latest_frame_size[1],
                ocr_text=self._latest_ocr_text,
                target_zone=tz,
            )

            # Spoken response in preferred language (default English)
            chosen_spoken = desc.spoken_en if (self.preferred_language == "en" or request.detected_language not in ("hi", "hinglish")) else desc.spoken_hi

            if self.voice_output:
                self.voice_output.speak(chosen_spoken, priority=SpeechPriority.USER_QUERY_RESPONSE)

            return IntentResponse(
                success=True,
                intent=request.intent,
                spoken_response_hi=desc.spoken_hi,
                spoken_response_en=desc.spoken_en,
                data={
                    "objects_by_zone": desc.objects_by_zone,
                    "people_count": desc.people_count,
                    "spatial_relations": desc.spatial_relations,
                    "ocr_text": desc.ocr_text,
                },
            )
