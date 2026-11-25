"""Alzheimer's Assistance Service coordinating person recognition, orientation, and cognitive grounding."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from shared.interfaces import IAssistiveModule, IVoiceOutput
from shared.schemas import (
    IntentRequest,
    IntentResponse,
    SpeechPriority,
    UserIntent,
)

try:
    from .patient_profile import PatientProfileManager, FamiliarPerson
    from .orientation_guide import OrientationGuide
except (ImportError, ValueError):
    from patient_profile import PatientProfileManager, FamiliarPerson
    from orientation_guide import OrientationGuide

logger = logging.getLogger("NAYRA-AlzheimerService")


class AlzheimerService(IAssistiveModule):
    """Module 13 service providing cognitive support, orientation, and caregiver information."""

    def __init__(
        self,
        voice_output: Optional[IVoiceOutput] = None,
        profile_manager: Optional[PatientProfileManager] = None,
        orientation_guide: Optional[OrientationGuide] = None,
        face_recognizer: Optional[Any] = None,
        preferred_language: str = "en",
    ) -> None:
        self.voice_output = voice_output
        self.profile = profile_manager or PatientProfileManager()
        self.orientation = orientation_guide or OrientationGuide(home_location=self.profile.home_address)
        self.face_recognizer = face_recognizer  # Can tie into Module 06 face recognizer
        self.preferred_language = preferred_language
        self._lock = threading.RLock()
        self._is_active = False

    @property
    def module_name(self) -> str:
        return "13_alzheimer_assistance"

    def initialize(self) -> bool:
        with self._lock:
            self._is_active = True
            logger.info("Module 13 AlzheimerService initialized.")
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._is_active = False
            logger.info("Module 13 AlzheimerService stopped.")

    def process_query(self, request: IntentRequest) -> IntentResponse:
        """Handle cognitive, orientation, and person identification intents."""
        if not self._is_active:
            return IntentResponse(
                success=False,
                intent=request.intent,
                spoken_response_hi="Cognitive assistance service inactive hai.",
                spoken_response_en="Cognitive assistance service is inactive.",
            )

        with self._lock:
            query = (request.raw_query or "").lower()

            # Orientation queries: "main kahan hoon", "kahan hu", "time kya hai", "aaj kaun sa din hai", "where am i"
            if any(k in query for k in ("kahan", "where am i", "time", "samay", "din", "date", "tarikh", "aaj kya")):
                spoken_hi, spoken_en = self.orientation.get_orientation()
                chosen = spoken_en if (self.preferred_language == "en" or request.detected_language not in ("hi", "hinglish")) else spoken_hi
                if self.voice_output:
                    self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)
                return IntentResponse(
                    success=True,
                    intent=request.intent,
                    spoken_response_hi=spoken_hi,
                    spoken_response_en=spoken_en,
                )

            # Person identification queries: "yeh kaun hai", "who is this", "kon hai", "Rahul kaun hai"
            if request.intent == UserIntent.IDENTIFY_PERSON or any(k in query for k in ("kaun", "kon", "who is", "pehechano", "pahchano")):
                target = request.target_entity or ""
                person = self.profile.get_person_by_name(target) if target else None

                # Check active face recognizer if no specific entity was named in query
                recognized_id = None
                if not person and self.face_recognizer and hasattr(self.face_recognizer, "get_current_identity"):
                    try:
                        recognized_id = self.face_recognizer.get_current_identity()
                        if recognized_id:
                            person = self.profile.get_person_by_name(recognized_id)
                    except Exception as e:
                        logger.error(f"Error checking face recognition: {e}")

                if person:
                    rel = person.relationship
                    spoken_hi = f"Yeh {person.name} hain, aapke {rel}. {person.notes}".strip()
                    spoken_en = f"This is {person.name}, your {rel}. {person.notes}".strip()
                    data = {"familiar_person": person.name, "relationship": rel, "identified": True}
                else:
                    # Never falsely claim an unknown person is a family member
                    spoken_hi = "Mujhe saamne koi jaana-pehchana vyakti nahi dikh raha. Kya main caregiver ko sampark karoon?"
                    spoken_en = "I do not recognize any familiar person in view. Would you like me to contact your caregiver?"
                    data = {"familiar_person": None, "identified": False}

                chosen = spoken_en if (self.preferred_language == "en" or request.detected_language not in ("hi", "hinglish")) else spoken_hi
                if self.voice_output:
                    self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)

                return IntentResponse(
                    success=True,
                    intent=request.intent,
                    spoken_response_hi=spoken_hi,
                    spoken_response_en=spoken_en,
                    data=data,
                )

            # Reassurance fallback for anxiety or repeated questioning
            spoken_hi, spoken_en = self.orientation.get_reassurance(self.profile.primary_caregiver)
            chosen = spoken_en if (self.preferred_language == "en" or request.detected_language not in ("hi", "hinglish")) else spoken_hi
            if self.voice_output:
                self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)

            return IntentResponse(
                success=True,
                intent=request.intent,
                spoken_response_hi=spoken_hi,
                spoken_response_en=spoken_en,
            )
