"""Safety Service coordinating emergency SOS intents and fall alerts."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from shared.interfaces import IAssistiveModule, IVoiceOutput
from shared.schemas import (
    IntentRequest,
    IntentResponse,
    SpeechPriority,
    UserIntent,
)

try:
    from .sos_manager import SOSManager, SOSState
except (ImportError, ValueError):
    from sos_manager import SOSManager, SOSState

logger = logging.getLogger("NAYRA-SafetyService")


class SafetyService(IAssistiveModule):
    """Module 15 service handling SOS voice triggers, confirmations, cancellations, and emergency alerts."""

    def __init__(
        self,
        voice_output: Optional[IVoiceOutput] = None,
        sos_manager: Optional[SOSManager] = None,
        preferred_language: str = "en",
    ) -> None:
        self.voice_output = voice_output
        self.preferred_language = preferred_language
        self.sos = sos_manager or SOSManager(voice_output=self.voice_output, preferred_language=preferred_language)
        self._lock = threading.RLock()
        self._is_active = False

    @property
    def module_name(self) -> str:
        return "15_safety_sos"

    def initialize(self) -> bool:
        with self._lock:
            self._is_active = True
            logger.info("Module 15 SafetyService initialized.")
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._is_active = False
            self.sos._cancel_existing_timer()
            logger.info("Module 15 SafetyService stopped.")

    def process_query(self, request: IntentRequest) -> IntentResponse:
        """Process SOS intents."""
        if not self._is_active:
            return IntentResponse(
                success=False,
                intent=request.intent,
                spoken_response_hi="Safety SOS service inactive hai.",
                spoken_response_en="Safety SOS service is inactive.",
            )

        with self._lock:
            query = (request.raw_query or "").lower()

            # Direct SOS dispatch command (e.g. "Nayra send SOS", "send SOS", "SOS bhejo")
            is_direct_dispatch = (
                request.parameters.get("direct_dispatch", False)
                or any(k in query for k in ("send sos", "nayra send sos", "sos bhejo", "bhejo sos", "send emergency"))
            )
            if is_direct_dispatch:
                spoken = self.sos.confirm_sos()
                primary_phone = "+919215793330"
                for c in self.sos.contacts:
                    if c.is_primary:
                        primary_phone = c.phone
                        break
                spoken_en = f"Emergency alert activated. SOS notification has been automatically dispatched to caregiver at {primary_phone}."
                return IntentResponse(
                    success=True,
                    intent=UserIntent.CONFIRM_SOS,
                    spoken_response_hi=spoken,
                    spoken_response_en=spoken_en,
                    priority=SpeechPriority.CRITICAL_SAFETY,
                    data={"state": self.sos.state.value, "dispatched_to": primary_phone},
                )

            # Confirm SOS
            if request.intent == UserIntent.CONFIRM_SOS or any(k in query for k in ("haan", "yes", "confirm", "sach mein", "bhejo")):
                spoken = self.sos.confirm_sos()
                primary_phone = "+919215793330"
                for c in self.sos.contacts:
                    if c.is_primary:
                        primary_phone = c.phone
                        break
                return IntentResponse(
                    success=True,
                    intent=request.intent,
                    spoken_response_hi=spoken,
                    spoken_response_en=f"Emergency alert confirmed. SOS alert dispatched to {primary_phone}.",
                    priority=SpeechPriority.CRITICAL_SAFETY,
                    data={"state": self.sos.state.value, "dispatched_to": primary_phone},
                )

            # Cancel SOS
            if request.intent == UserIntent.CANCEL_SOS or any(k in query for k in ("cancel", "nahi", "ruk jao", "galti", "no")):
                spoken = self.sos.cancel_sos()
                return IntentResponse(
                    success=True,
                    intent=request.intent,
                    spoken_response_hi=spoken,
                    spoken_response_en="Emergency alert has been cancelled.",
                    priority=SpeechPriority.EMERGENCY_INTERACTION,
                    data={"state": self.sos.state.value},
                )

            # Trigger SOS
            spoken = self.sos.request_sos(reason=f"Spoken query: {request.raw_query}")
            return IntentResponse(
                success=True,
                intent=UserIntent.TRIGGER_SOS,
                spoken_response_hi=spoken,
                spoken_response_en="Do you need emergency assistance? Say yes or cancel.",
                priority=SpeechPriority.EMERGENCY_INTERACTION,
                data={"state": self.sos.state.value},
            )
