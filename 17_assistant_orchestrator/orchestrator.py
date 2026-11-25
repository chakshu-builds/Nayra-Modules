"""Central Assistant Orchestrator dispatching queries to registered domain modules."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from shared.interfaces import IAssistiveModule, IVoiceOutput
from shared.schemas import (
    IntentRequest,
    IntentResponse,
    SpeechPriority,
    UserIntent,
)

try:
    from .intent_router import IntentClassifier
except (ImportError, ValueError):
    from intent_router import IntentClassifier

logger = logging.getLogger("NAYRA-AssistantOrchestrator")


class AssistantOrchestrator:
    """Coordinates voice queries, dialog context, and assistive domain modules."""

    def __init__(
        self,
        voice_output: Optional[IVoiceOutput] = None,
        classifier: Optional[IntentClassifier] = None,
        preferred_language: str = "en",
    ) -> None:
        self.voice_output = voice_output
        self.classifier = classifier or IntentClassifier()
        self.preferred_language = preferred_language
        self._modules: Dict[str, IAssistiveModule] = {}
        self._lock = threading.RLock()
        self._pending_context_intent: Optional[UserIntent] = None
        self._is_active = False

    def register_module(self, name: str, module: IAssistiveModule) -> None:
        """Register a domain module (e.g. '07_ocr', '10_object_memory', '11_obstacle_awareness', etc.)."""
        with self._lock:
            self._modules[name] = module
            logger.info(f"Registered assistive module: {name}")

    def initialize_all(self) -> bool:
        """Initialize all registered modules."""
        with self._lock:
            self._is_active = True
            all_ok = True
            for name, mod in self._modules.items():
                try:
                    ok = mod.initialize()
                    if not ok:
                        logger.warning(f"Module {name} initialize returned False.")
                        all_ok = False
                except Exception as e:
                    logger.error(f"Error initializing module {name}: {e}")
                    all_ok = False
            return all_ok

    def shutdown_all(self) -> None:
        """Shutdown all registered modules."""
        with self._lock:
            self._is_active = False
            for name, mod in self._modules.items():
                try:
                    mod.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down module {name}: {e}")

    def process_voice_query(self, raw_query: str) -> IntentResponse:
        """Receive speech string, classify intent, dispatch to appropriate module, and return response."""
        with self._lock:
            intent_req = self.classifier.classify(raw_query, pending_intent=self._pending_context_intent)
            logger.info(f"Classified query '{raw_query}' -> Intent: {intent_req.intent.name}, Target: {intent_req.target_entity}")

            # Track dialog state for SOS confirmation
            if intent_req.intent == UserIntent.TRIGGER_SOS:
                self._pending_context_intent = UserIntent.TRIGGER_SOS
            elif intent_req.intent in (UserIntent.CONFIRM_SOS, UserIntent.CANCEL_SOS):
                self._pending_context_intent = None

            response = self._route_intent(intent_req)

            # Speak response through voice output if available and not silent ignore
            if self.voice_output and intent_req.intent != UserIntent.SILENT_IGNORE:
                if self.preferred_language == "en":
                    chosen = response.spoken_response_en or response.spoken_response_hi
                else:
                    chosen = response.spoken_response_hi if intent_req.detected_language in ("hi", "hinglish") else response.spoken_response_en
                if chosen:
                    self.voice_output.speak(chosen, priority=response.priority)

            return response

    def _generate_conversational_response(self, request: IntentRequest) -> Tuple[str, str]:
        """Generate empathetic, warm conversational responses for general companion chat."""
        chat_type = request.parameters.get("chat_type", "general")
        if chat_type == "greeting":
            return (
                "Namaste! Main NAYRA hoon, aapki companion. Aapka din kaisa ja raha hai?",
                "Hello! I am NAYRA, your assistive companion. How is your day going?",
            )
        elif chat_type == "wellbeing":
            return (
                "Main bilkul theek hoon, shukriya! Aap bataiye, aap kaisa mehsoos kar rahe hain?",
                "I am doing very well, thank you! How are you feeling today?",
            )
        elif chat_type == "identity":
            return (
                "Main NAYRA hoon, aapki personal assistive companion. Main aapki har kadam par madad karne aur aapse baat karne ke liye hamesha yahan hoon.",
                "I am NAYRA, your personal assistive companion. I am always here to help you navigate, remember objects, manage reminders, and chat with you.",
            )
        elif chat_type == "companionship":
            return (
                "Main hamesha aapke saath hoon. Chinta mat kijiye, aap akele nahi hain. Chaliye bataiye, aaj aapka kya karne ka mann hai?",
                "I am always right here with you. You are never alone. Tell me, what would you like to talk about today?",
            )
        elif chat_type == "gratitude":
            return (
                "Aapka bahut-bahut swagat hai! Aapki madad karna meri khushi hai.",
                "You are very welcome! It is always my pleasure to assist you.",
            )
        elif chat_type == "activity":
            return (
                "Main aapke aas-paas ka dhyan rakh rahi hoon aur aapki madad ke liye taiyaar hoon.",
                "I am keeping an eye on your surroundings and ready to assist you whenever you need.",
            )
        return (
            "Main aapki baat sun rahi hoon. Main aapse baat karke bahut khush hoon. Bataiye, kya chal raha hai?",
            "I'm listening and I'm very glad to talk with you. What is on your mind?",
        )

    def _route_intent(self, request: IntentRequest) -> IntentResponse:
        """Route structured IntentRequest to target module."""
        intent = request.intent

        # 0A. Wake word greeting ("Hey Nayra")
        if intent == UserIntent.WAKE_GREETING:
            return IntentResponse(
                success=True,
                intent=UserIntent.WAKE_GREETING,
                spoken_response_hi="Haan ji, main sun rahi hoon. Kahiye, main aapki kya madad kar sakti hoon?",
                spoken_response_en="Yes, I am listening. How can I help you?",
                priority=SpeechPriority.USER_QUERY_RESPONSE,
            )

        # 0A2. Stop / Silence command (halts speech output immediately)
        if intent == UserIntent.STOP:
            if self.voice_output and hasattr(self.voice_output, "stop"):
                self.voice_output.stop()
            mod = self._modules.get("14_daily_assistance")
            if mod and hasattr(mod, "db"):
                try:
                    actives = mod.db.list_active()
                    for act in actives[:5]:
                        mod.db.mark_announced(act.id, re_alert_delay_sec=1800.0)
                except Exception:
                    pass
            return IntentResponse(
                success=True,
                intent=UserIntent.STOP,
                spoken_response_hi="",
                spoken_response_en="",
                priority=SpeechPriority.CRITICAL_SAFETY,
            )

        # 0B. Silent ignore (ambient chatter without wake word)
        if intent == UserIntent.SILENT_IGNORE:
            return IntentResponse(
                success=True,
                intent=UserIntent.SILENT_IGNORE,
                spoken_response_hi="",
                spoken_response_en="",
                priority=SpeechPriority.INFORMATIONAL,
            )

        # 0C. General conversational assistant
        if intent == UserIntent.GENERAL_CHAT:
            hi, en = self._generate_conversational_response(request)
            return IntentResponse(
                success=True,
                intent=UserIntent.GENERAL_CHAT,
                spoken_response_hi=hi,
                spoken_response_en=en,
                priority=SpeechPriority.USER_QUERY_RESPONSE,
            )

        # 1. Safety & SOS
        if intent in (UserIntent.TRIGGER_SOS, UserIntent.CONFIRM_SOS, UserIntent.CANCEL_SOS):
            mod = self._modules.get("15_safety_sos")
            if mod:
                return mod.process_query(request)

        # 2. Obstacle check
        if intent == UserIntent.OBSTACLE_CHECK:
            mod = self._modules.get("11_obstacle_awareness")
            if mod:
                return mod.process_query(request)

        # 3. Scene description
        if intent == UserIntent.DESCRIBE_SCENE:
            mod = self._modules.get("12_scene_understanding")
            if mod:
                return mod.process_query(request)

        # 4. Cognitive & Alzheimer's (person ID, temporal & spatial orientation)
        if intent == UserIntent.IDENTIFY_PERSON:
            mod = self._modules.get("13_alzheimer_assistance")
            if mod:
                return mod.process_query(request)

        if intent == UserIntent.HELP:
            # If the user specifically asks orientation or grounding questions ("kahan hoon", "samay", "din", "date"), route to Module 13
            query = (request.raw_query or "").lower()
            if any(k in query for k in ("kahan", "where am i", "time", "samay", "din", "date", "tarikh", "aaj kya", "chinta", "dar lag")):
                mod = self._modules.get("13_alzheimer_assistance")
                if mod:
                    return mod.process_query(request)
            # Otherwise, return truthful capability assistance overview
            spoken_hi = (
                "Main NAYRA hoon, aapki assistive companion. "
                "Aap mujhse cheezein dhundhne, saamne ka nazara dekhne, raste ki rukawat check karne, dawai reminder set karne, ya emergency alert ke liye keh sakte hain."
            )
            spoken_en = (
                "I am NAYRA, your assistive companion. "
                "You can ask me to find objects, describe surroundings, check for obstacles, manage reminders, or request emergency assistance."
            )
            return IntentResponse(
                success=True,
                intent=UserIntent.HELP,
                spoken_response_hi=spoken_hi,
                spoken_response_en=spoken_en,
                priority=SpeechPriority.USER_QUERY_RESPONSE,
            )

        # 5. Daily reminders & routine
        if intent in (UserIntent.SET_REMINDER, UserIntent.LIST_REMINDERS, UserIntent.DELETE_REMINDER, UserIntent.SNOOZE_REMINDER):
            mod = self._modules.get("14_daily_assistance")
            if mod:
                return mod.process_query(request)

        # 6. Object Location Memory
        if intent in (UserIntent.FIND_OBJECT, UserIntent.LIST_OBJECTS):
            mod = self._modules.get("10_object_memory")
            if mod:
                return mod.process_query(request)

        # 7. OCR / Text reading
        if intent == UserIntent.READ_TEXT:
            mod = self._modules.get("07_ocr")
            if mod:
                return mod.process_query(request)

        # Fallback / General assistance
        if request.intent == UserIntent.UNKNOWN:
            spoken_hi = "Maaf kijiye, main NAYRA hoon aur mujhe yeh samajh nahi aaya. Aap mujhse cheezein dhundhne, rasta check karne, ya dawai reminder ke liye keh sakte hain."
            spoken_en = "Sorry, I am NAYRA and I did not understand that. You can ask me to find objects, check your path, or manage reminders."
            return IntentResponse(
                success=True,
                intent=UserIntent.UNKNOWN,
                spoken_response_hi=spoken_hi,
                spoken_response_en=spoken_en,
                priority=SpeechPriority.USER_QUERY_RESPONSE,
            )

        spoken_hi = (
            "Main NAYRA hoon, aapki assistive companion. "
            "Aap mujhse cheezein dhundhne, saamne ka nazara janne, dawai yaad rakhne, ya emergency madad ke liye keh sakte hain."
        )
        spoken_en = (
            "I am NAYRA, your assistive companion. "
            "You can ask me to find objects, describe surroundings, manage reminders, or request emergency assistance."
        )
        return IntentResponse(
            success=True,
            intent=UserIntent.HELP,
            spoken_response_hi=spoken_hi,
            spoken_response_en=spoken_en,
            priority=SpeechPriority.USER_QUERY_RESPONSE,
        )
