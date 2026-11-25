"""Intent Router and Classifier for multilingual Hindi/Hinglish/English voice queries."""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from shared.schemas import IntentRequest, UserIntent

logger = logging.getLogger("NAYRA-IntentRouter")

# Known common object entity tokens for Hindi and English
OBJECT_TOKENS = [
    "phone", "mobile", "chashma", "spectacles", "glasses", "chabi", "keys",
    "bottle", "botal", "cup", "mug", "book", "kitaab", "remote", "wallet",
    "purse", "bag", "laptop", "pen", "plate", "dawai", "medicine"
]


class IntentClassifier:
    """Classifies spoken queries into structured UserIntents and extracts target entities."""

    def __init__(self, require_wake_word: bool = False) -> None:
        self.require_wake_word = require_wake_word

    def extract_wake_word(self, query: str) -> Tuple[bool, str]:
        """Detect wake word prefix or mention and strip it if present.

        Recognized wake triggers:
        - 'hey nayra', 'hey naira', 'hey nyra', 'ok nayra', 'ok naira', 'hello nayra', 'hello naira', 'sun nayra', 'sun naira', 'nayra', 'naira', 'nyra'
        """
        q = query.strip()
        pattern = r"\b(?:hey\s+nayra|hey\s+naira|hey\s+nyra|ok\s+nayra|ok\s+naira|hello\s+nayra|hello\s+naira|sun\s+nayra|sun\s+naira|nayra|naira|nyra)\b"
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            stripped = re.sub(pattern, "", q, count=1, flags=re.IGNORECASE).strip()
            stripped = re.sub(r"^[\s,.\-!?]+|[\s,.\-!?]+$", "", stripped).strip()
            return True, stripped
        return False, q

    def detect_language(self, query: str) -> str:
        """Detect whether language is predominantly Hindi/Hinglish or English."""
        q = query.lower()
        hindi_markers = ["mera", "meri", "kahan", "hai", "kya", "batao", "saamne", "madad", "dawai", "yaad", "kaun", "bhejo", "haan", "nahi"]
        for marker in hindi_markers:
            if re.search(r"\b" + marker + r"\b", q):
                return "hi"
        return "en"

    def classify(self, query: str, pending_intent: Optional[UserIntent] = None) -> IntentRequest:
        """Parse natural query string into structured IntentRequest."""
        q = query.strip()
        lang = self.detect_language(q)

        # Context-dependent overrides (e.g. pending SOS confirmation) bypass wake-word
        if pending_intent == UserIntent.TRIGGER_SOS:
            q_lower = q.lower()
            if any(w in q_lower for w in ("haan", "yes", "sach mein", "bhejo", "confirm", "haan madad")):
                return IntentRequest(raw_query=q, intent=UserIntent.CONFIRM_SOS, detected_language=lang)
            if any(w in q_lower for w in ("nahi", "cancel", "no", "ruk", "galti")):
                return IntentRequest(raw_query=q, intent=UserIntent.CANCEL_SOS, detected_language=lang)

        # Immediate STOP / Silence command bypasses wake word requirement
        q_lower_full = q.lower()
        if any(w in q_lower_full for w in ("stop", "chup", "be quiet", "stop talking", "shut up", "chup raho", "bas karo", "halt", "ruko", "ruk jao", "silence")):
            return IntentRequest(raw_query=q, intent=UserIntent.STOP, detected_language=lang)

        # Wake word detection
        has_wake, stripped_q = self.extract_wake_word(q)

        # If wake word is strictly required and not spoken (and no pending dialog), silently ignore
        if self.require_wake_word and not has_wake and pending_intent is None:
            return IntentRequest(raw_query=q, intent=UserIntent.SILENT_IGNORE, detected_language=lang)

        # If user uttered ONLY the wake word ("Hey Nayra" / "Nayra")
        if has_wake and not stripped_q:
            return IntentRequest(raw_query=q, intent=UserIntent.WAKE_GREETING, detected_language=lang)

        active_q = stripped_q if has_wake else q
        q_lower = active_q.lower()

        # 0. Capability & Help Query (evaluated before SOS so "kya madad kar sakti ho" does not trigger alarm)
        if any(w in q_lower for w in ("kya kar sakti", "kaise madad", "kya madad", "features", "commands", "options", "what can you do")):
            return IntentRequest(raw_query=q, intent=UserIntent.HELP, detected_language=lang)

        # 1. Emergency / SOS (supports "Nayra send SOS", "send SOS", "SOS bhejo")
        if any(w in q_lower for w in ("send sos", "sos bhejo", "bhejo sos")):
            return IntentRequest(
                raw_query=q,
                intent=UserIntent.TRIGGER_SOS,
                parameters={"direct_dispatch": True},
                detected_language=lang,
            )
        if any(w in q_lower for w in ("emergency", "bachao", "help me", "fell down", "gir gaya", "khatra", "madad karo", "mujhe bachao", "sos")):
            return IntentRequest(raw_query=q, intent=UserIntent.TRIGGER_SOS, detected_language=lang)

        # 2. General Conversational Assistant (normal, friendly companion chat)
        chat_type = None
        if any(w in q_lower for w in ("kaun ho", "who are you", "tum kaun ho", "aap kaun ho", "tera naam", "aapka naam", "what is your name")):
            chat_type = "identity"
        elif any(w in q_lower for w in ("how are you", "kaisi ho", "kaise ho", "kya haal", "sab theek", "how do you do")):
            chat_type = "wellbeing"
        elif any(w in q_lower for w in ("hello", "namaste", "good morning", "good evening", "good afternoon", "kem cho", "ram ram", "hi there")):
            chat_type = "greeting"
        elif any(w in q_lower for w in ("baat karo", "kuch batao", "bore ho", "feeling lonely", "kuch sunao", "talk to me", "tell me something", "mann nahi lag raha", "akele", "lonely")):
            chat_type = "companionship"
        elif any(w in q_lower for w in ("thank you", "dhanyawad", "shukriya", "thanks", "bahut accha", "good job", "bahut badhiya", "shabaash")):
            chat_type = "gratitude"
        elif any(w in q_lower for w in ("kya kar rahi ho", "what are you doing", "what are you up to")):
            chat_type = "activity"

        if chat_type:
            return IntentRequest(
                raw_query=q,
                intent=UserIntent.GENERAL_CHAT,
                parameters={"chat_type": chat_type},
                detected_language=lang,
            )

        # 3. Obstacle / Path check
        if any(w in q_lower for w in ("rasta saaf", "clear path", "obstacle", "rukawat", "takkar", "aage rasta", "koi rukawat", "saamne rukawat", "path clear", "khatra")):
            return IntentRequest(raw_query=q, intent=UserIntent.OBSTACLE_CHECK, detected_language=lang)

        # 3. Reading / OCR text
        if any(w in q_lower for w in ("parh", "padh", "read", "likha", "board", "text", "what is written", "what does it say", "read this", "read out", "kya likha", "padho")):
            return IntentRequest(raw_query=q, intent=UserIntent.READ_TEXT, detected_language=lang)

        # 4. Cognitive / Orientation / Person ID
        if any(w in q_lower for w in ("kaun hai", "kon hai", "who is", "pehchano", "identify")):
            # Extract target name if present
            target = None
            m = re.search(r"(?:who is|kaun hai|kya yeh)\s+([a-zA-Z]+)", q_lower)
            if m:
                target = m.group(1).capitalize()
            return IntentRequest(raw_query=q, intent=UserIntent.IDENTIFY_PERSON, target_entity=target, detected_language=lang)

        if any(w in q_lower for w in ("kahan hoon", "where am i", "time kya", "samay", "aaj kaun sa din", "date kya")):
            return IntentRequest(raw_query=q, intent=UserIntent.HELP, detected_language=lang)

        # 5. Reminders & Daily schedule
        if any(w in q_lower for w in ("snooze", "baad mein yaad")):
            return IntentRequest(raw_query=q, intent=UserIntent.SNOOZE_REMINDER, detected_language=lang)

        if any(w in q_lower for w in ("dismiss", "hata do", "delete reminder", "dawai le li", "kha li", "medicine taken")):
            return IntentRequest(raw_query=q, intent=UserIntent.DELETE_REMINDER, detected_language=lang)

        if any(w in q_lower for w in ("kya reminder", "reminders", "dawaiyan", "list reminder", "kab leni hai", "medicine kab", "dawai kab")):
            return IntentRequest(raw_query=q, intent=UserIntent.LIST_REMINDERS, detected_language=lang)

        if any(w in q_lower for w in ("yaad dilana", "remind me", "set reminder", "reminder laga", "remind")):
            target = self._extract_entity(q_lower) or "reminder"
            return IntentRequest(raw_query=q, intent=UserIntent.SET_REMINDER, target_entity=target, detected_language=lang)

        # 6. Object Location Memory
        if any(w in q_lower for w in ("kahan hai", "where is", "dhundo", "find", "locate", "dikha")):
            target = self._extract_entity(q_lower)
            return IntentRequest(raw_query=q, intent=UserIntent.FIND_OBJECT, target_entity=target, detected_language=lang)

        # 7. Scene description (surroundings, ambient environment)
        if any(w in q_lower for w in ("saamne kya", "aas paas", "describe", "surroundings", "kya dikh raha", "around me", "in front of", "in front", "what is front", "what is in front", "ahead", "describe scene")):
            return IntentRequest(raw_query=q, intent=UserIntent.DESCRIBE_SCENE, detected_language=lang)

        # 8. Help / Capabilities
        if any(w in q_lower for w in ("madad", "help", "kya kar sakti", "features", "commands")):
            return IntentRequest(raw_query=q, intent=UserIntent.HELP, detected_language=lang)

        # Fallback
        return IntentRequest(raw_query=q, intent=UserIntent.UNKNOWN, detected_language=lang)

    def _extract_entity(self, query: str) -> Optional[str]:
        """Extract object or target entity name from query string."""
        for token in OBJECT_TOKENS:
            if token in query:
                return token
        return None
