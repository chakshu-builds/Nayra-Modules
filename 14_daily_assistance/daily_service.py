"""Daily Assistance Service managing reminders and voice interactions."""

from __future__ import annotations

import datetime
import logging
import re
import threading
import time
from typing import Any, List, Optional

from shared.interfaces import IAssistiveModule, IVoiceOutput
from shared.schemas import (
    IntentRequest,
    IntentResponse,
    SpeechPriority,
    UserIntent,
)

try:
    from .reminder_db import ReminderDB, ReminderRecord
    from .reminder_scheduler import ReminderScheduler
except (ImportError, ValueError):
    from reminder_db import ReminderDB, ReminderRecord
    from reminder_scheduler import ReminderScheduler

logger = logging.getLogger("NAYRA-DailyService")


class DailyService(IAssistiveModule):
    """Module 14 service managing daily assistance, scheduling, hydration, and medication reminders."""

    def __init__(
        self,
        voice_output: Optional[IVoiceOutput] = None,
        db: Optional[ReminderDB] = None,
        preferred_language: str = "en",
    ) -> None:
        self.voice_output = voice_output
        self.db = db or ReminderDB()
        self.preferred_language = preferred_language
        self.scheduler = ReminderScheduler(
            db=self.db, voice_output=self.voice_output, preferred_language=self.preferred_language
        )
        self._lock = threading.RLock()
        self._is_active = False

    @property
    def module_name(self) -> str:
        return "14_daily_assistance"

    def initialize(self) -> bool:
        with self._lock:
            self._is_active = True
            if getattr(self.db, "was_recovered", False):
                msg = f"Reminder database was corrupted and automatically recovered. Backup saved at: {getattr(self.db, 'last_corrupted_backup_path', 'unknown')}"
                logger.warning(msg)
                if self.voice_output:
                    self.voice_output.speak(
                        "Savdhaan, purani reminders database kharab ho gayi thi. Ek surakshit backup banakar nayi database shuru ki gayi hai.",
                        priority=SpeechPriority.REMINDER,
                    )
            self.scheduler.start()
            logger.info("Module 14 DailyService initialized.")
            return True

    def shutdown(self) -> None:
        with self._lock:
            self._is_active = False
            self.scheduler.stop()
            logger.info("Module 14 DailyService stopped.")

    def parse_time_offset(self, query: str) -> float:
        """Parse relative time offsets like '10 minute', '2 hours' or clock times like '5:00 pm', '6 baje'."""
        q = query.lower()
        now = time.time()

        # 1. Clock time with AM/PM: e.g. "5:00 pm", "6 pm", "8 am", "at 4:30 pm"
        m_clk = re.search(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)", q)
        if m_clk:
            hr = int(m_clk.group(1))
            mn = int(m_clk.group(2)) if m_clk.group(2) else 0
            meridiem = m_clk.group(3).lower()
            if meridiem == "pm" and hr < 12:
                hr += 12
            elif meridiem == "am" and hr == 12:
                hr = 0
            now_dt = datetime.datetime.now()
            target_dt = now_dt.replace(hour=hr, minute=mn, second=0, microsecond=0)
            if target_dt.timestamp() <= now:
                target_dt += datetime.timedelta(days=1)
            return target_dt.timestamp()

        # 2. Hindi clock time: e.g. "shaam 6 baje", "subah 8 baje", "dopahar 2 baje", "5 baje"
        m_hi_clk = re.search(r"(subah|dopahar|shaam|raat)?\s*(\d{1,2})(?::(\d{2}))?\s*baje", q)
        if m_hi_clk:
            period = m_hi_clk.group(1) or ""
            hr = int(m_hi_clk.group(2))
            mn = int(m_hi_clk.group(3)) if m_hi_clk.group(3) else 0
            if period in ("shaam", "raat") and hr < 12:
                hr += 12
            elif period == "dopahar" and 1 <= hr <= 4:
                hr += 12
            now_dt = datetime.datetime.now()
            target_dt = now_dt.replace(hour=hr, minute=mn, second=0, microsecond=0)
            if target_dt.timestamp() <= now:
                target_dt += datetime.timedelta(days=1)
            return target_dt.timestamp()

        # 3. Relative seconds
        m_sec = re.search(r"(\d+)\s*(?:second|sec|second baad)", q)
        if m_sec:
            return now + float(m_sec.group(1))

        # 4. Relative minutes
        m_min = re.search(r"(\d+)\s*(?:minute|min|minute baad)", q)
        if m_min:
            return now + (float(m_min.group(1)) * 60.0)

        # 5. Relative hours
        m_hr = re.search(r"(\d+)\s*(?:ghante|hour|hours|ghante baad)", q)
        if m_hr:
            return now + (float(m_hr.group(1)) * 3600.0)

        # Default fallback: 1 hour from now
        return now + 3600.0

    def process_query(self, request: IntentRequest) -> IntentResponse:
        """Process reminder management intents."""
        if not self._is_active:
            return IntentResponse(
                success=False,
                intent=request.intent,
                spoken_response_hi="Daily assistance service inactive hai.",
                spoken_response_en="Daily assistance service is inactive.",
            )

        with self._lock:
            query = (request.raw_query or "").lower()

            # 1. Snooze intent
            if request.intent == UserIntent.SNOOZE_REMINDER or "snooze" in query or "baad mein" in query:
                actives = self.db.list_active()
                if actives:
                    self.db.snooze(actives[0].id, snooze_seconds=600.0)
                    spoken_hi = "Maine reminder ko 10 minute ke liye snooze kar diya hai."
                    spoken_en = "I have snoozed the reminder for 10 minutes."
                else:
                    spoken_hi = "Koi active reminder nahi hai jise snooze kiya ja sake."
                    spoken_en = "There are no active reminders to snooze."

                if self.voice_output:
                    chosen = spoken_en if (self.preferred_language == "en" or request.detected_language == "en") else spoken_hi
                    self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)

                return IntentResponse(
                    success=True,
                    intent=request.intent,
                    spoken_response_hi=spoken_hi,
                    spoken_response_en=spoken_en,
                )

            # 2. Delete / Dismiss / Medication taken intent
            if request.intent == UserIntent.DELETE_REMINDER or any(w in query for w in ("dismiss", "hata", "delete", "le li", "kha li", "taken", "complete", "done")):
                actives = self.db.list_active()
                if actives:
                    self.db.mark_completed(actives[0].id)
                    spoken_hi = f"Theek hai, maine '{actives[0].title}' ko complete mark kar diya hai."
                    spoken_en = f"Understood, I have marked '{actives[0].title}' as completed."
                else:
                    spoken_hi = "Hataane ya complete karne ke liye koi active reminder nahi mila."
                    spoken_en = "No active reminder found to mark as completed."

                if self.voice_output:
                    chosen = spoken_en if (self.preferred_language == "en" or request.detected_language == "en") else spoken_hi
                    self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)

                return IntentResponse(
                    success=True,
                    intent=request.intent,
                    spoken_response_hi=spoken_hi,
                    spoken_response_en=spoken_en,
                )

            # 3. List reminders intent
            if request.intent == UserIntent.LIST_REMINDERS or "kya reminder" in query or "kaunsi dawai" in query or "list" in query:
                actives = self.db.list_active()
                if not actives:
                    spoken_hi = "Aapka abhi koi pending reminder nahi hai."
                    spoken_en = "You have no pending reminders."
                else:
                    titles = [r.title for r in actives[:3]]
                    spoken_hi = f"Aapke pass {len(actives)} reminder hain: " + ", ".join(titles) + "."
                    plur = "reminders" if len(actives) != 1 else "reminder"
                    spoken_en = f"You have {len(actives)} {plur}: " + ", ".join(titles) + "."

                if self.voice_output:
                    chosen = spoken_en if (self.preferred_language == "en" or request.detected_language == "en") else spoken_hi
                    self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)

                return IntentResponse(
                    success=True,
                    intent=request.intent,
                    spoken_response_hi=spoken_hi,
                    spoken_response_en=spoken_en,
                    data={"count": len(actives)},
                )

            # 4. Set reminder intent
            if request.intent == UserIntent.SET_REMINDER or "yaad dilana" in query or "remind" in query or ("dawai" in query and "rakh" in query):
                if self.preferred_language == "en":
                    title = request.target_entity or "medicine"
                    if "paani" in query or "water" in query:
                        title = "drinking water" if ("peene" in query or "peena" in query or "drink" in query) else "water"
                        category = "hydration"
                    elif "dawai" in query or "medicine" in query:
                        title = "medicine"
                        category = "medication"
                    else:
                        category = "general"
                else:
                    title = request.target_entity or "Dawai"
                    if "paani" in query or "water" in query:
                        title = "Paani peene" if ("peene" in query or "peena" in query or "drink" in query) else "Paani"
                        category = "hydration"
                    elif "dawai" in query or "medicine" in query:
                        title = "Dawai"
                        category = "medication"
                    else:
                        category = "general"

                due_ts = self.parse_time_offset(query)
                rem_id = self.db.add_reminder(title=title, due_timestamp=due_ts, category=category)

                spoken_hi = f"Theek hai, maine {title} ke liye reminder set kar diya hai."
                spoken_en = f"Understood, I have set a reminder for {title}."

                if self.voice_output:
                    chosen = spoken_en if (self.preferred_language == "en" or request.detected_language == "en") else spoken_hi
                    self.voice_output.speak(chosen, priority=SpeechPriority.USER_QUERY_RESPONSE)

                return IntentResponse(
                    success=True,
                    intent=request.intent,
                    spoken_response_hi=spoken_hi,
                    spoken_response_en=spoken_en,
                    data={"reminder_id": rem_id, "title": title, "due_timestamp": due_ts},
                )

            # Default
            spoken_hi = "Main reminders aur daily routine mein aapki madad kar sakti hoon."
            spoken_en = "I can assist you with your daily schedule and reminders."
            return IntentResponse(
                success=True,
                intent=request.intent,
                spoken_response_hi=spoken_hi,
                spoken_response_en=spoken_en,
            )
