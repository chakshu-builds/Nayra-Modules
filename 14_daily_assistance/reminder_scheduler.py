"""Background reminder scheduler checking due items."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from shared.interfaces import IVoiceOutput
from shared.schemas import SpeechPriority
try:
    from .reminder_db import ReminderDB, ReminderRecord
except (ImportError, ValueError):
    from reminder_db import ReminderDB, ReminderRecord

logger = logging.getLogger("NAYRA-ReminderScheduler")


class ReminderScheduler:
    """Daemon thread monitoring SQLite database for due reminders and firing voice alerts."""

    def __init__(
        self,
        db: ReminderDB,
        voice_output: Optional[IVoiceOutput] = None,
        check_interval_sec: float = 2.0,
        reminder_callback: Optional[Callable[[ReminderRecord], None]] = None,
        preferred_language: str = "en",
    ) -> None:
        self.db = db
        self.voice_output = voice_output
        self.check_interval_sec = check_interval_sec
        self.reminder_callback = reminder_callback
        self.preferred_language = preferred_language
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background checking thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="NAYRA-ReminderScheduler", daemon=True)
        self._thread.start()
        logger.info("ReminderScheduler background worker started.")

    def stop(self) -> None:
        """Stop the background worker."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("ReminderScheduler stopped.")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                now = time.time()
                due_records = self.db.get_due_reminders(now)
                if due_records:
                    # Fire only one announcement per check cycle to prevent continuous repetitive spamming
                    self._dispatch_reminder(due_records[0])
                    # Postpone other concurrent overdue records so they don't blast back to back
                    for other in due_records[1:]:
                        self.db.mark_announced(other.id, re_alert_delay_sec=600.0)
            except Exception as e:
                logger.error(f"Error in reminder scheduler loop: {e}")

            self._stop_event.wait(self.check_interval_sec)

    def _dispatch_reminder(self, rec: ReminderRecord) -> None:
        """Announce due reminder and mark as processed."""
        raw_title = (rec.title or "medicine").strip()
        t_low = raw_title.lower()
        if "dawai" in t_low or "medicine" in t_low or "samay" in t_low:
            title_en = "medicine"
            title_hi = "dawai"
        elif "paani" in t_low or "water" in t_low:
            title_en = "drinking water"
            title_hi = "paani"
        else:
            title_en = raw_title
            title_hi = raw_title

        spoken_hi = f"Dhyan dijiye! Aapka time ho gaya hai {title_hi} lene ka."
        spoken_en = f"Attention! It's time to take your {title_en}."

        logger.info(f"Firing due reminder #{rec.id}: {title_en} (raw: {rec.title})")

        if self.voice_output:
            chosen = spoken_en if self.preferred_language == "en" else spoken_hi
            self.voice_output.speak(chosen, priority=SpeechPriority.REMINDER)

        if self.reminder_callback:
            try:
                self.reminder_callback(rec)
            except Exception as e:
                logger.error(f"Error in user reminder callback: {e}")

        # Postpone next alert by 10 minutes so it NEVER repeats continuously
        self.db.mark_announced(rec.id, re_alert_delay_sec=600.0)
