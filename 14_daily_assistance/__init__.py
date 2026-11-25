"""Module 14: Daily Assistance & Reminders."""

from __future__ import annotations

from pathlib import Path
import sys

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from .reminder_db import ReminderDB, ReminderRecord
    from .reminder_scheduler import ReminderScheduler
    from .daily_service import DailyService
except (ImportError, ValueError):
    from reminder_db import ReminderDB, ReminderRecord
    from reminder_scheduler import ReminderScheduler
    from daily_service import DailyService

if "daily_assistance" not in sys.modules:
    sys.modules["daily_assistance"] = sys.modules[__name__]

__all__ = [
    "ReminderDB",
    "ReminderRecord",
    "ReminderScheduler",
    "DailyService",
]
