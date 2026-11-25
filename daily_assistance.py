"""NAYRA — Public Daily Assistance Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "14_daily_assistance"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("14_daily_assistance")

ReminderDB = _pkg.ReminderDB
ReminderRecord = _pkg.ReminderRecord
ReminderScheduler = _pkg.ReminderScheduler
DailyService = _pkg.DailyService

if "daily_assistance" not in sys.modules:
    sys.modules["daily_assistance"] = sys.modules[__name__]

__all__ = [
    "ReminderDB",
    "ReminderRecord",
    "ReminderScheduler",
    "DailyService",
]
