"""NAYRA — Public Safety & SOS Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "15_safety_sos"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("15_safety_sos")

SOSManager = _pkg.SOSManager
SOSState = _pkg.SOSState
EmergencyContact = _pkg.EmergencyContact
SafetyService = _pkg.SafetyService

if "safety_sos" not in sys.modules:
    sys.modules["safety_sos"] = sys.modules[__name__]

__all__ = [
    "SOSManager",
    "SOSState",
    "EmergencyContact",
    "SafetyService",
]
