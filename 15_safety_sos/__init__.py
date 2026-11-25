"""Module 15: Safety & SOS Assistance."""

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
    from .sos_manager import SOSManager, SOSState, EmergencyContact
    from .safety_service import SafetyService
except (ImportError, ValueError):
    from sos_manager import SOSManager, SOSState, EmergencyContact
    from safety_service import SafetyService

if "safety_sos" not in sys.modules:
    sys.modules["safety_sos"] = sys.modules[__name__]

__all__ = [
    "SOSManager",
    "SOSState",
    "EmergencyContact",
    "SafetyService",
]
