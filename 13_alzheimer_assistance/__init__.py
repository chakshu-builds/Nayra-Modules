"""Module 13: Alzheimer's & Cognitive Assistance."""

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
    from .patient_profile import PatientProfileManager, FamiliarPerson
    from .orientation_guide import OrientationGuide
    from .alzheimer_service import AlzheimerService
except (ImportError, ValueError):
    from patient_profile import PatientProfileManager, FamiliarPerson
    from orientation_guide import OrientationGuide
    from alzheimer_service import AlzheimerService

if "alzheimer_assistance" not in sys.modules:
    sys.modules["alzheimer_assistance"] = sys.modules[__name__]

__all__ = [
    "PatientProfileManager",
    "FamiliarPerson",
    "OrientationGuide",
    "AlzheimerService",
]
