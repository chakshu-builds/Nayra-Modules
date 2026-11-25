"""NAYRA — Public Alzheimer's Assistance Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "13_alzheimer_assistance"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("13_alzheimer_assistance")

PatientProfileManager = _pkg.PatientProfileManager
FamiliarPerson = _pkg.FamiliarPerson
OrientationGuide = _pkg.OrientationGuide
AlzheimerService = _pkg.AlzheimerService

if "alzheimer_assistance" not in sys.modules:
    sys.modules["alzheimer_assistance"] = sys.modules[__name__]

__all__ = [
    "PatientProfileManager",
    "FamiliarPerson",
    "OrientationGuide",
    "AlzheimerService",
]
