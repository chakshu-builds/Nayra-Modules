"""NAYRA — Public Application Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "18_application"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("18_application")

AppConfig = _pkg.AppConfig
NayraApplication = _pkg.NayraApplication

if "nayra_app" not in sys.modules:
    sys.modules["nayra_app"] = sys.modules[__name__]

__all__ = [
    "AppConfig",
    "NayraApplication",
]
