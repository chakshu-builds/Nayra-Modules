"""NAYRA — Public Object Tracking Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "04_object_tracking"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("04_object_tracking")

if "object_tracking" not in sys.modules:
    sys.modules["object_tracking"] = sys.modules[__name__]

for attr in dir(_pkg):
    if not attr.startswith("_"):
        globals()[attr] = getattr(_pkg, attr)
