"""NAYRA — Public Camera Input Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "01_camera_input"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("01_camera_input")

if "camera_input" not in sys.modules:
    sys.modules["camera_input"] = sys.modules[__name__]

for attr in dir(_pkg):
    if not attr.startswith("_"):
        globals()[attr] = getattr(_pkg, attr)
