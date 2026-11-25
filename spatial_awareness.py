"""NAYRA — Public Spatial Awareness Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "05_spatial_awareness"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("05_spatial_awareness")

if "spatial_awareness" not in sys.modules:
    sys.modules["spatial_awareness"] = sys.modules[__name__]

for attr in dir(_pkg):
    if not attr.startswith("_"):
        globals()[attr] = getattr(_pkg, attr)
