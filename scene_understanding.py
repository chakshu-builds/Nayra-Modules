"""NAYRA — Public Scene Understanding Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "12_scene_understanding"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("12_scene_understanding")

SceneAnalyzer = _pkg.SceneAnalyzer
SceneDescription = _pkg.SceneDescription
SceneService = _pkg.SceneService

if "scene_understanding" not in sys.modules:
    sys.modules["scene_understanding"] = sys.modules[__name__]

__all__ = [
    "SceneAnalyzer",
    "SceneDescription",
    "SceneService",
]
