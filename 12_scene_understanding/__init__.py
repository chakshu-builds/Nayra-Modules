"""Module 12: Contextual Scene Understanding."""

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
    from .scene_analyzer import SceneAnalyzer, SceneDescription
    from .scene_service import SceneService
except (ImportError, ValueError):
    from scene_analyzer import SceneAnalyzer, SceneDescription
    from scene_service import SceneService

if "scene_understanding" not in sys.modules:
    sys.modules["scene_understanding"] = sys.modules[__name__]

__all__ = [
    "SceneAnalyzer",
    "SceneDescription",
    "SceneService",
]
