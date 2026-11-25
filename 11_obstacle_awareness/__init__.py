"""Module 11: Obstacle Awareness & Direction Guidance."""

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
    from .obstacle_detector import ObstacleDetector
    from .warning_generator import ObstacleWarningGenerator
    from .obstacle_service import ObstacleService
except (ImportError, ValueError):
    from obstacle_detector import ObstacleDetector
    from warning_generator import ObstacleWarningGenerator
    from obstacle_service import ObstacleService

if "obstacle_awareness" not in sys.modules:
    sys.modules["obstacle_awareness"] = sys.modules[__name__]

__all__ = [
    "ObstacleDetector",
    "ObstacleWarningGenerator",
    "ObstacleService",
]
