"""NAYRA — Public Obstacle Awareness Module Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "11_obstacle_awareness"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("11_obstacle_awareness")

ObstacleDetector = _pkg.ObstacleDetector
ObstacleWarningGenerator = _pkg.ObstacleWarningGenerator
ObstacleService = _pkg.ObstacleService

if "obstacle_awareness" not in sys.modules:
    sys.modules["obstacle_awareness"] = sys.modules[__name__]

__all__ = [
    "ObstacleDetector",
    "ObstacleWarningGenerator",
    "ObstacleService",
]
