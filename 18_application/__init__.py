"""Module 18: Unified Application."""

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
    from .app_config import AppConfig
    from .app import NayraApplication
except (ImportError, ValueError):
    from app_config import AppConfig
    from app import NayraApplication

__all__ = [
    "AppConfig",
    "NayraApplication",
]
