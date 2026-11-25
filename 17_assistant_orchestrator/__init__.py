"""Module 17: Assistant Orchestrator."""

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
    from .intent_router import IntentClassifier
    from .orchestrator import AssistantOrchestrator
except (ImportError, ValueError):
    from intent_router import IntentClassifier
    from orchestrator import AssistantOrchestrator

if "assistant_orchestrator" not in sys.modules:
    sys.modules["assistant_orchestrator"] = sys.modules[__name__]

__all__ = [
    "IntentClassifier",
    "AssistantOrchestrator",
]
