"""NAYRA — Public Assistant Orchestrator Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "17_assistant_orchestrator"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("17_assistant_orchestrator")

IntentClassifier = _pkg.IntentClassifier
AssistantOrchestrator = _pkg.AssistantOrchestrator

if "assistant_orchestrator" not in sys.modules:
    sys.modules["assistant_orchestrator"] = sys.modules[__name__]

__all__ = [
    "IntentClassifier",
    "AssistantOrchestrator",
]
