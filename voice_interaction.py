"""NAYRA — Public Shared Voice Interaction Layer Entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_module_dir = Path(__file__).resolve().parent / "16_voice_interaction"
__path__ = [str(_module_dir)]

if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

_pkg = importlib.import_module("16_voice_interaction")

AcousticShield = _pkg.AcousticShield
PrioritySpeechQueue = _pkg.PrioritySpeechQueue
VoiceManager = _pkg.VoiceManager

if "voice_interaction" not in sys.modules:
    sys.modules["voice_interaction"] = sys.modules[__name__]

__all__ = [
    "AcousticShield",
    "PrioritySpeechQueue",
    "VoiceManager",
]
