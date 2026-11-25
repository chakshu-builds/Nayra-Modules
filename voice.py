"""NAYRA — Public Voice Module Entrypoint.

Provides a standard Python identifier alias for '09_voice' to eliminate
numeric prefix import limitations:
    from voice import PhoneAudioReceiver, VoiceCommandRecognizer, CommandIntent
"""

from __future__ import annotations

import importlib
import sys

_voice_pkg = importlib.import_module("09_voice")

CommandIntent = _voice_pkg.CommandIntent
CommandMatcher = _voice_pkg.CommandMatcher
VoiceCommandRecognizer = _voice_pkg.VoiceCommandRecognizer
PhoneAudioReceiver = _voice_pkg.PhoneAudioReceiver
ReadAloudPipeline = getattr(_voice_pkg, "ReadAloudPipeline", None)

if "voice" not in sys.modules:
    sys.modules["voice"] = sys.modules[__name__]

__all__ = [
    "CommandIntent",
    "CommandMatcher",
    "VoiceCommandRecognizer",
    "PhoneAudioReceiver",
    "ReadAloudPipeline",
]
