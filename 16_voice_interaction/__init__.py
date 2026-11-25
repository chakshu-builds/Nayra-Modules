"""Module 16: Shared Voice Interaction Layer."""

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
    from .acoustic_shield import AcousticShield
    from .priority_speech_queue import PrioritySpeechQueue
    from .voice_manager import VoiceManager
except (ImportError, ValueError):
    from acoustic_shield import AcousticShield
    from priority_speech_queue import PrioritySpeechQueue
    from voice_manager import VoiceManager

if "voice_interaction" not in sys.modules:
    sys.modules["voice_interaction"] = sys.modules[__name__]

__all__ = [
    "AcousticShield",
    "PrioritySpeechQueue",
    "VoiceManager",
]
