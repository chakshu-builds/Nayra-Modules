"""NAYRA Module 09 — Phone Microphone Input & Voice Command Recognition.

Provides live microphone audio streaming from Android IP Webcam and
offline multilingual speech-to-intent recognition using faster-whisper.
"""

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
    from .audio_stream_receiver import PhoneAudioReceiver
    from .command_matcher import CommandIntent, CommandMatcher
    from .voice_command_recognizer import VoiceCommandRecognizer
    from .read_aloud_pipeline import ReadAloudPipeline
except (ImportError, ValueError):
    from audio_stream_receiver import PhoneAudioReceiver
    from command_matcher import CommandIntent, CommandMatcher
    from voice_command_recognizer import VoiceCommandRecognizer
    from read_aloud_pipeline import ReadAloudPipeline

if "voice" not in sys.modules:
    sys.modules["voice"] = sys.modules[__name__]

__all__ = [
    "CommandIntent",
    "CommandMatcher",
    "VoiceCommandRecognizer",
    "PhoneAudioReceiver",
    "ReadAloudPipeline",
]
