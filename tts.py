"""NAYRA — Public TTS Module Entrypoint.

Provides a standard Python identifier alias for '08_tts' to eliminate
numeric prefix import limitations:
    from tts import TTSManager, SpeechRequest, PriorityLevel, RequestStatus
"""

from __future__ import annotations

import importlib
import sys

# Import the actual module implementation
_tts_pkg = importlib.import_module("08_tts")

# Forward all public API functions and classes
TTSManager = _tts_pkg.TTSManager
SpeechRequest = _tts_pkg.SpeechRequest
PriorityLevel = _tts_pkg.PriorityLevel
RequestStatus = _tts_pkg.RequestStatus
TextNormalizer = _tts_pkg.TextNormalizer
TextChunker = _tts_pkg.TextChunker
KokoroEngine = _tts_pkg.KokoroEngine
BaseTTSEngine = _tts_pkg.BaseTTSEngine
TTSEngineError = _tts_pkg.TTSEngineError
ModelNotFoundError = _tts_pkg.ModelNotFoundError
SynthesisError = _tts_pkg.SynthesisError
AudioPlayer = _tts_pkg.AudioPlayer
PlayerState = _tts_pkg.PlayerState
AudioQueue = _tts_pkg.AudioQueue
NetworkAudioPlayer = _tts_pkg.NetworkAudioPlayer

# Ensure sys.modules has 'tts' pointing to this module
if "tts" not in sys.modules:
    sys.modules["tts"] = sys.modules[__name__]

__all__ = [
    "TTSManager",
    "SpeechRequest",
    "PriorityLevel",
    "RequestStatus",
    "TextNormalizer",
    "TextChunker",
    "KokoroEngine",
    "BaseTTSEngine",
    "TTSEngineError",
    "ModelNotFoundError",
    "SynthesisError",
    "AudioPlayer",
    "PlayerState",
    "AudioQueue",
    "NetworkAudioPlayer",
]
