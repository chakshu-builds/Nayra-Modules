"""NAYRA Module 08 — Offline Multilingual Text-to-Speech (TTS) Subsystem.

Provides asynchronous, priority-preemptive speech synthesis and playback
for Hindi, English, and Hinglish using Kokoro-82M ONNX.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure 08_tts directory and root workspace are on sys.path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

try:
    from .audio.audio_player import AudioPlayer, PlayerState
    from .audio.audio_queue import AudioQueue
    from .audio.network_audio_player import NetworkAudioPlayer
    from .chunker import TextChunker
    from .engines.base import BaseTTSEngine, ModelNotFoundError, SynthesisError, TTSEngineError
    from .engines.kokoro_engine import KokoroEngine
    from .normalizer import TextNormalizer
    from .request_schema import PriorityLevel, RequestStatus, SpeechRequest
    from .tts_manager import TTSManager
except (ImportError, ValueError):
    from audio.audio_player import AudioPlayer, PlayerState
    from audio.audio_queue import AudioQueue
    from audio.network_audio_player import NetworkAudioPlayer
    from chunker import TextChunker
    from engines.base import BaseTTSEngine, ModelNotFoundError, SynthesisError, TTSEngineError
    from engines.kokoro_engine import KokoroEngine
    from normalizer import TextNormalizer
    from request_schema import PriorityLevel, RequestStatus, SpeechRequest
    from tts_manager import TTSManager

# Register 'tts' in sys.modules if not present
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
