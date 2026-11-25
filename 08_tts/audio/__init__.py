"""NAYRA Module 08 — Audio Output & Request Scheduling Package."""

try:
    from .audio_player import AudioPlayer, PlayerState
    from .audio_queue import AudioQueue
    from .network_audio_player import NetworkAudioPlayer
except (ImportError, ValueError):
    from audio.audio_player import AudioPlayer, PlayerState
    from audio.audio_queue import AudioQueue
    from audio.network_audio_player import NetworkAudioPlayer

__all__ = [
    "AudioPlayer",
    "PlayerState",
    "AudioQueue",
    "NetworkAudioPlayer",
]
