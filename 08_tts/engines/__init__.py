"""NAYRA Module 08 — TTS Synthesis Engines."""

try:
    from .base import BaseTTSEngine, ModelNotFoundError, SynthesisError, TTSEngineError
    from .kokoro_engine import KokoroEngine
except (ImportError, ValueError):
    from engines.base import BaseTTSEngine, ModelNotFoundError, SynthesisError, TTSEngineError
    from engines.kokoro_engine import KokoroEngine

__all__ = [
    "BaseTTSEngine",
    "KokoroEngine",
    "TTSEngineError",
    "ModelNotFoundError",
    "SynthesisError",
]
