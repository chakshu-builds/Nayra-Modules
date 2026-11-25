"""NAYRA Module 08 — Kokoro-82M ONNX Speech Synthesis Engine.

Wraps Kokoro-82M ONNX model (StyleTTS 2 / ISTFTNet) for high-performance offline
multilingual synthesis on CPU (optimized default) or GPU (optional CUDA).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort

try:
    from .base import BaseTTSEngine, ModelNotFoundError, SynthesisError
except (ImportError, ValueError):
    from engines.base import BaseTTSEngine, ModelNotFoundError, SynthesisError

logger = logging.getLogger("NAYRA-KokoroEngine")


class KokoroEngine(BaseTTSEngine):
    """Offline neural TTS engine powered by Kokoro-82M ONNX.

    Attributes:
        model_path: Absolute or relative path to 'kokoro-v1.0.onnx'.
        voices_path: Absolute or relative path to 'voices-v1.0.bin'.
        use_gpu: If True, attempts to run via CUDAExecutionProvider.
        num_threads: Intra-op thread count for CPU execution (default: min(cores, 8)).
    """

    SAMPLE_RATE: int = 24000
    DEFAULT_ENGLISH_VOICE: str = "af_bella"
    DEFAULT_HINDI_VOICE: str = "hf_alpha"

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        voices_path: Optional[str | Path] = None,
        use_gpu: bool = False,
        num_threads: Optional[int] = None,
    ) -> None:
        # Locate project root dynamically
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.model_path = Path(model_path) if model_path else (root_dir / "models" / "tts" / "kokoro" / "kokoro-v1.0.onnx")
        self.voices_path = Path(voices_path) if voices_path else (root_dir / "models" / "tts" / "kokoro" / "voices-v1.0.bin")

        self.use_gpu = use_gpu
        cpu_count = os.cpu_count() or 4
        self.num_threads = num_threads or min(cpu_count, 8)

        self._session: Optional[ort.InferenceSession] = None
        self._kokoro = None
        self._available_voices: List[str] = []
        self._active_provider: str = "Uninitialized"
        self._initialized: bool = False

    def initialize(self) -> None:
        """Initialize the ONNX runtime session and load Kokoro model and voices."""
        if self._initialized:
            logger.debug("KokoroEngine is already initialized.")
            return

        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Kokoro model weights not found at: {self.model_path.resolve()}"
            )
        if not self.voices_path.exists():
            raise ModelNotFoundError(
                f"Kokoro voice vectors not found at: {self.voices_path.resolve()}"
            )

        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise SynthesisError(
                "kokoro-onnx package is not installed. Run 'pip install kokoro-onnx soundfile'."
            ) from exc

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ["CPUExecutionProvider"]
        if self.use_gpu:
            # Enable PyTorch CUDA libraries on Windows if available
            try:
                import torch
                torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
                if os.path.exists(torch_lib):
                    os.add_dll_directory(torch_lib)
            except Exception:
                pass

            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("Configuring KokoroEngine with CUDAExecutionProvider.")
            else:
                logger.warning(
                    "CUDAExecutionProvider requested but not available in ONNX Runtime. "
                    "Falling back to CPUExecutionProvider."
                )

        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )
            self._active_provider = self._session.get_providers()[0]

            self._kokoro = Kokoro.from_session(self._session, str(self.voices_path))
            self._available_voices = self._kokoro.get_voices()
            self._initialized = True

            logger.info(
                "KokoroEngine initialized successfully on %s (voices: %d).",
                self._active_provider,
                len(self._available_voices),
            )
        except Exception as exc:
            self.shutdown()
            raise SynthesisError(f"Failed to initialize Kokoro ONNX session: {exc}") from exc

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        lang: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize text into raw float32 audio samples.

        Args:
            text: Text to synthesize.
            voice: Voice ID (defaults to 'hf_alpha' for hi, 'af_bella' for en-us).
            lang: Language code ('hi' or 'en-us').
            speed: Speech rate multiplier in [0.5, 2.0].

        Returns:
            Tuple of (audio_samples, 24000).
        """
        if not self._initialized or self._kokoro is None:
            self.initialize()

        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        selected_lang = lang or ("hi" if voice and voice.startswith("h") else "en-us")
        selected_voice = voice or (
            self.DEFAULT_HINDI_VOICE if selected_lang == "hi" else self.DEFAULT_ENGLISH_VOICE
        )

        if selected_voice not in self._available_voices:
            raise ValueError(
                f"Requested voice '{selected_voice}' not found in available voices. "
                f"Available Hindi: ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi']. "
                f"Available English: ['af_bella', 'af_sarah', 'am_adam', 'am_michael']."
            )

        try:
            samples, sr = self._kokoro.create(
                text=text.strip(),
                voice=selected_voice,
                speed=float(speed),
                lang=selected_lang,
            )
            # Ensure float32 normalized output
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)

            return samples, sr
        except Exception as exc:
            logger.error("Synthesis failed for text '%s' (voice=%s, lang=%s): %s", text[:40], selected_voice, selected_lang, exc)
            raise SynthesisError(
                f"Kokoro synthesis failed for '{text[:40]}...' (voice={selected_voice}, lang={selected_lang}): {exc}"
            ) from exc

    def get_sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def get_available_voices(self) -> List[str]:
        return list(self._available_voices)

    def is_ready(self) -> bool:
        return self._initialized and self._kokoro is not None

    @property
    def active_provider(self) -> str:
        return self._active_provider

    def shutdown(self) -> None:
        """Cleanly release model and ONNX runtime session."""
        self._kokoro = None
        self._session = None
        self._available_voices = []
        self._initialized = False
        self._active_provider = "Uninitialized"
        logger.info("KokoroEngine shutdown completed.")
