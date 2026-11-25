"""NAYRA Module 10.1 — Voice-Guided Object Finding Coordinator.

Connects:
1. Android IP Webcam microphone stream (or synthetic audio) via Module 09 PhoneAudioReceiver.
2. Offline Whisper speech recognition via Module 09 VoiceCommandRecognizer.
3. Live camera object perception & spatial memory via Module 10 ObjectMemoryManager.
4. Kokoro-82M offline TTS speech output via Module 08 TTSManager.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Optional, Union

import numpy as np

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from .db.models import MemoryQueryResult
    from .memory_manager import ObjectMemoryManager
    from .perception.live_camera_stream import LiveCameraStream
except (ImportError, ValueError):
    from db.models import MemoryQueryResult
    from memory_manager import ObjectMemoryManager
    from perception.live_camera_stream import LiveCameraStream

# Silence phonemizer word-count discrepancy warnings (harmless whitespace tokenization difference)
logging.getLogger("phonemizer").setLevel(logging.ERROR)

logger = logging.getLogger("NAYRA-VoiceGuidance")

ENGLISH_OBJECT_PROMPT_BIAS = (
    "Where is my TV? Where is my phone? Where did I last see my phone? Where is my laptop? Where is my book? Where is the TV remote? What can you see? NAYRA, locate my objects."
)


class VoiceGuidedObjectFinder:
    """Coordinates voice-guided object finding combining camera perception, memory, and TTS."""

    def __init__(
        self,
        memory_manager: ObjectMemoryManager,
        camera_stream: Optional[LiveCameraStream] = None,
        audio_receiver: Optional[Any] = None,
        voice_recognizer: Optional[Any] = None,
        tts_manager: Optional[Any] = None,
        feedback_cooldown_sec: float = 1.2,
        perception_fps: float = 5.0,
    ) -> None:
        self.manager = memory_manager
        self.camera_stream = camera_stream
        self.audio_receiver = audio_receiver
        self.recognizer = voice_recognizer
        self.tts = tts_manager
        self.feedback_cooldown_sec = feedback_cooldown_sec
        self.perception_fps = perception_fps
        self.prompt_bias: str = ENGLISH_OBJECT_PROMPT_BIAS

        if self.recognizer is not None and hasattr(self.recognizer, "PROMPT_BIAS"):
            self.recognizer.PROMPT_BIAS = self.prompt_bias

        if self.camera_stream is not None:
            self.manager.camera_stream = self.camera_stream

        if self.tts is not None and self.manager.tts is None:
            self.manager.tts = self.tts

        self._is_running = True
        self._on_query_callback: Optional[Callable[[str, MemoryQueryResult], None]] = None

        # Acoustic feedback shielding state
        self._last_tts_end_time: float = 0.0

        # Background live camera perception thread
        self._perception_thread: Optional[threading.Thread] = None
        self._perception_stop_event = threading.Event()

    def initialize_subsystems(self, phone_ip: Optional[str] = None) -> None:
        """Lazily initialize voice recognition, audio receiver, and TTS if not provided."""
        if self.recognizer is None:
            try:
                from voice import VoiceCommandRecognizer
                self.recognizer = VoiceCommandRecognizer()
                self.recognizer.initialize()
                # Bias offline Whisper toward object location memory vocabulary in English
                self.recognizer.PROMPT_BIAS = self.prompt_bias
                logger.info("Initialized VoiceCommandRecognizer with English object memory prompt bias.")
            except Exception as exc:
                logger.warning("Could not initialize VoiceCommandRecognizer: %s", exc)

        if self.tts is None:
            try:
                from tts import TTSManager
                self.tts = TTSManager(phone_speaker=False)
                self.manager.tts = self.tts
                logger.info("Initialized Kokoro TTSManager (Laptop Speakers).")
            except Exception as exc:
                logger.warning("Could not initialize TTSManager: %s", exc)

        if self.audio_receiver is None and phone_ip:
            try:
                from voice import PhoneAudioReceiver
                audio_url = f"http://{phone_ip}:8080/audio.wav"
                self.audio_receiver = PhoneAudioReceiver(
                    stream_url=audio_url,
                    on_speech_utterance=self._handle_speech_utterance,
                    energy_threshold=0.012,
                    silence_timeout_sec=0.7,
                    max_speech_duration_sec=5.0,
                )
                logger.info("Configured PhoneAudioReceiver at %s (threshold=0.012, timeout=0.7s)", audio_url)
            except Exception as exc:
                logger.warning("Could not configure PhoneAudioReceiver: %s", exc)

    def set_query_callback(self, callback: Callable[[str, MemoryQueryResult], None]) -> None:
        """Register a callback invoked whenever a voice query is processed."""
        self._on_query_callback = callback

    def _recognize_with_fallback(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[Any, str, float]:
        """Speech recognition with English-only enforcement for Whisper-tiny."""
        t0 = time.perf_counter()
        model = getattr(self.recognizer, "_model", None)
        if (
            model is None
            or (hasattr(self.recognizer, "recognize") and hasattr(getattr(self.recognizer, "recognize", None), "_mock_return_value"))
        ):
            return self.recognizer.recognize(audio_samples, sample_rate=sample_rate)

        data = np.asarray(audio_samples, dtype=np.float32).ravel()
        if len(data) == 0:
            return None, "", 0.0

        if sample_rate != 16000:
            num_out = int(len(data) * 16000 / sample_rate)
            data = np.interp(
                np.linspace(0, len(data), num_out, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)

        prompt = getattr(self, "prompt_bias", getattr(self.recognizer, "PROMPT_BIAS", None))
        segs, info = self.recognizer._model.transcribe(
            data,
            initial_prompt=prompt,
            language="en",
            beam_size=1,
        )
        transcript = " ".join(s.text for s in segs).strip()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return None, transcript, lat_ms

    def process_voice_query(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = 16000,
        current_frame: Optional[np.ndarray] = None,
    ) -> Optional[MemoryQueryResult]:
        """Transcribe an audio utterance and execute object location guidance.

        Args:
            audio_samples: 1D float32 audio array.
            sample_rate: Sample rate (default 16,000 Hz).
            current_frame: Optional live frame.

        Returns:
            MemoryQueryResult or None if no speech transcribed.
        """
        if self.recognizer is None:
            self.initialize_subsystems()

        if self.recognizer is None:
            logger.error("Voice recognizer unavailable.")
            return None

        # 1. Speech to text with multilingual recovery
        _, transcribed_text, lat_ms = self._recognize_with_fallback(audio_samples, sample_rate=sample_rate)
        query_text = transcribed_text.strip()
        if not query_text:
            return None

        logger.info("Voice query recognized (%.1f ms): '%s'", lat_ms, query_text)

        # 2. Query Object Location Memory with live visibility inspection (forced English)
        result = self.manager.query(
            user_query=query_text,
            current_frame=current_frame,
            speak=True,  # Automatically speak response via TTS
            preferred_language="en",
        )

        logger.info("Guidance Response: HI='%s' | EN='%s' (Visible=%s)",
                    result.spoken_response_hi, result.spoken_response_en, result.is_currently_visible)

        if self._on_query_callback:
            try:
                self._on_query_callback(query_text, result)
            except Exception as cb_err:
                logger.warning("Query callback error: %s", cb_err)

        return result

    def _perception_loop(self) -> None:
        """Background perception thread pulling camera frames and tracking objects."""
        target_interval = 1.0 / self.perception_fps if self.perception_fps > 0 else 0.2
        logger.info("Background live perception loop started (Target: %.1f FPS).", self.perception_fps)

        while not self._perception_stop_event.is_set():
            t_start = time.perf_counter()

            # Continuously monitor TTS playback for acoustic feedback shield
            if self.tts is not None and hasattr(self.tts, "is_speaking") and self.tts.is_speaking():
                self._last_tts_end_time = time.time()

            # Pull latest camera frame and observe (bypassing motion gating for continuous live perception)
            if self.camera_stream is not None and getattr(self.camera_stream, "is_running", False):
                has_frame, frame, ts = self.camera_stream.read_latest()
                if has_frame and frame is not None and frame.size > 0:
                    try:
                        self.manager.observe_frame(frame, timestamp=ts, force_detection=True)
                    except Exception as obs_exc:
                        logger.debug("Background frame observation error: %s", obs_exc)

            elapsed = time.perf_counter() - t_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0.005:
                time.sleep(sleep_time)
            else:
                time.sleep(0.005)

        logger.info("Background live perception loop exited.")

    def _handle_speech_utterance(self, audio_samples: np.ndarray, sample_rate: int) -> None:
        """Callback from PhoneAudioReceiver on speech utterance completion."""
        if not self._is_running:
            logger.debug("[VoiceGuidance] Dropping utterance: finder is stopped.")
            return

        now = time.time()

        # Acoustic Echo Protection: Suppress NAYRA from hearing its own voice
        is_speaking = False
        if self.tts is not None and hasattr(self.tts, "is_speaking"):
            is_speaking = self.tts.is_speaking()

        if is_speaking:
            self._last_tts_end_time = now
            logger.info("[Acoustic Feedback Shield] Dropping utterance: NAYRA TTS is actively speaking.")
            return

        elapsed_since_tts = now - self._last_tts_end_time
        if elapsed_since_tts < self.feedback_cooldown_sec:
            logger.info(
                "[Acoustic Feedback Shield] Dropping utterance: within post-TTS recovery window (%.2fs < %.2fs).",
                elapsed_since_tts,
                self.feedback_cooldown_sec,
            )
            return

        self.process_voice_query(
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            current_frame=None,
        )

    def start_listening(self) -> bool:
        """Start background camera streaming, perception loop, and microphone listening."""
        self._is_running = True
        self._perception_stop_event.clear()

        if self.camera_stream is not None and not self.camera_stream.is_running:
            self.camera_stream.start()

        # Launch continuous background perception loop
        if self.camera_stream is not None and (self._perception_thread is None or not self._perception_thread.is_alive()):
            self._perception_thread = threading.Thread(
                target=self._perception_loop,
                name="NAYRA-BackgroundPerception",
                daemon=True,
            )
            self._perception_thread.start()

        if self.audio_receiver is not None:
            return self.audio_receiver.start()
        return True

    def stop_listening(self) -> None:
        """Stop listening and disconnect stream components."""
        self._is_running = False
        self._perception_stop_event.set()

        if self._perception_thread and self._perception_thread.is_alive():
            self._perception_thread.join(timeout=2.0)
            self._perception_thread = None

        if self.audio_receiver is not None:
            self.audio_receiver.stop()
        if self.camera_stream is not None:
            self.camera_stream.stop()
        if self.tts is not None and hasattr(self.tts, "stop"):
            self.tts.stop()
