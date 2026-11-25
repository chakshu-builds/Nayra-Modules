"""NAYRA Phase 3 — End-to-End Voice-Controlled Read Aloud Pipeline.

Integrates:
- Module 09 PhoneAudioReceiver (Android IP Webcam microphone stream /audio.wav)
- Module 09 VoiceCommandRecognizer (Whisper-tiny INT8 offline speech recognition)
- Module 07 OCR read_text (Unified Hindi + English DBNet + PP-OCRv5)
- Module 08 TTSManager (Kokoro-82M offline text-to-speech)
"""

from __future__ import annotations

import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

# Ensure root directory is on sys.path
_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

# Dynamic imports with fallbacks
try:
    from voice.audio_stream_receiver import PhoneAudioReceiver
    from voice.command_matcher import CommandIntent
    from voice.voice_command_recognizer import VoiceCommandRecognizer
except (ImportError, ValueError):
    from audio_stream_receiver import PhoneAudioReceiver
    from command_matcher import CommandIntent
    from voice_command_recognizer import VoiceCommandRecognizer

try:
    import ocr
    read_text = ocr.read_text
except (ImportError, AttributeError):
    import importlib
    _ocr_pkg = importlib.import_module("07_ocr")
    read_text = _ocr_pkg.read_text

try:
    import tts
    TTSManager = tts.TTSManager
    PriorityLevel = tts.PriorityLevel
except (ImportError, AttributeError):
    import importlib
    _tts_pkg = importlib.import_module("08_tts")
    TTSManager = _tts_pkg.TTSManager
    PriorityLevel = _tts_pkg.PriorityLevel

logger = logging.getLogger("NAYRA-ReadAloudPipeline")


class ReadAloudPipeline:
    """End-to-end voice-commanded assistive OCR reading pipeline.

    Guarantees:
    1. Fully asynchronous voice recognition: Audio stream processing and STT
       run on dedicated background threads and never freeze camera/vision loops.
    2. Zero redundant inference: A configurable debounce gate prevents duplicate
       OCR/TTS triggers from acoustic reverberation or repeated speech segments.
    3. Seamless interruption: "Stop reading" or "Band karo" immediately cancels
       active Kokoro TTS playback.
    4. Graceful fallbacks: Clear verbal notices when camera frames are unavailable
       or no printed text is detected in the captured scene.
    5. 100% offline local execution across Modules 07, 08, and 09.
    """

    def __init__(
        self,
        audio_receiver: Optional[PhoneAudioReceiver] = None,
        voice_recognizer: Optional[VoiceCommandRecognizer] = None,
        tts_manager: Optional[TTSManager] = None,
        frame_provider: Optional[Union[Callable[[], Optional[np.ndarray]], Any]] = None,
        ocr_fn: Optional[Callable[[np.ndarray], Any]] = None,
        preferred_ocr_provider: str = "CPU",
        phone_ip: Optional[str] = None,
        debounce_sec: float = 2.0,
        mock_mode: bool = False,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize ReadAloudPipeline.

        Args:
            audio_receiver: Existing PhoneAudioReceiver or None to auto-create.
            voice_recognizer: Existing VoiceCommandRecognizer or None to auto-create.
            tts_manager: Existing TTSManager or None to auto-create.
            frame_provider: Callable returning latest camera frame (or CameraManager instance).
            ocr_fn: Callable executing OCR on a frame (defaults to ocr.read_text).
            preferred_ocr_provider: Preferred ORT provider for OCR ('CPU' or 'CUDA').
            phone_ip: Optional IP of Android device (used if creating default receiver).
            debounce_sec: Minimum seconds between consecutive READ triggers.
            mock_mode: If True, uses mock audio receiver and mock audio player for tests.
            on_event: Optional callback for pipeline telemetry and lifecycle events.
        """
        self.mock_mode = mock_mode
        self.debounce_sec = max(0.5, float(debounce_sec))
        self.on_event = on_event
        self.ocr_fn = ocr_fn or read_text
        self.preferred_ocr_provider = preferred_ocr_provider

        # 1. Voice Recognizer
        self.voice_recognizer = voice_recognizer or VoiceCommandRecognizer()

        # 2. Audio Receiver
        if audio_receiver is not None:
            self.audio_receiver = audio_receiver
        else:
            stream_url = f"http://{phone_ip}:8080/audio.wav" if phone_ip else PhoneAudioReceiver.DEFAULT_STREAM_URL
            self.audio_receiver = PhoneAudioReceiver(
                stream_url=stream_url,
                mock_mode=mock_mode,
            )

        # Wire audio receiver callback
        self.audio_receiver.on_speech_utterance = self._on_speech_utterance_received

        # 3. TTS Manager
        self.tts_manager = tts_manager or TTSManager(mock_audio=mock_mode)

        # 4. Frame Provider
        self._frame_provider = frame_provider

        # Worker synchronization
        self._command_queue: queue.Queue[Tuple[np.ndarray, int]] = queue.Queue(maxsize=20)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # State tracking & debouncing
        self._lock = threading.Lock()
        self._is_running = False
        self._last_read_timestamp: float = 0.0
        self._last_intent: CommandIntent = CommandIntent.UNKNOWN
        self._last_transcription: str = ""
        self._total_utterances: int = 0
        self._total_reads: int = 0
        self._total_stops: int = 0

    def set_frame_provider(self, provider: Union[Callable[[], Optional[np.ndarray]], Any]) -> None:
        """Set or update the active frame provider."""
        self._frame_provider = provider

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Acquire latest frame from registered frame provider."""
        if self._frame_provider is None:
            return None

        # Check if provider is callable
        if callable(self._frame_provider):
            try:
                return self._frame_provider()
            except Exception as exc:
                logger.error("Error executing frame provider callback: %s", exc)
                return None

        # Check if provider has get_frame() (CameraManager protocol)
        if hasattr(self._frame_provider, "get_frame"):
            try:
                return self._frame_provider.get_frame(timeout=0.5)
            except Exception as exc:
                logger.error("Error calling get_frame() on camera manager: %s", exc)
                return None

        # Check if provider has get_latest_frame()
        if hasattr(self._frame_provider, "get_latest_frame"):
            try:
                res = self._frame_provider.get_latest_frame(timeout=0.5)
                if isinstance(res, tuple):
                    return res[0]
                return res
            except Exception as exc:
                logger.error("Error calling get_latest_frame() on frame reader: %s", exc)
                return None

        return None

    def start(self) -> bool:
        """Start all subsystem threads and the command dispatcher loop."""
        if self._is_running:
            return True

        logger.info("Starting NAYRA Read Aloud Pipeline...")
        self._stop_event.clear()

        # 1. Initialize Whisper STT model
        self.voice_recognizer.initialize()

        # 2. Start Command Dispatcher Worker Thread
        self._worker_thread = threading.Thread(
            target=self._command_dispatcher_loop,
            name="NAYRA-ReadAloudWorker",
            daemon=True,
        )
        self._worker_thread.start()

        # 3. Start Audio Receiver
        self.audio_receiver.start()

        self._is_running = True
        self._emit_event("pipeline_started", {"timestamp": time.time()})
        logger.info("NAYRA Read Aloud Pipeline active and listening.")
        return True

    def stop(self) -> None:
        """Cleanly stop listening and shut down worker threads."""
        if not self._is_running:
            return

        logger.info("Stopping NAYRA Read Aloud Pipeline...")
        self._is_running = False
        self._stop_event.set()

        # Stop audio receiver
        if self.audio_receiver:
            self.audio_receiver.stop()

        # Stop active TTS playback
        if self.tts_manager:
            self.tts_manager.stop()

        # Wait for worker thread to exit
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)

        self._emit_event("pipeline_stopped", {"timestamp": time.time()})
        logger.info("NAYRA Read Aloud Pipeline stopped.")

    def close(self) -> None:
        """Full resource cleanup including TTSManager shutdown."""
        self.stop()
        if self.tts_manager:
            self.tts_manager.shutdown()

    def __enter__(self) -> "ReadAloudPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _on_speech_utterance_received(self, samples: np.ndarray, sample_rate: int) -> None:
        """Callback invoked by PhoneAudioReceiver when an utterance ends."""
        if not self._is_running or self._stop_event.is_set():
            return

        with self._lock:
            self._total_utterances += 1

        try:
            self._command_queue.put_nowait((samples, sample_rate))
        except queue.Full:
            logger.warning("Command queue full; dropping speech utterance.")

    def _command_dispatcher_loop(self) -> None:
        """Dedicated background loop that transcribes speech and coordinates OCR/TTS."""
        logger.debug("ReadAloud command dispatcher thread running.")
        while not self._stop_event.is_set():
            try:
                item = self._command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            samples, sample_rate = item
            try:
                self._process_utterance(samples, sample_rate)
            except Exception as exc:
                logger.error("Error processing speech utterance: %s", exc, exc_info=True)
            finally:
                self._command_queue.task_done()

    def _process_utterance(self, samples: np.ndarray, sample_rate: int) -> None:
        """Run STT on utterance and dispatch command."""
        intent, transcript, lat_ms = self.voice_recognizer.recognize(samples, sample_rate)

        with self._lock:
            self._last_intent = intent
            self._last_transcription = transcript

        self._emit_event(
            "speech_recognized",
            {
                "intent": intent.value,
                "transcript": transcript,
                "latency_ms": lat_ms,
            },
        )

        logger.info(
            "Recognized voice utterance -> Intent: %s | Transcript: '%s' (%.1f ms)",
            intent.value,
            transcript,
            lat_ms,
        )

        if intent == CommandIntent.STOP:
            self._handle_stop_command()
        elif intent == CommandIntent.READ:
            self._handle_read_command(transcript)
        else:
            logger.debug("Utterance did not match a NAYRA command intent.")

    def _handle_stop_command(self) -> None:
        """Execute STOP intent: immediately halt active TTS playback."""
        with self._lock:
            self._total_stops += 1

        logger.info("STOP intent: Halting active TTS playback immediately.")
        self.tts_manager.stop()
        self._emit_event("command_stop_executed", {"timestamp": time.time()})

    def _handle_read_command(self, transcript: str) -> None:
        """Execute READ intent: debounce, capture frame, run OCR, and speak."""
        now = time.time()
        with self._lock:
            elapsed = now - self._last_read_timestamp
            if elapsed < self.debounce_sec:
                logger.info(
                    "Debounce active: ignoring duplicate READ command (elapsed=%.2fs < %.2fs).",
                    elapsed,
                    self.debounce_sec,
                )
                self._emit_event("command_read_debounced", {"elapsed_sec": elapsed})
                return

            self._last_read_timestamp = now
            self._total_reads += 1

        logger.info("READ intent accepted. Acquiring camera frame...")
        self._emit_event("command_read_started", {"timestamp": now})

        frame = self.get_latest_frame()
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Camera frame unavailable for OCR.")
            is_hindi = self._is_hindi_prompt(transcript)
            msg = "कैमरा उपलब्ध नहीं है।" if is_hindi else "Camera frame not available."
            lang_hint = "hi" if is_hindi else "en-us"
            self.tts_manager.speak(msg, priority=PriorityLevel.NORMAL, source="ocr", language_hint=lang_hint)
            self._emit_event("read_failed_no_frame", {})
            return

        # Run OCR
        t0_ocr = time.perf_counter()
        try:
            ocr_result = self.ocr_fn(frame, preferred_provider=self.preferred_ocr_provider)
        except TypeError:
            ocr_result = self.ocr_fn(frame)
        ocr_elapsed_ms = (time.perf_counter() - t0_ocr) * 1000.0

        text_found = bool(
            ocr_result.success
            and ocr_result.combined_text
            and ocr_result.combined_text.strip()
        )

        self._emit_event(
            "ocr_completed",
            {
                "success": ocr_result.success,
                "text_found": text_found,
                "char_count": len(ocr_result.combined_text) if text_found else 0,
                "primary_language": getattr(ocr_result.primary_language, "value", str(ocr_result.primary_language)),
                "ocr_latency_ms": ocr_elapsed_ms,
                "execution_providers": getattr(ocr_result, "execution_providers", {}),
            },
        )

        if text_found:
            text_to_speak = ocr_result.combined_text.strip()
            # Determine appropriate language routing
            prim_lang = getattr(ocr_result.primary_language, "value", str(ocr_result.primary_language)).upper()
            if "HINDI" in prim_lang:
                lang_hint = "hi"
            elif "ENGLISH" in prim_lang:
                lang_hint = "en-us"
            else:
                lang_hint = "auto"

            logger.info(
                "OCR successful (%d chars, lang=%s in %.1f ms). Enqueueing TTS: '%s'...",
                len(text_to_speak),
                lang_hint,
                ocr_elapsed_ms,
                text_to_speak[:60].replace("\n", " "),
            )

            req_id = self.tts_manager.speak(
                text=text_to_speak,
                priority=PriorityLevel.NORMAL,
                source="ocr",
                language_hint=lang_hint,
            )
            self._emit_event("tts_enqueued", {"request_id": req_id, "text": text_to_speak[:80]})

        else:
            # Fallback when no text detected in frame
            logger.info("OCR completed in %.1f ms: No text detected in scene.", ocr_elapsed_ms)
            is_hindi = self._is_hindi_prompt(transcript)
            msg = "कोई लिखावट नहीं मिली।" if is_hindi else "No text detected."
            lang_hint = "hi" if is_hindi else "en-us"
            self.tts_manager.speak(
                text=msg,
                priority=PriorityLevel.NORMAL,
                source="ocr",
                language_hint=lang_hint,
            )
            self._emit_event("tts_no_text_fallback", {"message": msg})

    @staticmethod
    def _is_hindi_prompt(transcript: str) -> bool:
        """Check if user prompt is Hindi or Hinglish."""
        t = transcript.lower()
        if any(c in transcript for c in "पढ़ोमेरेलिएयहबंद"):
            return True
        if any(w in t for w in ("mere", "liye", "padho", "parho", "band", "karo", "rok")):
            return True
        return False

    def _emit_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Emit telemetry event to registered listener."""
        if self.on_event:
            try:
                self.on_event(event_name, data)
            except Exception as exc:
                logger.warning("Telemetry event callback error: %s", exc)

    def trigger_mock_command(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> None:
        """Inject an audio sample directly (for unit testing and simulation)."""
        self.audio_receiver.inject_audio_sample(audio_samples, sample_rate)

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status of the entire Read Aloud pipeline."""
        with self._lock:
            return {
                "is_running": self._is_running,
                "audio_connected": self.audio_receiver.is_connected,
                "audio_last_error": self.audio_receiver.last_error,
                "last_intent": self._last_intent.value,
                "last_transcription": self._last_transcription,
                "total_utterances": self._total_utterances,
                "total_reads": self._total_reads,
                "total_stops": self._total_stops,
                "queue_size": self._command_queue.qsize(),
                "tts_status": self.tts_manager.get_status() if self.tts_manager else None,
            }
