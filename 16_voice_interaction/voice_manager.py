"""Voice Manager orchestrating priority queue, acoustic shield, and audio playback."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from shared.interfaces import IVoiceOutput
from shared.schemas import SpeechPriority, SpeechRequest

try:
    from .acoustic_shield import AcousticShield
    from .priority_speech_queue import PrioritySpeechQueue
except (ImportError, ValueError):
    from acoustic_shield import AcousticShield
    from priority_speech_queue import PrioritySpeechQueue

logger = logging.getLogger("NAYRA-VoiceManager")


class VoiceManager(IVoiceOutput):
    """System-wide voice manager with preemption, priority queuing, and feedback suppression."""

    def __init__(
        self,
        shield: Optional[AcousticShield] = None,
        queue: Optional[PrioritySpeechQueue] = None,
        tts_backend: Optional[Any] = None,
        playback_fn: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.shield = shield or AcousticShield(cooldown_sec=1.2)
        self.queue = queue or PrioritySpeechQueue()
        self.tts_backend = tts_backend
        self.playback_fn = playback_fn  # Optional mock or custom audio player function

        self._stop_event = threading.Event()
        self._preempt_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._current_request: Optional[SpeechRequest] = None
        self._state_lock = threading.RLock()
        self._is_active = False

    def initialize(self) -> bool:
        """Initialize and start background playback worker."""
        self.start()
        return True

    def start(self) -> None:
        """Start the background speech playback loop."""
        with self._state_lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._is_active = True
            self._worker_thread = threading.Thread(
                target=self._playback_loop,
                name="NAYRA-VoiceManagerWorker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("VoiceManager playback worker started.")

    def stop(self) -> None:
        """Halt playback and purge pending speech queue."""
        with self._state_lock:
            self.queue.clear()
            self._current_request = None
            if self.tts_backend and hasattr(self.tts_backend, "stop"):
                try:
                    self.tts_backend.stop()
                except Exception as e:
                    logger.error(f"Error stopping TTS backend: {e}")
            self.shield.on_speech_ended()
            logger.info("VoiceManager stopped and queue cleared.")

    def shutdown(self) -> None:
        """Stop worker thread completely."""
        with self._state_lock:
            self._stop_event.set()
            self.stop()
            self._is_active = False
            if self.tts_backend and hasattr(self.tts_backend, "shutdown"):
                try:
                    self.tts_backend.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down TTS backend: {e}")
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
            logger.info("VoiceManager worker thread terminated.")

    def is_speaking(self) -> bool:
        """Check if audio output is currently playing or shield is active."""
        if self.tts_backend and hasattr(self.tts_backend, "is_speaking"):
            try:
                if self.tts_backend.is_speaking():
                    return True
            except Exception:
                pass
        return self.shield.is_shield_active()

    def speak(
        self,
        text: str,
        language: str = "hi",
        priority: SpeechPriority = SpeechPriority.USER_QUERY_RESPONSE,
        block: bool = False,
        spoken_callback: Optional[Callable[[], None]] = None,
    ) -> bool:
        """Enqueue speech request with priority."""
        if not text or not text.strip():
            return False

        req = SpeechRequest(
            text=text.strip(),
            language=language,
            priority=priority,
            spoken_callback=spoken_callback,
        )

        should_preempt = self.queue.push(req)
        if should_preempt:
            # Interrupt lower priority speech if supported
            logger.info(f"Preempting current speech for {priority.name}")
            self._preempt_event.set()
            if self.tts_backend and hasattr(self.tts_backend, "stop"):
                try:
                    self.tts_backend.stop()
                except Exception as e:
                    logger.error(f"Error interrupting TTS backend on preemption: {e}")

        if block:
            # Wait until this specific request finishes
            done_evt = threading.Event()
            orig_cb = req.spoken_callback

            def _block_cb():
                if orig_cb:
                    orig_cb()
                done_evt.set()

            req.spoken_callback = _block_cb
            done_evt.wait(timeout=10.0)

        return True

    def _playback_loop(self) -> None:
        while not self._stop_event.is_set():
            req = self.queue.pop()
            if not req:
                time.sleep(0.05)
                continue

            self._preempt_event.clear()
            with self._state_lock:
                self._current_request = req

            self.shield.on_speech_started()
            try:
                self._execute_speech(req)
            except Exception as e:
                logger.error(f"Error during speech synthesis/playback: {e}")
            finally:
                self.shield.on_speech_ended()
                self.queue.mark_completed()
                with self._state_lock:
                    self._current_request = None

                if req.spoken_callback:
                    try:
                        req.spoken_callback()
                    except Exception as e:
                        logger.error(f"Error in spoken_callback: {e}")

    def _execute_speech(self, req: SpeechRequest) -> None:
        """Execute physical playback or custom hook."""
        logger.info(f"[SPEAKING - {req.priority.name}]: {req.text}")
        if self.playback_fn:
            self.playback_fn(req.text, req.language)
            return

        if self.tts_backend:
            # Map SpeechPriority to TTS priority level
            p = 2
            if req.priority in (SpeechPriority.CRITICAL_SAFETY, SpeechPriority.EMERGENCY_INTERACTION):
                p = 0
            elif req.priority == SpeechPriority.OBSTACLE_WARNING:
                p = 1

            if hasattr(self.tts_backend, "speak"):
                try:
                    self.tts_backend.speak(req.text, priority=p, interrupt=(p == 0))
                    # Wait while TTS backend is actively speaking to keep acoustic shield active
                    if hasattr(self.tts_backend, "is_speaking"):
                        time.sleep(0.05)
                        while self.tts_backend.is_speaking() and not self._stop_event.is_set():
                            if self._preempt_event.is_set():
                                logger.info(f"Speech interrupted mid-playback: {req.text[:30]}")
                                break
                            time.sleep(0.05)
                            time.sleep(0.03)
                    return
                except Exception as e:
                    logger.error(f"Error calling TTS backend speak: {e}")

        # Default fallback simulation: brief pause proportional to length (capped at 0.1s for tests)
        time.sleep(min(0.05, len(req.text) * 0.002))
