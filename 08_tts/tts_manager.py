"""NAYRA Module 08 — Central Text-to-Speech Manager.

Coordinates speech requests, conservative normalization, intelligent chunking,
neural Kokoro-82M synthesis, priority scheduling, and non-blocking playback.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

try:
    from .audio.audio_player import AudioPlayer, PlayerState
    from .audio.audio_queue import AudioQueue
    from .audio.network_audio_player import NetworkAudioPlayer
    from .chunker import TextChunker
    from .engines.base import BaseTTSEngine
    from .engines.kokoro_engine import KokoroEngine
    from .normalizer import TextNormalizer
    from .request_schema import PriorityLevel, RequestStatus, SpeechRequest
except (ImportError, ValueError):
    from audio.audio_player import AudioPlayer, PlayerState
    from audio.audio_queue import AudioQueue
    from audio.network_audio_player import NetworkAudioPlayer
    from chunker import TextChunker
    from engines.base import BaseTTSEngine
    from engines.kokoro_engine import KokoroEngine
    from normalizer import TextNormalizer
    from request_schema import PriorityLevel, RequestStatus, SpeechRequest

logger = logging.getLogger("NAYRA-TTSManager")


class TTSManager:
    """Asynchronous, priority-managed Text-to-Speech coordinator for NAYRA.

    Guarantees:
    1. Fully non-blocking speak() API returning unique tracking request IDs.
    2. Strict priority preemption: Critical safety alerts (P0) preempt active OCR reading (P2).
    3. Low Time-to-First-Audio (TTFA) via pipelined sentence chunking.
    4. Conservative text normalization preserving medical dosages, currency, and danda.
    5. Clean lifecycle management with safe thread shutdown and error recovery.
    """

    def __init__(
        self,
        engine: Optional[BaseTTSEngine] = None,
        player: Optional[AudioPlayer] = None,
        max_queue_size: int = 50,
        duplicate_window_sec: float = 3.0,
        max_chunk_chars: int = 150,
        mock_audio: bool = False,
        phone_speaker: bool = False,
        phone_audio_port: int = 8088,
    ) -> None:
        self.engine = engine or KokoroEngine()
        if player is not None:
            self.player = player
        elif phone_speaker:
            self.player = NetworkAudioPlayer(
                port=phone_audio_port,
                mock_mode=mock_audio,
                on_user_stop_callback=self.stop,
            )
        else:
            self.player = AudioPlayer(mock_mode=mock_audio)
        self.normalizer = TextNormalizer()
        self.chunker = TextChunker(max_chunk_chars=max_chunk_chars)
        self.queue = AudioQueue(max_size=max_queue_size, duplicate_window_sec=duplicate_window_sec)

        # Worker thread controls
        self._lock = threading.RLock()
        self._stop_worker_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Unpaused initially

        self._cancel_current_event = threading.Event()
        self._chunk_finished_event = threading.Event()

        # Active request tracking
        self._active_request: Optional[SpeechRequest] = None
        self._is_synthesizing: bool = False

        # Start background worker thread
        self._worker_thread = threading.Thread(
            target=self._process_queue_loop,
            name="NAYRA-TTSWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("TTSManager initialized and background worker started.")

    def speak(
        self,
        text: str,
        priority: int = PriorityLevel.NORMAL,
        source: str = "manual",
        interrupt: bool = False,
        language_hint: Optional[str] = None,
    ) -> str:
        """Enqueue speech request asynchronously without blocking the caller.

        Args:
            text: Input text string (Devanagari, Latin, or Hinglish).
            priority: 0 (Critical), 1 (Perception), or 2 (Normal OCR).
            source: Request source tag (e.g. 'safety', 'ocr', 'manual').
            interrupt: If True, halts lower-priority active playback immediately.
            language_hint: Optional manual language hint ('hi', 'en-us', 'auto').

        Returns:
            request_id (str) for tracking lifecycle status.

        Raises:
            ValueError: If text is empty/whitespace or priority is invalid.
        """
        # Normalize priority into PriorityLevel (0: CRITICAL, 1: PERCEPTION, 2: NORMAL)
        prio_val = getattr(priority, "value", priority)
        if isinstance(prio_val, int) and prio_val not in (0, 1, 2):
            if prio_val in (1, 2):
                prio_val = 0
            elif prio_val in (3, 5):
                prio_val = 1
            else:
                prio_val = 2

        req = SpeechRequest(
            text=text,
            priority=prio_val,
            source=source,
            interrupt=interrupt,
            language_hint=language_hint,
        )

        with self._lock:
            # If an incoming request interrupts active playback of lower priority:
            if req.interrupt and self._active_request is not None:
                if req.priority < self._active_request.priority:
                    logger.info(
                        "Preempting active request %s (prio %d) for incoming alert %s (prio %d).",
                        self._active_request.request_id,
                        self._active_request.priority,
                        req.request_id,
                        req.priority,
                    )
                    self._cancel_active_speech_locked()

            enqueued = self.queue.enqueue(req)
            if not enqueued:
                logger.warning("Request %s could not be enqueued (status=%s).", req.request_id, req.status)

        return req.request_id

    def stop(self) -> None:
        """Immediately interrupt active audio playback and cancel the active request."""
        with self._lock:
            self._cancel_active_speech_locked()
            logger.info("TTSManager stop() invoked; playback and active synthesis halted.")

    def pause(self) -> None:
        """Pause active audio output and suspend subsequent chunk synthesis.

        Note on Pause Semantics:
        - Active playback: PortAudio stream holds position without clearing buffer.
        - Active synthesis: The current chunk synthesis completes, but its audio
          is withheld from playback until resume() is called.
        """
        with self._lock:
            self._pause_event.clear()
            self.player.pause()
            logger.info("TTSManager paused.")

    def resume(self) -> None:
        """Resume playback and chunk synthesis from paused state."""
        with self._lock:
            self._pause_event.set()
            self.player.resume()
            logger.info("TTSManager resumed.")

    def is_speaking(self) -> bool:
        """Return True if actively processing, synthesizing, or outputting audio."""
        with self._lock:
            return (
                not self.queue.is_empty()
                or self._active_request is not None
                or self._is_synthesizing
                or self.player.is_playing()
            )

    def get_status(self) -> Dict[str, Any]:
        """Return structured diagnostics and status."""
        with self._lock:
            return {
                "engine_ready": self.engine.is_ready(),
                "active_provider": getattr(self.engine, "active_provider", "Unknown"),
                "player_state": self.player.get_state().value,
                "queue_size": self.queue.qsize(),
                "is_speaking": self.is_speaking(),
                "is_synthesizing": self._is_synthesizing,
                "active_request_id": self._active_request.request_id if self._active_request else None,
                "active_request_priority": self._active_request.priority if self._active_request else None,
                "active_request_source": self._active_request.source if self._active_request else None,
            }

    def _cancel_active_speech_locked(self) -> None:
        """Cancel current speech processing and stop audio playback."""
        self._cancel_current_event.set()
        self._chunk_finished_event.set()
        self.player.stop()

        if self._active_request is not None and self._active_request.status in (
            RequestStatus.PENDING,
            RequestStatus.SYNTHESIZING,
            RequestStatus.PLAYING,
        ):
            self._active_request.status = RequestStatus.CANCELLED
            self._active_request.error_message = "Preempted or stopped by user."

    def _process_queue_loop(self) -> None:
        """Dedicated background worker thread loop."""
        while not self._stop_worker_event.is_set():
            try:
                request = self.queue.get(timeout=0.1)
                if request is None:
                    continue

                with self._lock:
                    self._active_request = request
                    self._cancel_current_event.clear()
                    request.status = RequestStatus.SYNTHESIZING

                self._process_single_request(request)

                with self._lock:
                    if self._active_request == request:
                        if request.status not in (RequestStatus.CANCELLED, RequestStatus.FAILED):
                            request.status = RequestStatus.COMPLETED
                        self._active_request = None

            except Exception as exc:
                logger.error("Unhandled error in TTSManager worker loop: %s", exc, exc_info=True)
                time.sleep(0.05)

    def _process_single_request(self, request: SpeechRequest) -> None:
        """Normalize, chunk, synthesize, and stream playback for a single SpeechRequest."""
        try:
            # 1. Conservative Normalization
            normalized_text = self.normalizer.normalize(request.text)
            if not normalized_text:
                request.status = RequestStatus.COMPLETED
                return

            # 2. Voice and Language Routing
            voice, lang = self.normalizer.resolve_voice_and_lang(
                normalized_text, request.language_hint
            )

            # 3. Intelligent Sentence/Clause Chunking
            chunks = self.chunker.chunk(normalized_text)
            if not chunks:
                chunks = [normalized_text]

            logger.debug(
                "Processing request %s: %d chunks (voice=%s, lang=%s).",
                request.request_id,
                len(chunks),
                voice,
                lang,
            )

            # 4. Pipelined Chunk Synthesis & Playback
            for chunk_idx, chunk_text in enumerate(chunks):
                if self._cancel_current_event.is_set() or self._stop_worker_event.is_set():
                    logger.debug("Request %s cancelled before chunk %d.", request.request_id, chunk_idx)
                    return

                # Check if paused before synthesis
                self._pause_event.wait()

                # Synthesize chunk
                with self._lock:
                    self._is_synthesizing = True

                try:
                    samples, sr = self.engine.synthesize(
                        text=chunk_text,
                        voice=voice,
                        lang=lang,
                    )
                finally:
                    with self._lock:
                        self._is_synthesizing = False

                if self._cancel_current_event.is_set() or self._stop_worker_event.is_set():
                    return

                # Check if paused before playback
                self._pause_event.wait()

                # Play chunk audio
                self._chunk_finished_event.clear()

                def _on_chunk_done() -> None:
                    self._chunk_finished_event.set()

                with self._lock:
                    request.status = RequestStatus.PLAYING

                self.player.play(samples, sample_rate=sr, on_done=_on_chunk_done)

                # Wait for playback of this chunk to finish or preemption
                while not self._chunk_finished_event.is_set():
                    if self._cancel_current_event.is_set() or self._stop_worker_event.is_set():
                        self.player.stop()
                        return
                    time.sleep(0.01)

        except Exception as exc:
            logger.error("Failed to process SpeechRequest %s: %s", request.request_id, exc)
            request.status = RequestStatus.FAILED
            request.error_message = str(exc)

    def shutdown(self, timeout: float = 3.0) -> None:
        """Safely shut down the TTSManager, worker thread, player, and engine."""
        logger.info("Initiating TTSManager shutdown...")
        self.stop()
        self._stop_worker_event.set()
        self._pause_event.set()  # Unblock if paused
        self.queue.shutdown()

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        self.player.shutdown()
        self.engine.shutdown()
        logger.info("TTSManager shutdown successfully finalized.")
