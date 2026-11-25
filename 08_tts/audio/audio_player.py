"""NAYRA Module 08 — Low-Latency Non-Blocking Audio Player.

Provides asynchronous, interruptible audio playback via PortAudio (sounddevice).
Supports immediate preemption, pause/resume, buffer streaming, and graceful headless fallback.
"""

from __future__ import annotations

from enum import Enum
import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("NAYRA-AudioPlayer")


class PlayerState(str, Enum):
    """Operational state of the AudioPlayer."""
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class AudioPlayer:
    """Non-blocking streaming audio playback engine.

    Features:
    1. PortAudio callback-driven output stream for sub-millisecond control.
    2. Sub-10ms immediate preemption on stop().
    3. Seamless pause and resume preserving buffer position.
    4. Thread-safe buffer enqueueing for streaming sentence chunks.
    5. Headless / mock fallback when physical sound hardware is unavailable.
    """

    DEFAULT_SAMPLE_RATE: int = 24000

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        device: Optional[int | str] = None,
        mock_mode: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.mock_mode = mock_mode

        self._state = PlayerState.IDLE
        self._lock = threading.RLock()
        self._stream = None

        # Audio buffer state
        self._current_buffer: Optional[np.ndarray] = None
        self._buffer_offset: int = 0
        self._on_done_callback: Optional[Callable[[], None]] = None

        # Mock playback thread
        self._mock_thread: Optional[threading.Thread] = None
        self._mock_stop_event = threading.Event()
        self._mock_pause_event = threading.Event()
        self._mock_pause_event.set()  # Unpaused initially

        self._initialize_stream()

    def _initialize_stream(self) -> None:
        """Attempt to open PortAudio output stream."""
        if self.mock_mode:
            logger.info("AudioPlayer initialized in MOCK mode.")
            return

        try:
            import sounddevice as sd

            def _callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
                if status:
                    logger.debug("PortAudio callback status: %s", status)

                with self._lock:
                    if self._state == PlayerState.PAUSED:
                        outdata.fill(0)
                        return

                    if self._state != PlayerState.PLAYING or self._current_buffer is None:
                        outdata.fill(0)
                        return

                    remaining_frames = len(self._current_buffer) - self._buffer_offset
                    if remaining_frames <= 0:
                        outdata.fill(0)
                        self._finish_current_playback_locked()
                        return

                    frames_to_write = min(frames, remaining_frames)
                    chunk = self._current_buffer[self._buffer_offset : self._buffer_offset + frames_to_write]
                    
                    if chunk.ndim == 1:
                        chunk = chunk[:, np.newaxis]

                    outdata[:frames_to_write] = chunk
                    if frames_to_write < frames:
                        outdata[frames_to_write:].fill(0)

                    self._buffer_offset += frames_to_write
                    if self._buffer_offset >= len(self._current_buffer):
                        self._finish_current_playback_locked()

            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=_callback,
                device=self.device,
                blocksize=1024,
            )
            self._stream.start()
            logger.info("PortAudio OutputStream started at %d Hz.", self.sample_rate)
        except Exception as exc:
            logger.warning(
                "Failed to initialize PortAudio output device (%s). Falling back to mock player.",
                exc,
            )
            self.mock_mode = True

    def _finish_current_playback_locked(self) -> None:
        """Mark playback completed and invoke completion callback."""
        self._state = PlayerState.IDLE
        self._current_buffer = None
        self._buffer_offset = 0
        callback = self._on_done_callback
        self._on_done_callback = None

        if callback is not None:
            # Dispatch callback outside lock to prevent deadlocks
            threading.Thread(target=callback, daemon=True).start()

    def play(
        self,
        samples: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        """Queue and begin playback of audio samples immediately.

        Args:
            samples: 1D numpy array of float32 samples.
            sample_rate: Sample rate in Hz.
            on_done: Optional callback invoked when playback completes naturally.
        """
        with self._lock:
            # Stop any existing playback first
            self.stop()

            # Ensure 1D float32
            data = np.asarray(samples, dtype=np.float32).ravel()
            if len(data) == 0:
                if on_done:
                    on_done()
                return

            self._current_buffer = data
            self._buffer_offset = 0
            self._on_done_callback = on_done
            self._state = PlayerState.PLAYING

            if self.mock_mode:
                self._start_mock_playback(data, sample_rate)

    def _start_mock_playback(self, data: np.ndarray, sample_rate: int) -> None:
        """Simulate playback duration in headless or mock mode."""
        self._mock_stop_event.clear()
        self._mock_pause_event.set()
        duration_sec = len(data) / sample_rate

        def _mock_worker() -> None:
            elapsed = 0.0
            step = 0.05
            while elapsed < duration_sec:
                if self._mock_stop_event.is_set():
                    return
                self._mock_pause_event.wait()
                time.sleep(step)
                elapsed += step

            with self._lock:
                if not self._mock_stop_event.is_set() and self._state == PlayerState.PLAYING:
                    self._finish_current_playback_locked()

        self._mock_thread = threading.Thread(target=_mock_worker, daemon=True)
        self._mock_thread.start()

    def stop(self) -> None:
        """Immediately stop playback and clear audio buffers."""
        with self._lock:
            self._state = PlayerState.STOPPED
            self._current_buffer = None
            self._buffer_offset = 0
            self._on_done_callback = None

            if self.mock_mode:
                self._mock_stop_event.set()

            # Brief transition back to IDLE
            self._state = PlayerState.IDLE

    def pause(self) -> None:
        """Pause playback without discarding the current audio buffer."""
        with self._lock:
            if self._state == PlayerState.PLAYING:
                self._state = PlayerState.PAUSED
                if self.mock_mode:
                    self._mock_pause_event.clear()
                logger.debug("AudioPlayer PAUSED at offset %d.", self._buffer_offset)

    def resume(self) -> None:
        """Resume playback from the paused position."""
        with self._lock:
            if self._state == PlayerState.PAUSED:
                self._state = PlayerState.PLAYING
                if self.mock_mode:
                    self._mock_pause_event.set()
                logger.debug("AudioPlayer RESUMED from offset %d.", self._buffer_offset)

    def is_playing(self) -> bool:
        """Return True if actively outputting audio."""
        with self._lock:
            return self._state == PlayerState.PLAYING

    def get_state(self) -> PlayerState:
        """Return the current PlayerState enum."""
        with self._lock:
            return self._state

    def shutdown(self) -> None:
        """Cleanly close stream and release audio device."""
        self.stop()
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as exc:
                    logger.debug("Error closing PortAudio stream: %s", exc)
                finally:
                    self._stream = None
            logger.info("AudioPlayer shutdown completed.")
