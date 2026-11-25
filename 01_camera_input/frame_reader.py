"""Threaded frame reader, validation, scaling, FPS calculation, and buffering."""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass
from typing import Deque, Optional, Tuple, Union

import cv2
import numpy as np

from .android_stream import AndroidStreamReceiver, ConnectionState

logger = logging.getLogger("nayra.camera_input.frame_reader")


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: int
    timestamp_wall: float
    timestamp_monotonic: float
    width: int
    height: int
    channels: int
    read_latency_ms: float


@dataclass(frozen=True)
class FrameValidationResult:
    is_valid: bool
    reason: Optional[str] = None


class FrameValidator:
    """Validates frame arrays to ensure they are safe for downstream vision pipelines."""

    @staticmethod
    def validate(frame: Optional[np.ndarray]) -> FrameValidationResult:
        if frame is None:
            return FrameValidationResult(False, "Frame is None")

        if not isinstance(frame, np.ndarray):
            return FrameValidationResult(False, f"Expected numpy.ndarray, got {type(frame).__name__}")

        if frame.size == 0:
            return FrameValidationResult(False, "Frame is empty (size == 0)")

        if frame.ndim != 3:
            return FrameValidationResult(False, f"Expected 3 dimensions (H, W, C), got {frame.ndim}")

        height, width, channels = frame.shape
        if height <= 0 or width <= 0:
            return FrameValidationResult(False, f"Invalid dimensions: height={height}, width={width}")

        if channels != 3:
            return FrameValidationResult(False, f"Expected 3 channels (BGR), got {channels}")

        if frame.dtype != np.uint8:
            return FrameValidationResult(False, f"Expected dtype uint8, got {frame.dtype}")

        return FrameValidationResult(True, None)


class ResolutionHandler:
    """Handles resolution scaling of incoming video frames."""

    def __init__(
        self,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height

    @property
    def is_active(self) -> bool:
        return self.target_width is not None and self.target_height is not None

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.is_active:
            return frame

        curr_h, curr_w = frame.shape[:2]
        if curr_w == self.target_width and curr_h == self.target_height:
            return frame

        interpolation = (
            cv2.INTER_AREA
            if (self.target_width < curr_w and self.target_height < curr_h)
            else cv2.INTER_LINEAR
        )

        return cv2.resize(
            frame,
            (self.target_width, self.target_height),
            interpolation=interpolation,
        )


class FPSCalculator:
    """Tracks rolling frame rates using a sliding timestamp window."""

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = max(2, window_size)
        self._timestamps: Deque[float] = collections.deque(maxlen=self._window_size)
        self._last_instant_fps: float = 0.0
        self._lock = threading.Lock()

    def record_frame(self, timestamp: Optional[float] = None) -> float:
        now = time.monotonic() if timestamp is None else timestamp

        with self._lock:
            if self._timestamps:
                dt = now - self._timestamps[-1]
                if dt > 1e-6:
                    self._last_instant_fps = 1.0 / dt

            self._timestamps.append(now)

            if len(self._timestamps) < 2:
                return 0.0

            total_elapsed = self._timestamps[-1] - self._timestamps[0]
            if total_elapsed <= 1e-6:
                return self._last_instant_fps

            return float((len(self._timestamps) - 1) / total_elapsed)

    def get_fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            total_elapsed = self._timestamps[-1] - self._timestamps[0]
            if total_elapsed <= 1e-6:
                return 0.0
            return float((len(self._timestamps) - 1) / total_elapsed)

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()
            self._last_instant_fps = 0.0


class LatestFrameBuffer:
    """Thread-safe single-slot buffer implementing a latest-frame drop policy."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_metadata: Optional[FrameMetadata] = None
        self._new_frame_available: bool = False

    def push(self, frame: np.ndarray, metadata: FrameMetadata) -> None:
        with self._condition:
            self._latest_frame = frame
            self._latest_metadata = metadata
            self._new_frame_available = True
            self._condition.notify_all()

    def get_latest(
        self,
        timeout: Optional[float] = 1.0,
        wait_for_new: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        with self._condition:
            start_time = time.monotonic()
            if timeout is not None and timeout > 0:
                while True:
                    condition_met = (
                        (self._latest_frame is not None and self._new_frame_available)
                        if wait_for_new
                        else (self._latest_frame is not None)
                    )
                    if condition_met:
                        break
                    elapsed = time.monotonic() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)

            if self._latest_frame is None or (wait_for_new and not self._new_frame_available):
                return None, None

            frame = self._latest_frame.copy()
            meta = self._latest_metadata
            self._new_frame_available = False
            return frame, meta

    def clear(self) -> None:
        with self._condition:
            self._latest_frame = None
            self._latest_metadata = None
            self._new_frame_available = False
            self._condition.notify_all()


class FrameReader:
    """Background worker that continuously ingests, validates, and buffers frames."""

    def __init__(
        self,
        receiver: AndroidStreamReceiver,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[float] = None,
        reconnect_interval_sec: float = 2.0,
        max_reconnect_attempts: int = 0,
        enable_auto_reconnect: bool = True,
    ) -> None:
        self._receiver = receiver
        self._resolution_handler = ResolutionHandler(target_width, target_height)
        self._target_fps = target_fps
        self._reconnect_interval = reconnect_interval_sec
        self._max_reconnect_attempts = max_reconnect_attempts
        self._enable_auto_reconnect = enable_auto_reconnect

        self._buffer = LatestFrameBuffer()
        self._fps_calculator = FPSCalculator()
        self._validator = FrameValidator()

        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._frame_counter = 0
        self._dropped_frames = 0
        self._reconnect_count = 0
        self._last_error: Optional[str] = None

    @property
    def state(self) -> ConnectionState:
        with self._state_lock:
            return self._state

    @property
    def fps(self) -> float:
        return self._fps_calculator.get_fps()

    @property
    def frame_count(self) -> int:
        return self._frame_counter

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True

        self._stop_event.clear()
        self._set_state(ConnectionState.CONNECTING)

        self._thread = threading.Thread(
            target=self._capture_worker,
            name="NAYRA_FrameReaderWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("FrameReader background thread started.")
        return True

    def stop(self, timeout_sec: float = 3.0) -> None:
        logger.info("Stopping FrameReader...")
        self._stop_event.set()
        self._buffer.clear()
        self._receiver.disconnect()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_sec)
            self._thread = None

        self._fps_calculator.reset()
        self._set_state(ConnectionState.STOPPED)
        logger.info("FrameReader stopped and resources cleaned up.")

    def get_latest_frame(
        self,
        timeout: Optional[float] = 1.0,
        wait_for_new: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        return self._buffer.get_latest(timeout=timeout, wait_for_new=wait_for_new)

    def _set_state(self, new_state: ConnectionState) -> None:
        with self._state_lock:
            if self._state != new_state:
                logger.debug("State transition: %s -> %s", self._state.value, new_state.value)
                self._state = new_state

    def _capture_worker(self) -> None:
        consecutive_failures = 0
        target_interval = (
            (1.0 / self._target_fps) if (self._target_fps and self._target_fps > 0) else 0.0
        )

        while not self._stop_event.is_set():
            if not self._receiver.is_connected():
                self._set_state(ConnectionState.CONNECTING)
                connected = self._receiver.connect()
                if not connected:
                    consecutive_failures += 1
                    self._reconnect_count += 1
                    self._last_error = self._receiver.last_error

                    if (
                        self._max_reconnect_attempts > 0
                        and consecutive_failures >= self._max_reconnect_attempts
                    ):
                        self._set_state(ConnectionState.ERROR)
                        logger.error("Max reconnect attempts reached. Stopping capture worker.")
                        break

                    if not self._enable_auto_reconnect:
                        self._set_state(ConnectionState.ERROR)
                        break

                    self._set_state(ConnectionState.RECONNECTING)
                    self._stop_event.wait(self._reconnect_interval)
                    continue

                consecutive_failures = 0
                self._set_state(ConnectionState.CONNECTED)

            t_start = time.monotonic()
            grabbed, raw_frame = self._receiver.read_raw()
            read_latency_ms = (time.monotonic() - t_start) * 1000.0

            if not grabbed or raw_frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.warning("Stream interrupted. Disconnecting and triggering reconnect...")
                    self._receiver.disconnect()
                    self._reconnect_count += 1

                    if not self._enable_auto_reconnect:
                        self._set_state(ConnectionState.ERROR)
                        break

                    self._set_state(ConnectionState.RECONNECTING)
                    self._stop_event.wait(self._reconnect_interval)
                continue

            consecutive_failures = 0
            self._set_state(ConnectionState.STREAMING)

            validation = self._validator.validate(raw_frame)
            if not validation.is_valid:
                self._dropped_frames += 1
                logger.warning("Dropped invalid frame: %s", validation.reason)
                continue

            processed_frame = self._resolution_handler.process(raw_frame)

            t_monotonic = time.monotonic()
            t_wall = time.time()
            self._fps_calculator.record_frame(t_monotonic)

            self._frame_counter += 1
            h, w = processed_frame.shape[:2]
            c = processed_frame.shape[2] if processed_frame.ndim == 3 else 1

            metadata = FrameMetadata(
                frame_id=self._frame_counter,
                timestamp_wall=t_wall,
                timestamp_monotonic=t_monotonic,
                width=w,
                height=h,
                channels=c,
                read_latency_ms=read_latency_ms,
            )

            self._buffer.push(processed_frame, metadata)

            if target_interval > 0:
                elapsed = time.monotonic() - t_start
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)

        self._receiver.disconnect()
        if self._state not in (ConnectionState.ERROR, ConnectionState.STOPPED):
            self._set_state(ConnectionState.STOPPED)
