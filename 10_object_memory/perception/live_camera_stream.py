"""NAYRA Module 10 — High-Performance Threaded Live Camera Stream Receiver.

Features:
- Persistent camera connection (Android IP Webcam HTTP MJPEG / RTSP / local USB webcam).
- Background capture worker continuously acquiring frames at native camera FPS (15–30 FPS).
- Zero-lag latest-frame buffer (size = 1): stale frames are immediately discarded.
- Decouples camera acquisition from downstream YOLOv10 object detection and UI rendering.
- Real-time telemetry: independent capture FPS, read FPS, and frame drop counters.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Dynamically import AndroidStreamReceiver to respect numeric module directory
_cam_mod = importlib.import_module("01_camera_input.android_stream")
AndroidStreamReceiver = _cam_mod.AndroidStreamReceiver

logger = logging.getLogger("NAYRA-LiveCameraStream")


class LiveCameraStream:
    """Threaded camera stream receiver maintaining a strict 1-frame zero-lag buffer."""

    def __init__(
        self,
        source: Union[str, int],
        target_fps: float = 30.0,
        connection_timeout_sec: float = 5.0,
    ) -> None:
        self.source = source
        self.target_fps = target_fps
        self.connection_timeout_sec = connection_timeout_sec

        self._receiver = AndroidStreamReceiver(
            source=source,
            connection_timeout_sec=connection_timeout_sec,
        )

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_id: int = 0

        self._is_running = False
        self._capture_thread: Optional[threading.Thread] = None

        # Telemetry
        self._capture_fps: float = 0.0
        self._captured_count: int = 0
        self._dropped_stale_count: int = 0
        self._last_capture_time: float = 0.0
        self._ema_alpha = 0.15

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def capture_fps(self) -> float:
        return round(self._capture_fps, 1)

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def start(self) -> bool:
        """Connect to video source and start the background capture thread."""
        if self._is_running:
            return True

        logger.info("Connecting persistent camera stream to '%s'...", self.source)
        if not self._receiver.connect():
            logger.error("Failed to connect to camera source '%s': %s", self.source, self._receiver.last_error)
            return False

        self._is_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureWorker",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("Camera capture worker started for source '%s'.", self.source)
        return True

    def stop(self) -> None:
        """Stop background capture worker and close persistent camera stream."""
        if not self._is_running:
            return

        self._is_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._receiver.disconnect()
        with self._lock:
            self._latest_frame = None
        logger.info("Camera stream disconnected and capture thread stopped.")

    def _capture_loop(self) -> None:
        """Continuous background thread fetching latest frame into 1-frame buffer."""
        target_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.033

        while self._is_running:
            t_start = time.perf_counter()
            grabbed, frame = self._receiver.read_raw()

            if grabbed and frame is not None and frame.size > 0:
                now = time.time()
                with self._lock:
                    if self._latest_frame is not None:
                        self._dropped_stale_count += 1
                    self._latest_frame = frame
                    self._latest_timestamp = now
                    self._frame_id += 1

                self._captured_count += 1
                if self._last_capture_time > 0:
                    dt = now - self._last_capture_time
                    inst_fps = 1.0 / dt if dt > 0 else 0.0
                    if self._capture_fps == 0.0:
                        self._capture_fps = inst_fps
                    else:
                        self._capture_fps = (self._ema_alpha * inst_fps) + ((1.0 - self._ema_alpha) * self._capture_fps)
                self._last_capture_time = now
            else:
                # Brief sleep if frame unavailable or on error
                time.sleep(0.01)

            # Throttle if camera source is a local video file or unthrottled loop
            elapsed = time.perf_counter() - t_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Atomically fetch the freshest frame without blocking downstream inference.

        Returns:
            (has_frame, frame_copy, timestamp)
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            # Return copy or reference (return a copy to prevent race conditions during downstream drawing)
            frame_out = self._latest_frame.copy()
            ts_out = self._latest_timestamp
            return True, frame_out, ts_out

    def get_telemetry(self) -> dict[str, Union[float, int, str]]:
        """Return stream health and performance statistics."""
        stream_info = self._receiver.get_stream_info()
        return {
            "source": str(self.source),
            "backend": stream_info.backend_name,
            "resolution": f"{stream_info.native_width}x{stream_info.native_height}",
            "capture_fps": self.capture_fps,
            "total_frames_captured": self._captured_count,
            "stale_frames_overwritten": self._dropped_stale_count,
            "is_connected": self._receiver.is_connected(),
        }
