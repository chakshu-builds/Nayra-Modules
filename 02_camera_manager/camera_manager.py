"""High-level camera facade providing frame acquisition, status monitoring, and lifecycle management."""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

_cam_in = importlib.import_module("01_camera_input")
AndroidStreamReceiver = _cam_in.AndroidStreamReceiver
ConnectionState = _cam_in.ConnectionState
StreamInfo = _cam_in.StreamInfo
FrameReader = _cam_in.FrameReader
FrameMetadata = _cam_in.FrameMetadata

if "camera_input" not in sys.modules:
    sys.modules["camera_input"] = _cam_in

from config.settings import CameraSettings, get_default_settings

logger = logging.getLogger("nayra.camera_manager")


@dataclass(frozen=True)
class CameraStatus:
    is_running: bool
    connection_state: str
    source: Union[str, int]
    fps: float
    capture_fps: float
    resolution: Tuple[int, int]
    frame_count: int
    reconnect_count: int
    last_timestamp_wall: Optional[float]
    last_timestamp_monotonic: Optional[float]
    last_error: Optional[str]
    native_resolution: Tuple[int, int]
    buffersize_applied: bool = False


class CameraManager:
    """Provides a thread-safe latest-frame camera interface for downstream perception pipelines."""

    def __init__(
        self,
        settings: Optional[CameraSettings] = None,
        source: Optional[Union[str, int]] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[float] = None,
        receiver: Optional[AndroidStreamReceiver] = None,
    ) -> None:
        self._settings = settings or get_default_settings()
        self._source = source if source is not None else self._settings.stream_url
        self._target_width = target_width if target_width is not None else self._settings.target_width
        self._target_height = target_height if target_height is not None else self._settings.target_height
        self._target_fps = target_fps if target_fps is not None else self._settings.target_fps
        self._custom_receiver = receiver

        self._receiver: Optional[AndroidStreamReceiver] = None
        self._reader: Optional[FrameReader] = None
        self._is_started = False
        self._last_metadata: Optional[FrameMetadata] = None

    @property
    def source(self) -> Union[str, int]:
        return self._source

    @property
    def is_running(self) -> bool:
        return self._is_started and self._reader is not None

    def start(self, wait_for_first_frame: bool = True, timeout_sec: float = 3.0) -> bool:
        if self._is_started:
            return True

        logger.info("Starting CameraManager with source: %s", self._source)

        if self._custom_receiver is not None:
            self._receiver = self._custom_receiver
        else:
            self._receiver = AndroidStreamReceiver(
                source=self._source,
                connection_timeout_sec=self._settings.connection_timeout_sec,
            )

        self._reader = FrameReader(
            receiver=self._receiver,
            target_width=self._target_width,
            target_height=self._target_height,
            target_fps=self._target_fps,
            reconnect_interval_sec=self._settings.reconnect_interval_sec,
            max_reconnect_attempts=self._settings.max_reconnect_attempts,
            enable_auto_reconnect=self._settings.enable_auto_reconnect,
        )

        if not self._reader.start():
            logger.error("Failed to start FrameReader worker.")
            return False

        self._is_started = True

        if wait_for_first_frame:
            frame, meta = self._reader.get_latest_frame(timeout=timeout_sec)
            if frame is not None:
                self._last_metadata = meta
                logger.info("First frame received (%dx%d)", frame.shape[1], frame.shape[0])
                return True
            logger.warning("Capture thread started, but initial frame timed out after %.1fs", timeout_sec)
            return False

        return True

    def stop(self) -> None:
        if not self._is_started:
            return

        logger.info("Stopping CameraManager...")
        if self._reader is not None:
            self._reader.stop()
            self._reader = None

        if self._receiver is not None:
            self._receiver.disconnect()
            self._receiver = None

        self._is_started = False
        self._last_metadata = None
        logger.info("CameraManager stopped.")

    def get_frame(
        self,
        timeout: Optional[float] = 1.0,
        wait_for_new: bool = False,
    ) -> Optional[np.ndarray]:
        if not self._is_started or self._reader is None:
            return None

        frame, meta = self._reader.get_latest_frame(timeout=timeout, wait_for_new=wait_for_new)
        if meta is not None:
            self._last_metadata = meta
        return frame

    def get_frame_with_metadata(
        self,
        timeout: Optional[float] = 1.0,
        wait_for_new: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        if not self._is_started or self._reader is None:
            return None, None

        frame, meta = self._reader.get_latest_frame(timeout=timeout, wait_for_new=wait_for_new)
        if meta is not None:
            self._last_metadata = meta
        return frame, meta

    def get_status(self) -> CameraStatus:
        if not self._is_started or self._reader is None or self._receiver is None:
            return CameraStatus(
                is_running=False,
                connection_state=ConnectionState.STOPPED.value,
                source=self._source,
                fps=0.0,
                capture_fps=0.0,
                resolution=(0, 0),
                frame_count=0,
                reconnect_count=0,
                last_timestamp_wall=None,
                last_timestamp_monotonic=None,
                last_error=None,
                native_resolution=(0, 0),
                buffersize_applied=False,
            )

        stream_info = self._receiver.get_stream_info()
        native_res = (stream_info.native_width, stream_info.native_height)

        current_res = (0, 0)
        wall_ts = None
        mono_ts = None

        if self._last_metadata is not None:
            current_res = (self._last_metadata.width, self._last_metadata.height)
            wall_ts = self._last_metadata.timestamp_wall
            mono_ts = self._last_metadata.timestamp_monotonic
        elif native_res != (0, 0):
            current_res = native_res

        capture_rate = self._reader.fps

        return CameraStatus(
            is_running=self.is_running,
            connection_state=self._reader.state.value,
            source=self._source,
            fps=capture_rate,
            capture_fps=capture_rate,
            resolution=current_res,
            frame_count=self._reader.frame_count,
            reconnect_count=self._reader.reconnect_count,
            last_timestamp_wall=wall_ts,
            last_timestamp_monotonic=mono_ts,
            last_error=self._reader.last_error,
            native_resolution=native_res,
            buffersize_applied=stream_info.buffersize_applied,
        )

    def __enter__(self) -> CameraManager:
        self.start(wait_for_first_frame=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
