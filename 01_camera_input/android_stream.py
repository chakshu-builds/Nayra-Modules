"""Network and local camera stream receiver with HTTP MJPEG and OpenCV backends."""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import cv2
import numpy as np

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000",
)

logger = logging.getLogger("nayra.camera_input.android_stream")

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
MAX_MJPEG_BUFFER_BYTES = 5 * 1024 * 1024


class ConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    STREAMING = "STREAMING"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass
class StreamInfo:
    source: Union[str, int]
    is_opened: bool
    native_width: int
    native_height: int
    reported_fps: float
    backend_name: str
    buffersize_applied: bool = False


class HttpMjpegClient:
    """Parses multipart/x-mixed-replace HTTP MJPEG streams directly."""

    def __init__(self, url: str, timeout_sec: float = 5.0) -> None:
        self._url = url
        self._timeout_sec = max(1.0, timeout_sec)
        self._response = None
        self._buffer = bytearray()
        self._is_connected = False
        self._last_error: Optional[str] = None
        self._native_width = 0
        self._native_height = 0

    @property
    def is_connected(self) -> bool:
        return bool(self._is_connected and self._response is not None)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def native_resolution(self) -> Tuple[int, int]:
        return self._native_width, self._native_height

    def connect(self) -> bool:
        self._last_error = None
        self._buffer.clear()
        logger.info("HttpMjpegClient: connecting to %s (timeout=%.1fs)", self._url, self._timeout_sec)

        req = urllib.request.Request(
            self._url,
            headers={
                "User-Agent": "NAYRA/1.0",
                "Accept": "multipart/x-mixed-replace, image/jpeg, */*",
                "Connection": "keep-alive",
            },
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout_sec)
            self._response = resp
            self._is_connected = True

            got_frame, frame = self.read_raw()
            if got_frame and frame is not None:
                self._native_height, self._native_width = frame.shape[:2]
                logger.info("HttpMjpegClient: first frame decoded (%dx%d)", self._native_width, self._native_height)
                return True

            err = "HttpMjpegClient: stream opened but initial JPEG frame decode failed"
            logger.warning(err)
            self.disconnect()
            self._last_error = err
            return False

        except urllib.error.HTTPError as err:
            self._last_error = f"HTTP Error {err.code}: {err.reason}"
            logger.error("HttpMjpegClient: %s", self._last_error)
            self.disconnect()
            return False
        except urllib.error.URLError as err:
            self._last_error = f"URL Error: {err.reason}"
            logger.error("HttpMjpegClient: %s", self._last_error)
            self.disconnect()
            return False
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception("HttpMjpegClient: unexpected connection failure")
            self.disconnect()
            return False

    def disconnect(self) -> None:
        self._is_connected = False
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            finally:
                self._response = None
        self._buffer.clear()

    def read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected or self._response is None:
            return False, None

        while True:
            start_idx = self._buffer.find(JPEG_START)
            if start_idx > 0:
                del self._buffer[:start_idx]
                start_idx = 0

            if start_idx == 0:
                end_idx = self._buffer.find(JPEG_END, 2)
                if end_idx != -1:
                    # Drop stale frames if newer complete frame already in buffer
                    next_start = self._buffer.find(JPEG_START, end_idx + 2)
                    if next_start != -1:
                        next_end = self._buffer.find(JPEG_END, next_start + 2)
                        if next_end != -1:
                            del self._buffer[:next_start]
                            continue

                    jpeg_bytes = bytes(self._buffer[: end_idx + 2])
                    del self._buffer[: end_idx + 2]

                    frame = cv2.imdecode(
                        np.frombuffer(jpeg_bytes, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is not None and frame.size > 0:
                        self._native_height, self._native_width = frame.shape[:2]
                        return True, frame

            try:
                chunk = self._response.read(16384)
                if not chunk:
                    self._last_error = "HTTP stream closed by peer"
                    self._is_connected = False
                    return False, None
                self._buffer.extend(chunk)
            except Exception as read_err:
                self._last_error = str(read_err)
                self._is_connected = False
                return False, None

            if len(self._buffer) > MAX_MJPEG_BUFFER_BYTES:
                logger.warning("HttpMjpegClient: buffer overflow (>5MB). Resetting.")
                self._buffer.clear()


class AndroidStreamReceiver:
    """Manages video streaming from Android HTTP MJPEG, RTSP, local cameras, or files."""

    def __init__(
        self,
        source: Union[str, int],
        connection_timeout_sec: float = 5.0,
    ) -> None:
        if isinstance(source, str) and source.strip().isdigit():
            self._source: Union[str, int] = int(source.strip())
        elif isinstance(source, str):
            self._source = self._normalize_stream_url(source.strip())
        else:
            self._source = source

        self._timeout_sec = connection_timeout_sec
        self._cap: Optional[cv2.VideoCapture] = None
        self._http_client: Optional[HttpMjpegClient] = None
        self._active_backend: str = "None"
        self._state: ConnectionState = ConnectionState.DISCONNECTED
        self._last_error: Optional[str] = None
        self._buffersize_applied: bool = False

    @staticmethod
    def _normalize_stream_url(url: str) -> str:
        lower = url.lower()
        if lower.startswith(("http://", "https://")):
            parsed = urllib.parse.urlparse(url)
            if parsed.port in (8080, 4747) and parsed.path in ("", "/"):
                return urllib.parse.urljoin(url, "/video")
        return url

    @property
    def source(self) -> Union[str, int]:
        return self._source

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def buffersize_applied(self) -> bool:
        return self._buffersize_applied

    def is_connected(self) -> bool:
        if self._http_client is not None:
            return self._http_client.is_connected
        return bool(self._cap is not None and self._cap.isOpened())

    def connect(self) -> bool:
        self._state = ConnectionState.CONNECTING
        self._last_error = None
        self._buffersize_applied = False
        self.disconnect()

        # Preferred strategy for HTTP endpoints: dedicated MJPEG client
        if isinstance(self._source, str) and self._source.lower().startswith(("http://", "https://")):
            http_client = HttpMjpegClient(self._source, timeout_sec=self._timeout_sec)
            if http_client.connect():
                self._http_client = http_client
                self._active_backend = "HTTP_MJPEG"
                self._buffersize_applied = True
                self._state = ConnectionState.CONNECTED
                return True
            logger.warning(
                "HTTP_MJPEG failed (%s). Attempting OpenCV VideoCapture fallback...",
                http_client.last_error,
            )

        # Fallback or standard path for local webcams, RTSP, and video files
        try:
            timeout_ms = int(self._timeout_sec * 1000)
            open_params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
            ]

            if isinstance(self._source, int):
                cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self._source)
            else:
                try:
                    cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG, open_params)
                except Exception:
                    cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)

                if not cap.isOpened():
                    cap = cv2.VideoCapture(self._source)

            if not cap.isOpened():
                backend_info = "Unknown"
                try:
                    if hasattr(cap, "getBackendName"):
                        backend_info = cap.getBackendName()
                except Exception:
                    pass
                err_msg = f"Failed to open video source '{self._source}' with OpenCV ({backend_info})."
                logger.error(err_msg)
                self._state = ConnectionState.ERROR
                self._last_error = err_msg
                cap.release()
                return False

            try:
                self._buffersize_applied = bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            except Exception:
                self._buffersize_applied = False

            self._cap = cap
            self._active_backend = cap.getBackendName()
            self._state = ConnectionState.CONNECTED
            return True

        except Exception as exc:
            self._state = ConnectionState.ERROR
            self._last_error = str(exc)
            logger.exception("Exception opening video source %s", self._source)
            return False

    def disconnect(self) -> None:
        if self._http_client is not None:
            try:
                self._http_client.disconnect()
            finally:
                self._http_client = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Error releasing cv2.VideoCapture: %s", exc)
            finally:
                self._cap = None

        self._active_backend = "None"
        if self._state not in (ConnectionState.STOPPED, ConnectionState.ERROR):
            self._state = ConnectionState.DISCONNECTED

    def read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._http_client is not None:
            grabbed, frame = self._http_client.read_raw()
            if not grabbed:
                self._last_error = self._http_client.last_error
            return grabbed, frame

        if not self.is_connected() or self._cap is None:
            return False, None

        try:
            grabbed, frame = self._cap.read()
            if not grabbed or frame is None:
                return False, None
            return True, frame
        except Exception as exc:
            self._last_error = str(exc)
            return False, None

    def get_stream_info(self) -> StreamInfo:
        if self._http_client is not None and self._http_client.is_connected:
            w, h = self._http_client.native_resolution
            return StreamInfo(
                source=self._source,
                is_opened=True,
                native_width=w,
                native_height=h,
                reported_fps=0.0,
                backend_name="HTTP_MJPEG",
                buffersize_applied=True,
            )

        if not self.is_connected() or self._cap is None:
            return StreamInfo(
                source=self._source,
                is_opened=False,
                native_width=0,
                native_height=0,
                reported_fps=0.0,
                backend_name="None",
                buffersize_applied=False,
            )

        try:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            backend_name = self._cap.getBackendName()
        except Exception:
            width, height, fps, backend_name = 0, 0, 0.0, "Unknown"

        return StreamInfo(
            source=self._source,
            is_opened=True,
            native_width=width,
            native_height=height,
            reported_fps=fps,
            backend_name=backend_name,
            buffersize_applied=self._buffersize_applied,
        )
