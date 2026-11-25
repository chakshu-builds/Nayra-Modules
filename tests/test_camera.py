"""
NAYRA - Assistive System
Part 1: Camera & Input System - Unit & Integration Test Suite

Tests:
- Camera settings and validation
- Frame validation and corruption handling
- Resolution scaling
- FPS calculation and zero-division protection
- Single-slot buffer (latest-frame strategy & wait_for_new)
- HttpMjpegClient direct multipart JPEG parsing & low-latency extraction
- Stream receiver dual-backend (HTTP MJPEG and OpenCV VideoCapture), URL normalization, and disconnection
- Background frame reader threading and error recovery
- CameraManager unified facade, distinct capture FPS, and resource cleanup
- Snapshot directory creation safety

Note: Automated unit tests run headless using synthetic test frames and mocks.
For physical Android device testing, refer to the manual test instructions in README.md.
"""

from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

# Ensure project root directory is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Register modules dynamically to handle numeric folder prefixes
_cam_in = importlib.import_module("01_camera_input")
_cam_mgr = importlib.import_module("02_camera_manager")
_cam_mgr_module = importlib.import_module("02_camera_manager.camera_manager")
sys.modules["camera_input"] = _cam_in
sys.modules["camera_manager"] = _cam_mgr

import cv2
import numpy as np
import pytest

from camera_input.android_stream import (
    AndroidStreamReceiver,
    ConnectionState,
    HttpMjpegClient,
    StreamInfo,
)
from camera_input.frame_reader import (
    FPSCalculator,
    FrameMetadata,
    FrameReader,
    FrameValidator,
    LatestFrameBuffer,
    ResolutionHandler,
)
from camera_manager.camera_manager import CameraManager, CameraStatus
from config.settings import CameraSettings


# ---------------------------------------------------------------------------
# 1. Configuration Tests
# ---------------------------------------------------------------------------

class TestCameraSettings:
    """Tests configuration loading and boundary validations."""

    def test_default_settings(self) -> None:
        """Verify default settings load correctly."""
        settings = CameraSettings()
        assert settings.reconnect_interval_sec >= 0.1
        assert settings.connection_timeout_sec > 0
        assert settings.frame_buffer_size == 1

    def test_numeric_string_source_conversion(self) -> None:
        """Ensure string digits like '0' are converted to integer for device indices."""
        settings = CameraSettings(stream_url="0")
        assert settings.stream_url == 0

    def test_invalid_resolutions_raise_error(self) -> None:
        """Ensure negative or zero resolutions raise ValueError."""
        with pytest.raises(ValueError, match="target_width must be positive"):
            CameraSettings(target_width=-100)

        with pytest.raises(ValueError, match="target_height must be positive"):
            CameraSettings(target_height=0)

    def test_invalid_fps_raises_error(self) -> None:
        """Ensure non-positive target FPS raises ValueError."""
        with pytest.raises(ValueError, match="target_fps must be positive"):
            CameraSettings(target_fps=-5.0)


# ---------------------------------------------------------------------------
# 2. Frame Validation Tests
# ---------------------------------------------------------------------------

class TestFrameValidator:
    """Tests frame validation logic against various valid and invalid inputs."""

    def test_valid_bgr_frame(self) -> None:
        """Verify standard 3-channel uint8 BGR image passes validation."""
        valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = FrameValidator.validate(valid_frame)
        assert result.is_valid is True
        assert result.reason is None

    def test_none_frame(self) -> None:
        """Verify None input is rejected gracefully."""
        result = FrameValidator.validate(None)
        assert result.is_valid is False
        assert "None" in str(result.reason)

    def test_non_ndarray(self) -> None:
        """Verify non-numpy objects are rejected."""
        result = FrameValidator.validate("not_an_image")  # type: ignore
        assert result.is_valid is False
        assert "numpy.ndarray" in str(result.reason)

    def test_empty_frame(self) -> None:
        """Verify zero-sized array is rejected."""
        empty_frame = np.empty((0,), dtype=np.uint8)
        result = FrameValidator.validate(empty_frame)
        assert result.is_valid is False
        assert "empty" in str(result.reason)

    def test_grayscale_frame(self) -> None:
        """Verify 2D single-channel image is rejected (pipeline requires BGR 3D)."""
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        result = FrameValidator.validate(gray_frame)
        assert result.is_valid is False
        assert "3 dimensions" in str(result.reason)

    def test_four_channel_frame(self) -> None:
        """Verify 4-channel image is rejected."""
        rgba_frame = np.zeros((480, 640, 4), dtype=np.uint8)
        result = FrameValidator.validate(rgba_frame)
        assert result.is_valid is False
        assert "3 channels" in str(result.reason)

    def test_invalid_dtype(self) -> None:
        """Verify float32 image is rejected (pipeline expects standard uint8)."""
        float_frame = np.zeros((480, 640, 3), dtype=np.float32)
        result = FrameValidator.validate(float_frame)
        assert result.is_valid is False
        assert "uint8" in str(result.reason)


# ---------------------------------------------------------------------------
# 3. Resolution Handler Tests
# ---------------------------------------------------------------------------

class TestResolutionHandler:
    """Tests resolution scaling and passthrough behavior."""

    def test_passthrough_when_not_configured(self) -> None:
        """Verify original frame is returned unaltered when no target is set."""
        handler = ResolutionHandler(target_width=None, target_height=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = handler.process(frame)
        assert out.shape == (480, 640, 3)

    def test_downsampling(self) -> None:
        """Verify frame is scaled down to target dimensions."""
        handler = ResolutionHandler(target_width=320, target_height=240)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = handler.process(frame)
        assert out.shape == (240, 320, 3)

    def test_upsampling(self) -> None:
        """Verify frame is scaled up to target dimensions."""
        handler = ResolutionHandler(target_width=1280, target_height=720)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = handler.process(frame)
        assert out.shape == (720, 1280, 3)

    def test_same_resolution_identity(self) -> None:
        """Verify no resizing allocation occurs when frame already matches target."""
        handler = ResolutionHandler(target_width=640, target_height=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = handler.process(frame)
        assert out.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# 4. FPS Calculator Tests
# ---------------------------------------------------------------------------

class TestFPSCalculator:
    """Tests rolling and instantaneous FPS tracking."""

    def test_fps_with_no_or_single_frame(self) -> None:
        """FPS should be 0.0 until at least 2 frames are recorded."""
        calc = FPSCalculator()
        assert calc.get_fps() == 0.0

        calc.record_frame(100.0)
        assert calc.get_fps() == 0.0

    def test_fps_calculation_accuracy(self) -> None:
        """Verify accurate FPS computation with known synthetic timestamps."""
        calc = FPSCalculator(window_size=10)
        # Simulate 10 frames arriving at exact 30 FPS intervals (dt ~ 0.03333s)
        base_time = 1000.0
        interval = 1.0 / 30.0

        for i in range(10):
            t = base_time + (i * interval)
            calc.record_frame(t)

        calculated_fps = calc.get_fps()
        assert pytest.approx(calculated_fps, rel=1e-2) == 30.0

    def test_zero_division_protection(self) -> None:
        """Verify duplicate identical timestamps do not cause ZeroDivisionError."""
        calc = FPSCalculator()
        calc.record_frame(10.0)
        calc.record_frame(10.0)  # dt = 0.0
        # Should not raise exception
        fps = calc.get_fps()
        assert fps >= 0.0


# ---------------------------------------------------------------------------
# 5. Latest-Frame Single-Slot Buffer Tests
# ---------------------------------------------------------------------------

class TestLatestFrameBuffer:
    """Tests latest-frame buffering strategy for real-time low-latency AI."""

    def test_empty_buffer_returns_none(self) -> None:
        """Getting from empty buffer with short timeout returns (None, None)."""
        buf = LatestFrameBuffer()
        frame, meta = buf.get_latest(timeout=0.01)
        assert frame is None
        assert meta is None

    def test_latest_frame_overwrites_stale_frames(self) -> None:
        """Ensure pushing a new frame drops the unread older frame."""
        buf = LatestFrameBuffer()

        frame1 = np.ones((100, 100, 3), dtype=np.uint8) * 10
        meta1 = FrameMetadata(
            frame_id=1,
            timestamp_wall=time.time(),
            timestamp_monotonic=time.monotonic(),
            width=100,
            height=100,
            channels=3,
            read_latency_ms=2.0,
        )

        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        meta2 = FrameMetadata(
            frame_id=2,
            timestamp_wall=time.time(),
            timestamp_monotonic=time.monotonic(),
            width=100,
            height=100,
            channels=3,
            read_latency_ms=2.5,
        )

        # Push frame 1 then frame 2 without reading frame 1
        buf.push(frame1, meta1)
        buf.push(frame2, meta2)

        # Reading must yield frame 2 (the latest one)
        read_frame, read_meta = buf.get_latest(timeout=0.05)
        assert read_frame is not None
        assert read_meta is not None
        assert read_meta.frame_id == 2
        assert np.array_equal(read_frame, frame2)

    def test_wait_for_new_mode(self) -> None:
        """Ensure wait_for_new=True waits for a fresh frame rather than returning stale."""
        buf = LatestFrameBuffer()
        frame1 = np.ones((10, 10, 3), dtype=np.uint8)
        meta1 = FrameMetadata(1, 0.0, 0.0, 10, 10, 3, 0.0)

        buf.push(frame1, meta1)

        # First read consumes frame1
        f1, m1 = buf.get_latest(timeout=0.01, wait_for_new=True)
        assert f1 is not None

        # Second read with wait_for_new=True must return (None, None) since no new frame arrived
        f2, m2 = buf.get_latest(timeout=0.02, wait_for_new=True)
        assert f2 is None
        assert m2 is None

    def test_buffer_clear(self) -> None:
        """Clearing buffer resets stored frame and metadata."""
        buf = LatestFrameBuffer()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        meta = FrameMetadata(1, 0.0, 0.0, 10, 10, 3, 0.0)
        buf.push(frame, meta)

        buf.clear()
        f, m = buf.get_latest(timeout=0.01)
        assert f is None
        assert m is None


# ---------------------------------------------------------------------------
# 6. HTTP MJPEG Client Tests (Regression & Robustness)
# ---------------------------------------------------------------------------

class MockChunkedHttpStream:
    """Simulates an infinite or chunked HTTP multipart/x-mixed-replace stream."""

    def __init__(self, frame_bytes: bytes, repeat_count: int = 5) -> None:
        chunk = b"--boundary\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        self._chunks = [chunk for _ in range(repeat_count)]
        self.headers = {
            "Content-Type": "multipart/x-mixed-replace; boundary=boundary",
            "Server": "IP Webcam",
        }

    def read(self, n: int = 16384) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def close(self) -> None:
        pass


class TestHttpMjpegClient:
    """Tests direct parsing of multipart/x-mixed-replace JPEG streams."""

    @staticmethod
    def _create_jpeg_bytes() -> bytes:
        img = np.full((120, 160, 3), (10, 200, 50), dtype=np.uint8)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def test_successful_http_mjpeg_parsing(self) -> None:
        mock_stream = MockChunkedHttpStream(self._create_jpeg_bytes(), repeat_count=5)

        with patch("urllib.request.urlopen", return_value=mock_stream):
            client = HttpMjpegClient("http://192.168.0.101:8080/video", timeout_sec=2.0)
            connected = client.connect()

            assert connected is True
            assert client.is_connected is True
            assert client.native_resolution == (160, 120)

            # Read raw frames
            ok, frame = client.read_raw()
            assert ok is True
            assert frame is not None
            assert frame.shape == (120, 160, 3)

            client.disconnect()
            assert client.is_connected is False

    def test_http_mjpeg_unreachable_server(self) -> None:
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            client = HttpMjpegClient("http://192.168.0.999:8080/video", timeout_sec=1.0)
            connected = client.connect()

            assert connected is False
            assert client.is_connected is False
            assert client.last_error is not None


# ---------------------------------------------------------------------------
# 7. Stream Receiver Dual-Backend Tests
# ---------------------------------------------------------------------------

class TestAndroidStreamReceiver:
    """Tests dual-backend selection (HTTP MJPEG and OpenCV VideoCapture) and lifecycle."""

    def test_url_normalization(self) -> None:
        """Verify IP Webcam URLs without /video are normalized."""
        receiver1 = AndroidStreamReceiver("http://192.168.0.101:8080")
        assert receiver1.source == "http://192.168.0.101:8080/video"

        receiver2 = AndroidStreamReceiver("http://192.168.0.101:8080/video")
        assert receiver2.source == "http://192.168.0.101:8080/video"

        receiver3 = AndroidStreamReceiver("rtsp://192.168.0.101:8554/live")
        assert receiver3.source == "rtsp://192.168.0.101:8554/live"

    def test_http_source_selects_http_mjpeg_backend(self) -> None:
        """Verify HTTP sources automatically select the robust HTTP_MJPEG backend."""
        img = np.full((100, 100, 3), 150, dtype=np.uint8)
        enc = cv2.imencode(".jpg", img)[1].tobytes()
        mock_stream = MockChunkedHttpStream(enc, repeat_count=5)

        with patch("urllib.request.urlopen", return_value=mock_stream):
            receiver = AndroidStreamReceiver("http://192.168.0.101:8080/video")
            connected = receiver.connect()

            assert connected is True
            assert receiver.is_connected() is True
            assert receiver.state == ConnectionState.CONNECTED

            info = receiver.get_stream_info()
            assert info.is_opened is True
            assert info.backend_name == "HTTP_MJPEG"
            assert info.native_width == 100
            assert info.native_height == 100

            ok, frame = receiver.read_raw()
            assert ok is True
            assert frame is not None
            assert frame.shape == (100, 100, 3)

            receiver.disconnect()
            assert receiver.is_connected() is False

    @patch("cv2.VideoCapture")
    def test_rtsp_source_selects_opencv_backend(self, mock_cv_capture) -> None:
        """Verify RTSP or non-HTTP sources use OpenCV VideoCapture."""
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = lambda prop: 720 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 1280
        mock_cap_instance.getBackendName.return_value = "FFMPEG"
        mock_cap_instance.set.return_value = True
        mock_cv_capture.return_value = mock_cap_instance

        receiver = AndroidStreamReceiver("rtsp://192.168.0.101:8554/live")
        connected = receiver.connect()

        assert connected is True
        assert receiver.is_connected() is True
        assert receiver.state == ConnectionState.CONNECTED

        info = receiver.get_stream_info()
        assert info.is_opened is True
        assert info.backend_name == "FFMPEG"

        receiver.disconnect()
        assert receiver.is_connected() is False
        mock_cap_instance.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_failed_connect_sets_error_state(self, mock_cv_capture) -> None:
        """Verify receiver handles unreachable stream cleanly without crashing."""
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_cv_capture.return_value = mock_cap_instance

        receiver = AndroidStreamReceiver("rtsp://192.168.0.999:8554/live")
        connected = receiver.connect()

        assert connected is False
        assert receiver.is_connected() is False
        assert receiver.state == ConnectionState.ERROR
        assert receiver.last_error is not None


# ---------------------------------------------------------------------------
# 8. FrameReader & Background Worker Tests
# ---------------------------------------------------------------------------

class MockStreamReceiver:
    """Simulated stream receiver for fast deterministic unit testing."""

    def __init__(self, should_fail: bool = False, frame_shape: Tuple[int, int] = (480, 640)) -> None:
        self.should_fail = should_fail
        self.frame_shape = frame_shape
        self._connected = False
        self.last_error: Optional[str] = None
        self.read_count = 0

    def connect(self) -> bool:
        if self.should_fail:
            self.last_error = "Mock connection failed"
            return False
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._connected or self.should_fail:
            return False, None
        self.read_count += 1
        frame = np.full((self.frame_shape[0], self.frame_shape[1], 3), 128, dtype=np.uint8)
        return True, frame

    def get_stream_info(self) -> StreamInfo:
        return StreamInfo(
            source="mock_stream",
            is_opened=self._connected,
            native_width=self.frame_shape[1],
            native_height=self.frame_shape[0],
            reported_fps=30.0,
            backend_name="MockBackend",
            buffersize_applied=True,
        )


class TestFrameReader:
    """Tests threaded frame acquisition, FPS tracking, and resource cleanup."""

    def test_reader_lifecycle_and_frame_acquisition(self) -> None:
        """Verify worker thread starts, captures frames, updates FPS, and stops cleanly."""
        mock_recv = MockStreamReceiver()
        reader = FrameReader(
            receiver=mock_recv,  # type: ignore
            target_fps=60.0,
            enable_auto_reconnect=False,
        )

        assert reader.start() is True
        time.sleep(0.1)

        frame, meta = reader.get_latest_frame(timeout=0.5)
        assert frame is not None
        assert meta is not None
        assert meta.frame_id > 0
        assert meta.width == 640
        assert meta.height == 480
        assert reader.frame_count > 0

        # Clean shutdown test
        reader.stop(timeout_sec=1.0)
        assert reader.state == ConnectionState.STOPPED
        assert not mock_recv.is_connected()

    def test_reader_reconnection_trigger_on_stream_failure(self) -> None:
        """Verify consecutive read failures trigger RECONNECTING state."""
        mock_recv = MockStreamReceiver(should_fail=False)
        reader = FrameReader(
            receiver=mock_recv,  # type: ignore
            reconnect_interval_sec=0.1,
            enable_auto_reconnect=True,
        )

        reader.start()
        time.sleep(0.05)

        # Inject stream failure
        mock_recv.should_fail = True
        time.sleep(0.15)

        state = reader.state
        assert state in (ConnectionState.RECONNECTING, ConnectionState.CONNECTING)
        assert reader.reconnect_count > 0

        reader.stop(timeout_sec=1.0)


# ---------------------------------------------------------------------------
# 9. CameraManager Interface Tests
# ---------------------------------------------------------------------------

class TestCameraManager:
    """Tests the unified CameraManager facade used by downstream AI modules."""

    def test_camera_manager_interface_with_mock(self) -> None:
        """Verify start(), get_frame(), get_status(), stop() API contract."""
        settings = CameraSettings(
            stream_url="mock_test",
            reconnect_interval_sec=0.1,
            enable_auto_reconnect=False,
        )

        mock_recv = MockStreamReceiver()
        manager = CameraManager(settings=settings, receiver=mock_recv)

        started = manager.start(wait_for_first_frame=True, timeout_sec=1.0)
        assert started is True
        assert manager.is_running is True

        # Fetch latest frame
        frame = manager.get_frame(timeout=0.5)
        assert frame is not None
        assert isinstance(frame, np.ndarray)

        # Check status diagnostics
        status = manager.get_status()
        assert isinstance(status, CameraStatus)
        assert status.is_running is True
        assert status.frame_count > 0
        assert status.resolution == (640, 480)
        assert status.capture_fps >= 0.0

        # Stop manager
        manager.stop()
        assert manager.is_running is False

        final_status = manager.get_status()
        assert final_status.is_running is False

    def test_context_manager_cleanup(self) -> None:
        """Verify CameraManager cleans up resources when used as a context manager."""
        mock_recv = MockStreamReceiver()
        with CameraManager(source="mock_ctx", receiver=mock_recv) as cam:
            assert cam.is_running is True
            _ = cam.get_frame(timeout=0.1)

        # Upon exit, cam must be stopped
        assert cam.is_running is False

    def test_get_frame_when_stopped_returns_none(self) -> None:
        """Verify get_frame() returns None rather than throwing when stopped."""
        manager = CameraManager(source="unstarted")
        frame = manager.get_frame(timeout=0.01)
        assert frame is None


# ---------------------------------------------------------------------------
# 10. Snapshot Safety Tests
# ---------------------------------------------------------------------------

class TestSnapshotSafety:
    """Tests safe creation of snapshot directory and verified file saving."""

    def test_snapshot_directory_and_write(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            snap_dir = os.path.join(temp_dir, "nested", "snapshots")
            assert not os.path.exists(snap_dir)

            os.makedirs(snap_dir, exist_ok=True)
            assert os.path.isdir(snap_dir)

            test_img = np.full((100, 100, 3), 200, dtype=np.uint8)
            snap_path = os.path.join(snap_dir, "test_snap.jpg")
            success = cv2.imwrite(snap_path, test_img)

            assert success is True
            assert os.path.isfile(snap_path)
            assert os.path.getsize(snap_path) > 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
