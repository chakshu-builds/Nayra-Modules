"""Application configuration dataclasses with environment variable defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Union


def _get_env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    val = os.getenv(key)
    if val is not None and val.strip().isdigit():
        return int(val.strip())
    return default


def _get_env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    val = os.getenv(key)
    if val is not None:
        try:
            return float(val.strip())
        except ValueError:
            pass
    return default


def _get_env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is not None:
        return val.strip().lower() in ("1", "true", "yes")
    return default


@dataclass
class CameraSettings:
    stream_url: Union[str, int] = field(
        default_factory=lambda: os.getenv("NAYRA_STREAM_URL", "http://192.168.0.101:8080/video")
    )
    target_width: Optional[int] = field(default_factory=lambda: _get_env_int("NAYRA_CAMERA_WIDTH"))
    target_height: Optional[int] = field(default_factory=lambda: _get_env_int("NAYRA_CAMERA_HEIGHT"))
    target_fps: Optional[float] = field(default_factory=lambda: _get_env_float("NAYRA_TARGET_FPS"))
    reconnect_interval_sec: float = field(
        default_factory=lambda: _get_env_float("NAYRA_RECONNECT_INTERVAL", 2.0) or 2.0
    )
    max_reconnect_attempts: int = field(
        default_factory=lambda: _get_env_int("NAYRA_MAX_RECONNECT_ATTEMPTS", 0) or 0
    )
    connection_timeout_sec: float = field(
        default_factory=lambda: _get_env_float("NAYRA_CONNECTION_TIMEOUT", 5.0) or 5.0
    )
    frame_buffer_size: int = field(
        default_factory=lambda: _get_env_int("NAYRA_BUFFER_SIZE", 1) or 1
    )
    enable_auto_reconnect: bool = field(
        default_factory=lambda: _get_env_bool("NAYRA_AUTO_RECONNECT", True)
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("NAYRA_LOG_LEVEL", "INFO").upper()
    )

    def __post_init__(self) -> None:
        if isinstance(self.stream_url, str) and self.stream_url.strip().isdigit():
            self.stream_url = int(self.stream_url.strip())
        if self.target_width is not None and self.target_width <= 0:
            raise ValueError(f"target_width must be positive, got {self.target_width}")
        if self.target_height is not None and self.target_height <= 0:
            raise ValueError(f"target_height must be positive, got {self.target_height}")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if self.reconnect_interval_sec < 0.1:
            raise ValueError(f"reconnect_interval_sec must be >= 0.1s, got {self.reconnect_interval_sec}")
        if self.connection_timeout_sec <= 0:
            raise ValueError(f"connection_timeout_sec must be positive, got {self.connection_timeout_sec}")
        if self.frame_buffer_size < 1:
            raise ValueError(f"frame_buffer_size must be >= 1, got {self.frame_buffer_size}")


@dataclass
class ObjectDetectionSettings:
    model_name: str = field(
        default_factory=lambda: os.getenv("NAYRA_DETECTOR_MODEL", "yolov10n")
    )
    model_path: Optional[str] = field(
        default_factory=lambda: os.getenv("NAYRA_DETECTOR_PATH")
    )
    confidence_threshold: float = field(
        default_factory=lambda: _get_env_float("NAYRA_CONFIDENCE_THRESHOLD", 0.35) or 0.35
    )
    device: str = field(
        default_factory=lambda: os.getenv("NAYRA_INFERENCE_DEVICE", "auto").lower()
    )
    models_dir: str = field(
        default_factory=lambda: os.getenv("NAYRA_MODELS_DIR", "models")
    )
    input_width: int = field(
        default_factory=lambda: _get_env_int("NAYRA_INPUT_WIDTH", 640) or 640
    )
    input_height: int = field(
        default_factory=lambda: _get_env_int("NAYRA_INPUT_HEIGHT", 640) or 640
    )
    filter_assistive_only: bool = field(
        default_factory=lambda: _get_env_bool("NAYRA_ASSISTIVE_ONLY", False)
    )
    iou_threshold: float = field(
        default_factory=lambda: _get_env_float("NAYRA_IOU_THRESHOLD", 0.65) or 0.65
    )
    letterbox: bool = field(
        default_factory=lambda: _get_env_bool("NAYRA_LETTERBOX", True)
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}")
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {self.iou_threshold}")
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError(f"Input dimensions must be positive, got {self.input_width}x{self.input_height}")
        valid_devices = ("auto", "cuda", "cpu")
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of {valid_devices} or 'cuda:X', got {self.device}")


@dataclass
class ObjectTrackingSettings:
    track_thresh: float = field(
        default_factory=lambda: _get_env_float("NAYRA_TRACK_THRESH", 0.45) or 0.45
    )
    low_thresh: float = field(
        default_factory=lambda: _get_env_float("NAYRA_TRACK_LOW_THRESH", 0.15) or 0.15
    )
    match_thresh: float = field(
        default_factory=lambda: _get_env_float("NAYRA_TRACK_MATCH_THRESH", 0.70) or 0.70
    )
    match_thresh_second: float = field(
        default_factory=lambda: _get_env_float("NAYRA_TRACK_MATCH_THRESH_SECOND", 0.50) or 0.50
    )
    track_buffer: int = field(
        default_factory=lambda: _get_env_int("NAYRA_TRACK_BUFFER", 30) or 30
    )
    min_hits: int = field(
        default_factory=lambda: _get_env_int("NAYRA_TRACK_MIN_HITS", 2) or 2
    )
    stationary_threshold_px: float = field(
        default_factory=lambda: _get_env_float("NAYRA_TRACK_STATIONARY_THRESH", 6.0) or 6.0
    )
    enforce_class_consistency: bool = field(
        default_factory=lambda: _get_env_bool("NAYRA_TRACK_CLASS_CONSISTENCY", True)
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.low_thresh <= self.track_thresh <= 1.0):
            raise ValueError(f"Expected 0.0 <= low_thresh ({self.low_thresh}) <= track_thresh ({self.track_thresh}) <= 1.0")
        if not (0.0 <= self.match_thresh <= 1.0):
            raise ValueError(f"match_thresh must be in [0.0, 1.0], got {self.match_thresh}")
        if not (0.0 <= self.match_thresh_second <= 1.0):
            raise ValueError(f"match_thresh_second must be in [0.0, 1.0], got {self.match_thresh_second}")
        if self.track_buffer < 1:
            raise ValueError(f"track_buffer must be >= 1, got {self.track_buffer}")
        if self.min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {self.min_hits}")
        if self.stationary_threshold_px < 0.0:
            raise ValueError(f"stationary_threshold_px must be >= 0.0, got {self.stationary_threshold_px}")


@dataclass
class SpatialAwarenessSettings:
    model_name: str = field(
        default_factory=lambda: os.getenv("NAYRA_DEPTH_MODEL", "midas_v21_small")
    )
    model_path: Optional[str] = field(
        default_factory=lambda: os.getenv("NAYRA_DEPTH_MODEL_PATH")
    )
    device: str = field(
        default_factory=lambda: os.getenv("NAYRA_DEPTH_DEVICE", "auto").lower()
    )
    models_dir: str = field(
        default_factory=lambda: os.getenv("NAYRA_MODELS_DIR", "models")
    )
    depth_interval: int = field(
        default_factory=lambda: _get_env_int("NAYRA_DEPTH_INTERVAL", 1) or 1
    )
    zone_left_boundary: float = field(
        default_factory=lambda: _get_env_float("NAYRA_ZONE_LEFT_BOUNDARY", 0.33) or 0.33
    )
    zone_right_boundary: float = field(
        default_factory=lambda: _get_env_float("NAYRA_ZONE_RIGHT_BOUNDARY", 0.66) or 0.66
    )
    depth_crop_ratio: float = field(
        default_factory=lambda: _get_env_float("NAYRA_DEPTH_CROP_RATIO", 0.60) or 0.60
    )
    near_threshold: float = field(
        default_factory=lambda: _get_env_float("NAYRA_PROXIMITY_NEAR_THRESH", 0.65) or 0.65
    )
    far_threshold: float = field(
        default_factory=lambda: _get_env_float("NAYRA_PROXIMITY_FAR_THRESH", 0.35) or 0.35
    )
    min_obstacle_box_area_ratio: float = field(
        default_factory=lambda: _get_env_float("NAYRA_OBSTACLE_MIN_AREA", 0.015) or 0.015
    )
    show_depth_map: bool = field(
        default_factory=lambda: _get_env_bool("NAYRA_SHOW_DEPTH_MAP", True)
    )
    depth_pip_scale: float = field(
        default_factory=lambda: _get_env_float("NAYRA_DEPTH_PIP_SCALE", 0.25) or 0.25
    )

    def __post_init__(self) -> None:
        if not (0.0 < self.zone_left_boundary < self.zone_right_boundary < 1.0):
            raise ValueError(
                f"Zone boundaries must satisfy 0.0 < left ({self.zone_left_boundary}) < "
                f"right ({self.zone_right_boundary}) < 1.0"
            )
        if not (0.0 < self.depth_crop_ratio <= 1.0):
            raise ValueError(f"depth_crop_ratio must be in (0.0, 1.0], got {self.depth_crop_ratio}")
        if not (0.0 <= self.far_threshold < self.near_threshold <= 1.0):
            raise ValueError(
                f"Proximity thresholds must satisfy 0.0 <= far ({self.far_threshold}) < "
                f"near ({self.near_threshold}) <= 1.0"
            )
        if self.depth_interval < 1:
            raise ValueError(f"depth_interval must be >= 1, got {self.depth_interval}")
        if not (0.0 < self.depth_pip_scale <= 0.5):
            raise ValueError(f"depth_pip_scale must be in (0.0, 0.5], got {self.depth_pip_scale}")
        valid_devices = ("auto", "cuda", "cpu")
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of {valid_devices} or 'cuda:X', got {self.device}")


@dataclass
class PersonRecognitionSettings:
    enabled: bool = field(
        default_factory=lambda: _get_env_bool("NAYRA_FACE_REC_ENABLED", True)
    )
    detector_model_name: str = "face_detection_yunet_2023mar"
    detector_model_path: Optional[str] = field(
        default_factory=lambda: os.getenv("NAYRA_FACE_DETECTOR_PATH")
    )
    recognizer_model_name: str = "face_recognition_sface_2021dec"
    recognizer_model_path: Optional[str] = field(
        default_factory=lambda: os.getenv("NAYRA_FACE_RECOGNIZER_PATH")
    )
    models_dir: str = field(
        default_factory=lambda: os.getenv("NAYRA_MODELS_DIR", "models")
    )
    device: str = field(
        default_factory=lambda: os.getenv("NAYRA_FACE_REC_DEVICE", "auto").lower()
    )
    score_threshold: float = field(
        default_factory=lambda: _get_env_float("NAYRA_FACE_SCORE_THRESH", 0.60) or 0.60
    )
    nms_threshold: float = field(
        default_factory=lambda: _get_env_float("NAYRA_FACE_NMS_THRESH", 0.30) or 0.30
    )
    recognition_threshold: float = field(
        default_factory=lambda: _get_env_float("NAYRA_FACE_RECOGNITION_THRESH", 0.40) or 0.40
    )
    ambiguity_margin: float = field(
        default_factory=lambda: _get_env_float("NAYRA_FACE_AMBIGUITY_MARGIN", 0.05) or 0.05
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("NAYRA_FACE_DB_PATH", "data/face_db.bin")
    )
    key_path: str = field(
        default_factory=lambda: os.getenv("NAYRA_FACE_KEY_PATH", "data/face_db.key")
    )
    context_db_path: str = field(
        default_factory=lambda: os.getenv("NAYRA_CONTEXT_DB_PATH", "data/person_context.db")
    )
    recognition_interval: int = field(
        default_factory=lambda: _get_env_int("NAYRA_FACE_REC_INTERVAL", 2) or 2
    )
    announcement_cooldown_sec: float = field(
        default_factory=lambda: _get_env_float("NAYRA_FACE_ANNOUNCE_COOLDOWN", 10.0) or 10.0
    )
    min_face_size: int = field(
        default_factory=lambda: _get_env_int("NAYRA_FACE_MIN_SIZE", 48) or 48
    )
    min_samples_per_enrollment: int = field(
        default_factory=lambda: _get_env_int("NAYRA_FACE_MIN_SAMPLES", 3) or 3
    )
    max_samples_per_enrollment: int = field(
        default_factory=lambda: _get_env_int("NAYRA_FACE_MAX_SAMPLES", 10) or 10
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.score_threshold <= 1.0):
            raise ValueError(f"score_threshold must be in [0.0, 1.0], got {self.score_threshold}")
        if not (0.0 <= self.nms_threshold <= 1.0):
            raise ValueError(f"nms_threshold must be in [0.0, 1.0], got {self.nms_threshold}")
        if not (0.0 <= self.recognition_threshold <= 1.0):
            raise ValueError(f"recognition_threshold must be in [0.0, 1.0], got {self.recognition_threshold}")
        if not (0.0 <= self.ambiguity_margin <= 0.5):
            raise ValueError(f"ambiguity_margin must be in [0.0, 0.5], got {self.ambiguity_margin}")
        if self.recognition_interval < 1:
            raise ValueError(f"recognition_interval must be >= 1, got {self.recognition_interval}")
        if self.min_face_size < 16:
            raise ValueError(f"min_face_size must be >= 16, got {self.min_face_size}")
        if self.min_samples_per_enrollment < 1:
            raise ValueError(f"min_samples_per_enrollment must be >= 1, got {self.min_samples_per_enrollment}")
        if self.max_samples_per_enrollment < self.min_samples_per_enrollment:
            raise ValueError(f"max_samples must be >= min_samples, got {self.max_samples_per_enrollment}")
        valid_devices = ("auto", "cuda", "cpu")
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of {valid_devices} or 'cuda:X', got {self.device}")


def get_default_settings() -> CameraSettings:
    return CameraSettings()


def get_default_detection_settings() -> ObjectDetectionSettings:
    return ObjectDetectionSettings()


def get_default_tracking_settings() -> ObjectTrackingSettings:
    return ObjectTrackingSettings()


def get_default_spatial_settings() -> SpatialAwarenessSettings:
    return SpatialAwarenessSettings()


def get_default_person_recognition_settings() -> PersonRecognitionSettings:
    return PersonRecognitionSettings()
