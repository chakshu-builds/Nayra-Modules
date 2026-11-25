"""Configuration management for NAYRA unified application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    camera_index: int | str = field(
        default_factory=lambda: os.getenv("NAYRA_STREAM_URL", "http://192.168.0.101:8080/video")
    )
    enable_camera: bool = True
    enable_voice_input: bool = True
    use_phone_mic: bool = False
    require_wake_word: bool = False
    mock_audio: bool = True
    headless: bool = False
    text_debug: bool = False
    default_language: str = "en"
    yolo_model_path: Path = Path("models/yolov10n.onnx")
    data_dir: Path = Path("data")
    reminders_db_path: Optional[Path | str] = None
    object_memory_db_path: Optional[Path | str] = None
