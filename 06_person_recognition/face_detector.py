"""Lightweight face and landmark detector using YuNet."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from config.settings import PersonRecognitionSettings, get_default_person_recognition_settings
from recognition_types import FaceDetection

logger = logging.getLogger(__name__)

YUNET_DEFAULT_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    """YuNet lightweight face detector with 5-point facial landmark extraction."""

    def __init__(self, settings: PersonRecognitionSettings | None = None) -> None:
        self.settings = settings or get_default_person_recognition_settings()
        self.model_path = self._resolve_model_path()

        self._detector: cv2.FaceDetectorYN | None = None
        self._current_input_size: tuple[int, int] = (320, 320)
        self._load_model()

        logger.info(
            "FaceDetector initialized (%s, conf>=%.2f, min_size=%d)",
            self.model_path.name,
            self.settings.score_threshold,
            self.settings.min_face_size,
        )

    def _resolve_model_path(self) -> Path:
        if self.settings.detector_model_path and Path(self.settings.detector_model_path).exists():
            return Path(self.settings.detector_model_path)

        models_dir = Path(self.settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        target_path = models_dir / f"{self.settings.detector_model_name}.onnx"

        if not target_path.exists():
            logger.info("Downloading YuNet face detection weights to %s...", target_path)
            try:
                urllib.request.urlretrieve(YUNET_DEFAULT_URL, str(target_path))
            except Exception as dl_err:
                logger.error("Failed to download YuNet model: %s", dl_err)
                raise FileNotFoundError(f"Could not download YuNet from {YUNET_DEFAULT_URL}: {dl_err}") from dl_err

        return target_path

    def _load_model(self) -> None:
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path.resolve()),
                config="",
                input_size=self._current_input_size,
                score_threshold=float(self.settings.score_threshold),
                nms_threshold=float(self.settings.nms_threshold),
                top_k=5000,
            )
        except Exception as err:
            logger.error("Failed to initialize YuNet face detector: %s", err)
            raise RuntimeError(f"YuNet initialization failed: {err}") from err

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces in BGR frame with bounding boxes and 5 landmarks."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if w < 10 or h < 10:
            return []

        if self._current_input_size != (w, h):
            self._current_input_size = (w, h)
            self._detector.setInputSize(self._current_input_size)

        retval, raw_faces = self._detector.detect(frame)
        if retval != 1 or raw_faces is None or len(raw_faces) == 0:
            return []

        valid_detections: list[FaceDetection] = []
        for face_row in raw_faces:
            conf = float(face_row[14])
            if conf < self.settings.score_threshold:
                continue

            x, y, bw, bh = int(round(face_row[0])), int(round(face_row[1])), int(round(face_row[2])), int(round(face_row[3]))

            # Validate dimensions
            if bw < self.settings.min_face_size or bh < self.settings.min_face_size:
                continue

            # Clip bounding box to frame boundaries
            x1 = max(0, min(w - 1, x))
            y1 = max(0, min(h - 1, y))
            x2 = max(x1 + 1, min(w, x + bw))
            y2 = max(y1 + 1, min(h, y + bh))
            box_xyxy = (x1, y1, x2, y2)
            box_xywh = (x1, y1, x2 - x1, y2 - y1)

            # 5 landmarks: (re, le, nt, rcm, lcm)
            landmarks: list[tuple[float, float]] = [
                (float(face_row[4]), float(face_row[5])),
                (float(face_row[6]), float(face_row[7])),
                (float(face_row[8]), float(face_row[9])),
                (float(face_row[10]), float(face_row[11])),
                (float(face_row[12]), float(face_row[13])),
            ]

            valid_detections.append(
                FaceDetection(
                    box_xywh=box_xywh,
                    box_xyxy=box_xyxy,
                    landmarks_5=landmarks,
                    confidence=round(conf, 4),
                    raw_array=[float(v) for v in face_row],
                )
            )

        return valid_detections
