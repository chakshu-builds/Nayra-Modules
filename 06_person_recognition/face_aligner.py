"""Facial landmark alignment and quality validation for feature extraction."""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from recognition_types import FaceDetection

logger = logging.getLogger(__name__)

# Standard 112x112 canonical face landmark reference coordinates
CANONICAL_5_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],  # right eye
        [73.5318, 51.6963],  # left eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # right mouth corner
        [70.7299, 92.3655],  # left mouth corner
    ],
    dtype=np.float32,
)


class FaceAligner:
    """Extracts, aligns, and validates 112x112 face crops using 5-point facial landmarks."""

    def __init__(self, target_size: tuple[int, int] = (112, 112)) -> None:
        self.target_size = target_size

    def align(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray | None:
        """Align face in frame to target_size using similarity transform from 5 landmarks."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        src_pts = np.array(detection.landmarks_5, dtype=np.float32)
        if src_pts.shape != (5, 2):
            return None

        # Estimate 2D affine similarity transformation
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, CANONICAL_5_LANDMARKS_112, method=cv2.LMEDS)
        if matrix is None:
            # Fallback to simple bounding box crop and resize if matrix estimation fails
            x1, y1, x2, y2 = detection.box_xyxy
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            return cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)

        aligned = cv2.warpAffine(
            frame,
            matrix,
            self.target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned

    def check_quality(self, face_crop: np.ndarray) -> tuple[bool, str]:
        """Validate face crop sharpness, illumination, and non-degeneracy."""
        if face_crop is None or face_crop.size == 0:
            return False, "Empty face crop"

        if face_crop.shape[:2] != self.target_size:
            return False, f"Invalid crop dimensions {face_crop.shape[:2]}"

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop

        # 1. Illumination check
        mean_intensity = float(np.mean(gray))
        if mean_intensity < 25.0:
            return False, "Face is too dark"
        if mean_intensity > 240.0:
            return False, "Face is overexposed"

        # 2. Sharpness / blur check using Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < 30.0:
            return False, f"Face crop too blurry (Laplacian var {laplacian_var:.1f} < 30)"

        return True, "Valid quality"
