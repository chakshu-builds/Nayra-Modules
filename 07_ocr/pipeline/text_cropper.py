"""NAYRA Module 07 — Perspective-Aware Text Crop Extraction.

Extracts rectified, perspective-corrected horizontal text line crops from
4-point detection polygons using OpenCV perspective transformations.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("TextCropper")


def extract_perspective_crop(
    image: np.ndarray,
    pts: np.ndarray,
    min_side: int = 3,
    auto_rotate_vertical: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Extract a perspective-rectified text crop from 4-point polygon vertices.

    Args:
        image: Original input image array (H, W, 3) in BGR uint8.
        pts: Array of shape (4, 2) or (4, 1, 2) containing clockwise corner points [TL, TR, BR, BL].
        min_side: Minimum allowed width or height in pixels (default 3).
        auto_rotate_vertical: Whether to rotate vertical crops (height > 1.5 * width)
            by 90 degrees clockwise to present horizontal text to recognizers.

    Returns:
        Tuple of (crop_image, error_message).
        If extraction succeeds, crop_image is a 2D/3D numpy array and error_message is None.
        If extraction fails or points are degenerate, returns (None, error_str).
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None, "Invalid input image: expected non-empty numpy.ndarray"

    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return None, f"Expected 4 corner points, got shape {pts.shape}"

    # Guard against NaNs or infinite values
    if not np.isfinite(pts).all():
        return None, "Polygon points contain non-finite values (NaN or Inf)"

    # Compute Euclidean edge lengths
    # pts: 0=TL, 1=TR, 2=BR, 3=BL
    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bottom = np.linalg.norm(pts[2] - pts[3])
    crop_w = int(round(max(w_top, w_bottom)))

    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])
    crop_h = int(round(max(h_left, h_right)))

    # Reject degenerate or sub-pixel crops
    if crop_w < min_side or crop_h < min_side:
        return None, f"Degenerate crop dimensions ({crop_w}x{crop_h} < min_side={min_side})"

    # Guard against unreasonable memory allocation (e.g. inverted crazy coordinates)
    img_h, img_w = image.shape[:2]
    if crop_w > img_w * 2 or crop_h > img_h * 2:
        return None, f"Crop dimensions ({crop_w}x{crop_h}) exceed plausible canvas limits"

    # Destination coordinates for rectified rectangle
    dst_pts = np.array(
        [
            [0, 0],
            [crop_w - 1, 0],
            [crop_w - 1, crop_h - 1],
            [0, crop_h - 1],
        ],
        dtype=np.float32,
    )

    try:
        matrix = cv2.getPerspectiveTransform(pts, dst_pts)
        crop = cv2.warpPerspective(
            image,
            matrix,
            (crop_w, crop_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        if crop is None or crop.size == 0:
            return None, "cv2.warpPerspective produced empty crop"

        # If crop is vertical (height significantly exceeds width), rotate 90 degrees clockwise
        if auto_rotate_vertical and crop_h >= int(crop_w * 1.5):
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        return crop, None

    except Exception as e:
        logger.warning("Perspective transform failed for box %s: %s", pts.tolist(), e)
        return None, f"Perspective transform exception: {e}"
