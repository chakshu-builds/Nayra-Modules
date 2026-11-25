"""NAYRA Module 07 — Postprocessing for DBNet Text Detection.

Extracts text polygons and oriented bounding boxes from predicted probability maps,
applies confidence scoring, candidate filtering, unclip expansion, and coordinate
restoration to the original image canvas.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import (
    box_score_fast,
    order_points_clockwise,
    polygon_to_xyxy,
    restore_coordinates,
    unclip_rotated_rect,
)

logger = logging.getLogger("DBNet-Postprocess")


class DBPostProcessor:
    """Postprocessor for Differentiable Binarization (DBNet) text detection output.

    Attributes:
        thresh: Binarization threshold on the continuous probability map (default 0.3).
        box_thresh: Minimum average probability inside the bounding polygon (default 0.6).
        unclip_ratio: Factor to expand the detected text region (default 1.5).
        max_candidates: Maximum number of contours to evaluate per frame (default 1000).
        min_size: Minimum width or height for valid text candidate in pixels (default 3).
    """

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        min_size: int = 3,
    ) -> None:
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.unclip_ratio = float(unclip_ratio)
        self.max_candidates = int(max_candidates)
        self.min_size = int(min_size)

    def __call__(
        self,
        prob_map: np.ndarray,
        scale_x: float,
        scale_y: float,
        orig_shape: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[float, float, float, float]], List[float]]:
        """Process model output probability map into bounding boxes in original image coordinates.

        Args:
            prob_map: 2D or 3D numpy array representing the predicted probability map in [0, 1].
                Shapes accepted: (H, W), (1, H, W), or (1, 1, H, W).
            scale_x: Width scaling factor used in preprocessing (model_w / orig_w).
            scale_y: Height scaling factor used in preprocessing (model_h / orig_h).
            orig_shape: (orig_h, orig_w) dimensions of the original unscaled image.

        Returns:
            Tuple of:
                polygons: List of (N, 2) contour points in original coordinates.
                boxes_4pt: List of (4, 2) clockwise-ordered vertices in original coordinates.
                boxes_xyxy: List of (x_min, y_min, x_max, y_max) bounding rectangles.
                confidences: List of float scores in [0.0, 1.0].
        """
        orig_h, orig_w = orig_shape

        # Squeeze down to 2D (H, W)
        pred = np.asarray(prob_map, dtype=np.float32)
        while pred.ndim > 2:
            pred = pred[0]

        h_map, w_map = pred.shape[:2]

        # Fast rejection if map is empty or completely below threshold
        if pred.size == 0 or float(np.max(pred)) < self.thresh:
            return [], [], [], []

        # Step 1: Binarize probability map
        binary_mask = (pred > self.thresh).astype(np.uint8) * 255

        # Step 2: Extract contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return [], [], [], []

        # Sort by contour area descending and limit candidates
        if len(contours) > self.max_candidates:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[: self.max_candidates]

        polygons: List[np.ndarray] = []
        boxes_4pt: List[np.ndarray] = []
        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        confidences: List[float] = []

        for cnt in contours:
            # Skip degenerate or point-like contours
            if len(cnt) < 3 or cv2.contourArea(cnt) < 4:
                continue

            # Fit rotated rectangle
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            min_side = min(w, h)

            # Minimum size check (must exceed min_size in model space)
            if min_side < self.min_size:
                continue

            # Step 3: Raw bounding box vertices for scoring
            raw_box = cv2.boxPoints(rect)

            # Step 4: Polygon confidence scoring on core region before unclip
            # (Matches official DBNet where scoring evaluates the shrunken text core)
            score = box_score_fast(pred, raw_box)
            if score < self.box_thresh:
                continue

            # Step 5: Unclip rotated rectangle
            unclipped_box = unclip_rotated_rect(rect, self.unclip_ratio)
            if unclipped_box.shape[0] != 4:
                continue

            # Step 6: Restore coordinates back to original frame space
            restored_box = restore_coordinates(unclipped_box, scale_x, scale_y, orig_w, orig_h)
            restored_cnt = restore_coordinates(cnt.reshape(-1, 2), scale_x, scale_y, orig_w, orig_h)

            xyxy = polygon_to_xyxy(restored_box)

            # Guard against invalid or zero-area boxes after restoration
            if (xyxy[2] - xyxy[0]) <= 1 or (xyxy[3] - xyxy[1]) <= 1:
                continue

            polygons.append(restored_cnt)
            boxes_4pt.append(restored_box)
            boxes_xyxy.append(xyxy)
            confidences.append(float(score))

        return polygons, boxes_4pt, boxes_xyxy, confidences
