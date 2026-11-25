"""NAYRA Module 07 — Geometry & Coordinate Utilities for DBNet Text Detection.

Provides coordinate transformation, bounding box ordering, rotated rectangle unclip,
polygon scoring, and reading-order sorting using pure OpenCV/NumPy.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("DBNet-Geometry")


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """Order 4 2D points clockwise starting from top-left: [TL, TR, BR, BL].
    
    Args:
        pts: Array of shape (4, 2) containing corner points.
        
    Returns:
        np.ndarray of shape (4, 2) with sorted points float32.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum and difference of coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Top-left has smallest sum, Bottom-right has largest sum
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest diff (x - y > 0 -> y - x < 0), Bottom-left has largest diff
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def unclip_rotated_rect(
    rect: Tuple[Tuple[float, float], Tuple[float, float], float],
    unclip_ratio: float = 1.5,
) -> np.ndarray:
    """Expand a rotated rectangle using the DBNet unclip distance formula.

    Formula:
        Distance D = (Area * unclip_ratio) / Perimeter
        For a rectangle of dimensions (w, h):
            Area = w * h
            Perimeter = 2 * (w + h)
            D = (w * h * unclip_ratio) / (2 * (w + h))
            w_new = w + 2 * D
            h_new = h + 2 * D

    NOTE ON MATHEMATICAL DIFFERENCE FROM PYCLIPPER:
    This analytical expansion expands the fitted minimum area rectangle by distance D
    along both primary axes. In contrast, pyclipper applies Vatti polygon offset with
    circular arc bevels directly to the arbitrary N-gon contour. For rectangular text
    strips (which represent >95% of printed text lines), this rotated-rectangle expansion
    produces a clean bounding box tightly containing the unclipped text without
    introducing contour-noise artifacts or requiring unverified C-extension dependencies.

    Args:
        rect: Rotated rectangle tuple ((cx, cy), (w, h), angle_deg) from cv2.minAreaRect.
        unclip_ratio: DBNet unclip factor (default 1.5).

    Returns:
        np.ndarray of shape (4, 2) with clockwise ordered corner vertices.
    """
    (cx, cy), (w, h), angle = rect

    # Guard against degenerate rectangles
    if w <= 0 or h <= 0:
        return np.zeros((4, 2), dtype=np.float32)

    perimeter = 2.0 * (w + h)
    area = float(w * h)
    distance = (area * unclip_ratio) / max(perimeter, 1e-4)

    w_new = w + 2.0 * distance
    h_new = h + 2.0 * distance

    expanded_rect = ((cx, cy), (w_new, h_new), angle)
    box_pts = cv2.boxPoints(expanded_rect)
    return order_points_clockwise(box_pts)


def box_score_fast(prob_map: np.ndarray, box: np.ndarray) -> float:
    """Calculate the average probability inside a 4-point polygon or box.

    Uses a cropped sub-mask for O(Area) efficiency instead of rasterizing over
    the entire image canvas.

    Args:
        prob_map: 2D probability map array of shape (H, W), values in [0.0, 1.0].
        box: Array of shape (4, 2) containing polygon corner coordinates.

    Returns:
        Mean probability score float in [0.0, 1.0]. Returns 0.0 if empty.
    """
    h_map, w_map = prob_map.shape[:2]
    box = np.asarray(box, dtype=np.float32).copy()

    x_min = int(np.clip(np.floor(box[:, 0].min()), 0, w_map - 1))
    x_max = int(np.clip(np.ceil(box[:, 0].max()), 0, w_map - 1))
    y_min = int(np.clip(np.floor(box[:, 1].min()), 0, h_map - 1))
    y_max = int(np.clip(np.ceil(box[:, 1].max()), 0, h_map - 1))

    crop_w = x_max - x_min + 1
    crop_h = y_max - y_min + 1

    if crop_w <= 0 or crop_h <= 0:
        return 0.0

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    shifted_box = box.copy()
    shifted_box[:, 0] -= x_min
    shifted_box[:, 1] -= y_min

    cv2.fillPoly(mask, [shifted_box.astype(np.int32)], 1)
    cropped_prob = prob_map[y_min : y_max + 1, x_min : x_max + 1]

    mean_score = cv2.mean(cropped_prob, mask=mask)[0]
    return float(mean_score)


def polygon_to_xyxy(polygon: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a polygon (N, 2) to axis-aligned bounding box (x_min, y_min, x_max, y_max)."""
    poly = np.asarray(polygon, dtype=np.float32)
    x_min = float(poly[:, 0].min())
    y_min = float(poly[:, 1].min())
    x_max = float(poly[:, 0].max())
    y_max = float(poly[:, 1].max())
    return (x_min, y_min, x_max, y_max)


def restore_coordinates(
    points: np.ndarray,
    scale_x: float,
    scale_y: float,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Map points from preprocessed model resolution back to original image space.

    Args:
        points: Array of shape (N, 2) in model coordinates.
        scale_x: Width scale factor (model_w / orig_w).
        scale_y: Height scale factor (model_h / orig_h).
        orig_w: Original image width.
        orig_h: Original image height.

    Returns:
        np.ndarray of shape (N, 2) with coordinates mapped to [0, orig_w - 1], [0, orig_h - 1].
    """
    pts = np.asarray(points, dtype=np.float32).copy()
    if scale_x > 0:
        pts[:, 0] /= scale_x
    if scale_y > 0:
        pts[:, 1] /= scale_y

    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(max(orig_w - 1, 0)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(max(orig_h - 1, 0)))
    return pts


def sort_reading_order(
    boxes: Sequence[Tuple[float, float, float, float]],
    polygons: Optional[Sequence[np.ndarray]] = None,
    confidences: Optional[Sequence[float]] = None,
    y_threshold_ratio: float = 0.5,
) -> Tuple[List[int], List[Any], Optional[List[Any]], Optional[List[float]]]:
    """Sort text detections in standard top-to-bottom, left-to-right reading order.

    Groups detections into lines based on vertical overlap/proximity, then sorts
    lines top-to-bottom and boxes within each line left-to-right.

    Args:
        boxes: List of axis-aligned boxes (x_min, y_min, x_max, y_max).
        polygons: Optional corresponding polygon list.
        confidences: Optional corresponding confidence list.
        y_threshold_ratio: Vertical distance factor relative to box height
            to determine same line membership (default 0.5).

    Returns:
        Tuple of:
            sorted_indices: List of original indices in sorted order.
            sorted_boxes: List of sorted boxes.
            sorted_polygons: List of sorted polygons (or None if input was None).
            sorted_confidences: List of sorted confidences (or None if input was None).
    """
    n = len(boxes)
    if n <= 1:
        idx = list(range(n))
        return (
            idx,
            list(boxes),
            list(polygons) if polygons is not None else None,
            list(confidences) if confidences is not None else None,
        )

    # Compute center-y and height for each box
    items = []
    for i, b in enumerate(boxes):
        xmin, ymin, xmax, ymax = b
        cy = (ymin + ymax) / 2.0
        h = max(ymax - ymin, 1.0)
        items.append((i, xmin, ymin, xmax, ymax, cy, h))

    # Sort initially by cy
    items.sort(key=lambda it: it[5])

    # Cluster into lines
    lines: List[List[Tuple[int, float, float, float, float, float, float]]] = []
    current_line: List[Tuple[int, float, float, float, float, float, float]] = []

    for item in items:
        if not current_line:
            current_line.append(item)
            continue

        # Compare with median cy of current line
        line_cy = float(np.median([it[5] for it in current_line]))
        line_h = float(np.median([it[6] for it in current_line]))
        y_thresh = line_h * y_threshold_ratio

        if abs(item[5] - line_cy) <= y_thresh:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]

    if current_line:
        lines.append(current_line)

    # Sort lines by median y, and within each line sort by xmin
    sorted_indices: List[int] = []
    for line in lines:
        line.sort(key=lambda it: it[1])  # sort by xmin (left to right)
        for it in line:
            sorted_indices.append(it[0])

    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_polys = [polygons[i] for i in sorted_indices] if polygons is not None else None
    sorted_confs = [confidences[i] for i in sorted_indices] if confidences is not None else None

    return sorted_indices, sorted_boxes, sorted_polys, sorted_confs
