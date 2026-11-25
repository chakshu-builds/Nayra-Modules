"""Obstacle candidate identification and proximity classification."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from shared.schemas import ObstacleHazard

logger = logging.getLogger("NAYRA-ObstacleDetector")

# Common physical obstacle classes that impede walking or navigation
PHYSICAL_OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "bench", "chair", "couch", "bed", "dining table", "toilet", "tv",
    "refrigerator", "suitcase", "backpack", "door", "stairs", "table", "desk",
    "potted plant", "traffic light", "fire hydrant", "stop sign"
}

# Small handheld or tabletop items that do NOT constitute navigational ground obstacles
NON_OBSTACLE_CLASSES = {
    "cell phone", "remote", "mouse", "keyboard", "cup", "bottle", "wine glass",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush", "book", "clock"
}

# Hindi names for common obstacle classes
OBSTACLE_HINDI_NAMES = {
    "person": "person",
    "chair": "chair",
    "couch": "sofa",
    "sofa": "sofa",
    "dining table": "table",
    "table": "table",
    "desk": "table",
    "bed": "bistar",
    "door": "darwaza",
    "stairs": "seedhiyan",
    "bicycle": "cycle",
    "car": "gaadi",
    "motorcycle": "bike",
    "bench": "bench",
    "refrigerator": "fridge",
    "potted plant": "gamla",
    "suitcase": "suitcase",
    "backpack": "bag",
}


class ObstacleDetector:
    """Detects, zones, and categorizes navigational obstacles."""

    def __init__(
        self,
        near_threshold_depth: float = 0.65,
        medium_threshold_depth: float = 0.35,
        zone_left_boundary: float = 0.33,
        zone_right_boundary: float = 0.66,
    ) -> None:
        self.near_threshold = near_threshold_depth
        self.medium_threshold = medium_threshold_depth
        self.zone_left = zone_left_boundary
        self.zone_right = zone_right_boundary

    def classify_zone(self, box_xyxy: Tuple[int, int, int, int], frame_width: int) -> str:
        """Estimate horizontal zone: LEFT, CENTER, or RIGHT."""
        w = max(1, frame_width)
        cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
        norm_cx = cx / w
        if norm_cx < self.zone_left:
            return "LEFT"
        elif norm_cx > self.zone_right:
            return "RIGHT"
        return "CENTER"

    def estimate_proximity(
        self,
        box_xyxy: Tuple[int, int, int, int],
        frame_height: int,
        depth_map: Optional[np.ndarray] = None,
    ) -> Tuple[str, float]:
        """Estimate relative proximity category (NEAR, MEDIUM, FAR) and normalized depth [0.0, 1.0]."""
        # If monocular depth map is provided, sample inner-ROI median
        if depth_map is not None and depth_map.size > 0:
            h, w = depth_map.shape[:2]
            x1 = max(0, min(w - 1, int(box_xyxy[0])))
            y1 = max(0, min(h - 1, int(box_xyxy[1])))
            x2 = max(0, min(w, int(box_xyxy[2])))
            y2 = max(0, min(h, int(box_xyxy[3])))
            # Sample central 60% of ROI to reject edge bleed
            rw = x2 - x1
            rh = y2 - y1
            if rw > 4 and rh > 4:
                ix1 = int(x1 + rw * 0.2)
                ix2 = int(x2 - rw * 0.2)
                iy1 = int(y1 + rh * 0.2)
                iy2 = int(y2 - rh * 0.2)
                inner = depth_map[iy1:iy2, ix1:ix2]
                if inner.size > 0:
                    med_val = float(np.median(inner))
                    rel_d = max(0.0, min(1.0, med_val / 255.0 if depth_map.dtype == np.uint8 else med_val))
                    if rel_d >= self.near_threshold:
                        return "NEAR", rel_d
                    elif rel_d >= self.medium_threshold:
                        return "MEDIUM", rel_d
                    return "FAR", rel_d

        # Fallback: geometric heuristic based on bottom-edge vertical position & bounding box height
        h = max(1, frame_height)
        y2 = box_xyxy[3]
        box_h = box_xyxy[3] - box_xyxy[1]
        norm_bottom = y2 / h
        norm_h = box_h / h

        # Ground obstacles closer to user appear lower in the frame and occupy larger vertical height
        heuristic_depth = 0.6 * norm_bottom + 0.4 * norm_h
        heuristic_depth = max(0.0, min(1.0, heuristic_depth))

        if heuristic_depth >= 0.65:
            return "NEAR", heuristic_depth
        elif heuristic_depth >= 0.40:
            return "MEDIUM", heuristic_depth
        return "FAR", heuristic_depth

    def evaluate_obstacles(
        self,
        detections: Sequence[Any],
        tracks: Optional[Sequence[Any]] = None,
        frame_width: int = 640,
        frame_height: int = 480,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[ObstacleHazard]:
        """Filter detections/tracks into validated physical obstacle hazards."""
        hazards: List[ObstacleHazard] = []

        # Prefer active tracks if available; fallback to raw detections
        items = tracks if (tracks is not None and len(tracks) > 0) else detections

        for item in items:
            cname = getattr(item, "class_name", "").lower().strip()
            if not cname:
                continue

            # Filter for physical obstacles only
            if cname in NON_OBSTACLE_CLASSES or cname not in PHYSICAL_OBSTACLE_CLASSES:
                continue

            box = getattr(item, "box_xyxy", None)
            if box is None:
                continue

            track_id = getattr(item, "track_id", None)
            zone = self.classify_zone(box, frame_width)
            prox, rel_d = self.estimate_proximity(box, frame_height, depth_map)

            # Hazard is urgent if directly ahead (CENTER) and NEAR
            is_urgent = (zone == "CENTER" and prox == "NEAR")

            hi_name = OBSTACLE_HINDI_NAMES.get(cname, cname)

            # Generate structured warning phrases
            if is_urgent:
                warn_hi = f"Savdhaan, dhyan se chaliye, aapke saamne {hi_name} hai."
                warn_en = f"Move carefully, you have a {cname} in your front."
            elif zone == "CENTER":
                warn_hi = f"Saamne {hi_name} hai, sambhal kar chaliye."
                warn_en = f"Move carefully, there is a {cname} ahead."
            elif zone == "LEFT":
                warn_hi = f"Aapke left side mein {hi_name} hai."
                warn_en = f"There is a {cname} on your left."
            else:
                warn_hi = f"Aapke right side mein {hi_name} hai."
                warn_en = f"There is a {cname} on your right."

            hazards.append(
                ObstacleHazard(
                    track_id=track_id,
                    class_name=cname,
                    zone=zone,
                    proximity=prox,
                    relative_depth=round(rel_d, 3),
                    is_urgent=is_urgent,
                    warning_hi=warn_hi,
                    warning_en=warn_en,
                )
            )

        # Sort hazards by urgency first, then relative proximity
        hazards.sort(key=lambda h: (h.is_urgent, h.relative_depth), reverse=True)
        return hazards
