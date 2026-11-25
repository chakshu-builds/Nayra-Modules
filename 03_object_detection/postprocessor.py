"""Object detection representations, bounding box transformations, and postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .model_registry import ASSISTIVE_CLASSES, COCO_CLASSES


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    box_normalized: Tuple[float, float, float, float]
    spatial_zone: str
    spatial_hint: str
    frame_timestamp: float
    frame_width: int
    frame_height: int

    @property
    def center_x(self) -> float:
        return (self.box_xyxy[0] + self.box_xyxy[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.box_xyxy[1] + self.box_xyxy[3]) / 2.0

    @property
    def area_pixels(self) -> int:
        return max(0, self.box_xyxy[2] - self.box_xyxy[0]) * max(0, self.box_xyxy[3] - self.box_xyxy[1])


@dataclass
class DetectionMetrics:
    inference_latency_ms: float = 0.0
    inference_fps: float = 0.0
    device_used: str = "CPU"
    gpu_memory_allocated_mb: Optional[float] = None
    gpu_memory_reserved_mb: Optional[float] = None
    num_detections: int = 0
    mean_confidence: float = 0.0


class DetectionPostprocessor:
    """Parses raw detector predictions into validated Detection instances."""

    def __init__(
        self,
        class_names: Sequence[str] = COCO_CLASSES,
        assistive_filter: Optional[Set[str]] = None,
    ) -> None:
        self.class_names = list(class_names)
        self.assistive_filter = assistive_filter

    @staticmethod
    def calculate_box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area) / float(union_area)

    @classmethod
    def deduplicate_by_class(
        cls,
        detections: List[Detection],
        iou_threshold: float = 0.65,
    ) -> List[Detection]:
        if not detections or iou_threshold <= 0.0:
            return detections

        kept: List[Detection] = []
        for det in detections:
            duplicate = False
            for prev in kept:
                if det.class_id == prev.class_id and cls.calculate_box_iou(det.box_xyxy, prev.box_xyxy) >= iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)

        return kept

    @staticmethod
    def calculate_spatial_zone(
        x1_norm: float,
        x2_norm: float,
        left_boundary: float = 0.33,
        right_boundary: float = 0.66,
    ) -> Tuple[str, str]:
        center_x = (x1_norm + x2_norm) / 2.0
        if center_x < left_boundary:
            return "left", "on the left"
        elif center_x > right_boundary:
            return "right", "on the right"
        return "center", "ahead"

    def process(
        self,
        raw_predictions: np.ndarray,
        frame_width: int,
        frame_height: int,
        frame_timestamp: float,
        confidence_threshold: float = 0.35,
        filter_assistive_only: bool = False,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        ratio: Optional[float] = None,
        pad_x: float = 0.0,
        pad_y: float = 0.0,
        iou_threshold: float = 0.65,
    ) -> List[Detection]:
        if raw_predictions is None or len(raw_predictions) == 0:
            return []

        preds = np.asarray(raw_predictions)
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        if preds.ndim != 2 or preds.shape[1] < 6:
            return []

        detections: List[Detection] = []

        for row in preds:
            x1, y1, x2, y2, score, class_idx = row[:6]
            confidence = float(score)

            if confidence < confidence_threshold or np.isnan(confidence):
                continue

            class_id = int(round(float(class_idx)))
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"object_{class_id}"

            if filter_assistive_only:
                target_set = self.assistive_filter or ASSISTIVE_CLASSES
                if class_name.lower() not in target_set:
                    continue

            # Restore original coordinates from model input space
            if ratio is not None and ratio > 0:
                orig_x1 = (x1 - pad_x) / ratio
                orig_y1 = (y1 - pad_y) / ratio
                orig_x2 = (x2 - pad_x) / ratio
                orig_y2 = (y2 - pad_y) / ratio
            else:
                orig_x1 = x1 * scale_x
                orig_y1 = y1 * scale_y
                orig_x2 = x2 * scale_x
                orig_y2 = y2 * scale_y

            clipped_x1 = max(0, min(frame_width - 1, int(round(orig_x1))))
            clipped_y1 = max(0, min(frame_height - 1, int(round(orig_y1))))
            clipped_x2 = max(0, min(frame_width, int(round(orig_x2))))
            clipped_y2 = max(0, min(frame_height, int(round(orig_y2))))

            if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
                continue

            norm_x1 = float(clipped_x1) / float(frame_width)
            norm_y1 = float(clipped_y1) / float(frame_height)
            norm_x2 = float(clipped_x2) / float(frame_width)
            norm_y2 = float(clipped_y2) / float(frame_height)

            zone, phrase = self.calculate_spatial_zone(norm_x1, norm_x2)
            display_name = class_name.replace("_", " ").capitalize()
            spatial_hint = f"{display_name} detected {phrase}."

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    box_xyxy=(clipped_x1, clipped_y1, clipped_x2, clipped_y2),
                    box_normalized=(round(norm_x1, 4), round(norm_y1, 4), round(norm_x2, 4), round(norm_y2, 4)),
                    spatial_zone=zone,
                    spatial_hint=spatial_hint,
                    frame_timestamp=frame_timestamp,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        if iou_threshold > 0.0:
            detections = self.deduplicate_by_class(detections, iou_threshold=iou_threshold)

        return detections
