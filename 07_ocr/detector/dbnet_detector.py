"""NAYRA Module 07 — DBNet Text Detection Implementation.

Provides real-time multilingual text line detection using the audited
lightweight PP-OCRv4 Mobile DBNet ONNX checkpoint on NVIDIA RTX 4050
GPU with automatic CPU fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

from .geometry import sort_reading_order
from .postprocess import DBPostProcessor

logger = logging.getLogger("DBNet-Detector")

# Default ImageNet normalization constants (BGR channel order)
DEFAULT_MEAN_BGR = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD_BGR = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def configure_cuda_dlls() -> bool:
    """Ensure Windows can locate CUDA/cuDNN DLLs from PyTorch distribution."""
    try:
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            try:
                os.add_dll_directory(str(torch_lib.resolve()))
                os.environ["PATH"] = f"{torch_lib.resolve()};{os.environ.get('PATH', '')}"
            except (AttributeError, OSError):
                pass
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


@dataclass
class DetectionResult:
    """Structured detection result containing bounding geometries and execution metadata.

    Attributes:
        original_shape: (H, W) dimensions of the input image.
        polygons: List of contour point arrays (N, 2) in original image coordinates.
        boxes_4pt: List of 4-point bounding boxes (4, 2) ordered clockwise [TL, TR, BR, BL].
        boxes_xyxy: List of axis-aligned bounding rectangles (x_min, y_min, x_max, y_max).
        confidences: List of detection confidence scores in [0.0, 1.0].
        num_detections: Total number of accepted text regions detected.
        pre_time_ms: Preprocessing elapsed time in milliseconds.
        infer_time_ms: ONNX model inference time in milliseconds.
        post_time_ms: Postprocessing elapsed time in milliseconds.
        total_time_ms: Total end-to-end pipeline latency in milliseconds.
        execution_provider: The active ONNX Runtime provider (e.g. 'CUDAExecutionProvider').
        success: Boolean flag indicating if detection completed successfully.
        error_message: Optional error message if detection failed.
    """

    original_shape: Tuple[int, int]
    polygons: List[np.ndarray] = field(default_factory=list)
    boxes_4pt: List[np.ndarray] = field(default_factory=list)
    boxes_xyxy: List[Tuple[float, float, float, float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    num_detections: int = 0
    pre_time_ms: float = 0.0
    infer_time_ms: float = 0.0
    post_time_ms: float = 0.0
    total_time_ms: float = 0.0
    execution_provider: str = "Unknown"
    success: bool = True
    error_message: Optional[str] = None


class DBNetDetector:
    """Differentiable Binarization text detector using PP-OCRv4 ONNX checkpoint.

    Attributes:
        model_path: Path to the ch_PP-OCRv4_det.onnx checkpoint.
        max_side_len: Maximum dimension for aspect-preserving dynamic resize (default 960).
        preferred_provider: Preferred ORT provider ('CUDA' or 'CPU').
        sort_reading_order: Whether to sort detected boxes in top-to-bottom reading order.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        preferred_provider: str = "CUDA",
        max_side_len: int = 960,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        min_size: int = 3,
        sort_reading_order: bool = True,
    ) -> None:
        self.model_path = Path(model_path or "models/ocr/ch_PP-OCRv4_det.onnx")
        self.config_path = Path(config_path or "models/ocr/ch_PP-OCRv4_det.yml")
        self.max_side_len = int(max_side_len)
        self.sort_reading_order_enabled = bool(sort_reading_order)

        if not self.model_path.exists():
            raise FileNotFoundError(f"DBNet model checkpoint not found at: {self.model_path}")

        # Postprocessor
        self.postprocessor = DBPostProcessor(
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            max_candidates=max_candidates,
            min_size=min_size,
        )

        # Initialize ONNX Runtime session
        self._init_session(preferred_provider)

    def _init_session(self, preferred_provider: str) -> None:
        """Create and configure the persistent ONNX Runtime inference session."""
        cuda_ready = configure_cuda_dlls()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Suppress verbose initializers warnings from clean export
        sess_options.log_severity_level = 3

        if preferred_provider.upper() == "CUDA":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize requested providers %s: %s. Falling back to CPU.",
                providers,
                e,
            )
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

        active = self.session.get_providers()
        self.active_provider = active[0] if active else "CPUExecutionProvider"
        logger.info(
            "DBNet initialized with active provider: %s (all: %s, PyTorch CUDA: %s)",
            self.active_provider,
            active,
            cuda_ready,
        )

        # Cache tensor names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Preprocess an OpenCV BGR image for DBNet inference.

        Operations:
          1. Aspect-ratio preserving resize where max(H, W) <= max_side_len.
          2. Round both dimensions to multiples of 32.
          3. Normalize with ImageNet mean/std (BGR order).
          4. Convert HWC to NCHW float32.

        Args:
            image: OpenCV BGR uint8 image of shape (H, W, 3).

        Returns:
            Tuple of:
                tensor: NCHW float32 array ready for inference.
                scales: (scale_x, scale_y) factors used for resizing.
                orig_shape: (orig_h, orig_w) original pixel dimensions.
        """
        orig_h, orig_w = image.shape[:2]
        max_side = max(orig_h, orig_w)

        # Scale factor calculation
        ratio = 1.0
        if max_side > self.max_side_len:
            ratio = float(self.max_side_len) / float(max_side)

        # Dimensions must be positive multiples of 32
        resize_h = max(int(round(orig_h * ratio / 32) * 32), 32)
        resize_w = max(int(round(orig_w * ratio / 32) * 32), 32)

        scale_x = float(resize_w) / float(orig_w)
        scale_y = float(resize_h) / float(orig_h)

        # Resize
        resized = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        # Normalize (BGR ImageNet)
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - DEFAULT_MEAN_BGR) / DEFAULT_STD_BGR

        # HWC -> NCHW
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
        return tensor, (scale_x, scale_y), (orig_h, orig_w)

    def detect(self, image: Any) -> DetectionResult:
        """Run text detection on an input image.

        Args:
            image: OpenCV BGR image numpy array.

        Returns:
            DetectionResult dataclass with detection polygons, bounding boxes,
            confidence scores, and latency metrics.
        """
        t_total_start = time.perf_counter()

        # Validate input
        if image is None or not isinstance(image, np.ndarray):
            return DetectionResult(
                original_shape=(0, 0),
                execution_provider=self.active_provider,
                success=False,
                error_message="Invalid image input: expected a non-None numpy.ndarray.",
            )

        if image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            orig_shape = (image.shape[0], image.shape[1]) if image.ndim >= 2 else (0, 0)
            return DetectionResult(
                original_shape=orig_shape,
                execution_provider=self.active_provider,
                success=False,
                error_message=f"Invalid image format: expected 3-channel BGR image, got shape {getattr(image, 'shape', None)}.",
            )

        try:
            # 1. Preprocess
            t_pre_start = time.perf_counter()
            tensor, (scale_x, scale_y), orig_shape = self.preprocess(image)
            pre_time_ms = (time.perf_counter() - t_pre_start) * 1000.0

            # 2. Inference
            t_infer_start = time.perf_counter()
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            infer_time_ms = (time.perf_counter() - t_infer_start) * 1000.0
            prob_map = outputs[0]

            # 3. Postprocess
            t_post_start = time.perf_counter()
            polygons, boxes_4pt, boxes_xyxy, confidences = self.postprocessor(
                prob_map=prob_map,
                scale_x=scale_x,
                scale_y=scale_y,
                orig_shape=orig_shape,
            )

            # Optional reading order sorting
            if self.sort_reading_order_enabled and len(boxes_xyxy) > 1:
                _, sorted_boxes_xyxy, sorted_boxes_4pt, sorted_confs = sort_reading_order(
                    boxes=boxes_xyxy,
                    polygons=boxes_4pt,
                    confidences=confidences,
                )
                boxes_xyxy = sorted_boxes_xyxy
                boxes_4pt = sorted_boxes_4pt if sorted_boxes_4pt is not None else boxes_4pt
                confidences = sorted_confs if sorted_confs is not None else confidences

            post_time_ms = (time.perf_counter() - t_post_start) * 1000.0
            total_time_ms = (time.perf_counter() - t_total_start) * 1000.0

            return DetectionResult(
                original_shape=orig_shape,
                polygons=polygons,
                boxes_4pt=boxes_4pt,
                boxes_xyxy=boxes_xyxy,
                confidences=confidences,
                num_detections=len(boxes_xyxy),
                pre_time_ms=pre_time_ms,
                infer_time_ms=infer_time_ms,
                post_time_ms=post_time_ms,
                total_time_ms=total_time_ms,
                execution_provider=self.active_provider,
                success=True,
            )

        except Exception as e:
            logger.exception("Error during DBNet text detection: %s", e)
            total_time_ms = (time.perf_counter() - t_total_start) * 1000.0
            return DetectionResult(
                original_shape=image.shape[:2],
                execution_provider=self.active_provider,
                total_time_ms=total_time_ms,
                success=False,
                error_message=f"Detection execution failure: {e}",
            )
