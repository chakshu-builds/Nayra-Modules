"""Assistive object detector using ONNX Runtime with CUDA acceleration and OpenCV fallback."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from config.hardware import resolve_device
from config.settings import ObjectDetectionSettings, get_default_detection_settings
from .model_registry import (
    ASSISTIVE_CLASSES,
    COCO_CLASSES,
    ensure_model_weights,
)
from .postprocessor import Detection, DetectionMetrics, DetectionPostprocessor

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Performs real-time object detection with letterbox preprocessing and coordinate restoration."""

    def __init__(self, settings: Optional[ObjectDetectionSettings] = None) -> None:
        self.settings = settings or get_default_detection_settings()
        self.postprocessor = DetectionPostprocessor(
            class_names=COCO_CLASSES,
            assistive_filter=ASSISTIVE_CLASSES if self.settings.filter_assistive_only else None,
        )

        self._cuda_available, self._active_device, self._gpu_name = resolve_device(self.settings.device)

        if self.settings.model_path and Path(self.settings.model_path).exists():
            self.model_path = Path(self.settings.model_path)
        else:
            self.model_path = ensure_model_weights(
                model_name=self.settings.model_name,
                models_dir=self.settings.models_dir,
            )

        self._ort_session = None
        self._cv2_net = None
        self._load_engine()

        self._last_latency_ms: float = 0.0
        self._fps_ema: float = 0.0
        self._ema_alpha: float = 0.15

        logger.info(
            "ObjectDetector initialized. Model: %s | Device: %s | Threshold: %.2f",
            self.model_path.name,
            self._active_device,
            self.settings.confidence_threshold,
        )

    def _load_engine(self) -> None:
        model_str = str(self.model_path.resolve())

        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            providers: List[str] = []

            use_cuda = (
                self._cuda_available
                and self.settings.device.lower() in ("auto", "cuda")
                and "CUDAExecutionProvider" in available
            )
            if use_cuda:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            self._ort_session = ort.InferenceSession(
                model_str,
                sess_options=sess_options,
                providers=providers,
            )
            active_prov = self._ort_session.get_providers()[0]
            if "CUDA" in active_prov:
                self._active_device = f"{self._gpu_name or 'NVIDIA RTX'} (CUDA)"
            else:
                self._active_device = f"CPU ({active_prov})"
            logger.info("Loaded ONNX Runtime session with provider: %s", active_prov)
            return
        except Exception as exc:
            logger.warning("ONNX Runtime load failed or unavailable (%s). Falling back to OpenCV DNN.", exc)

        try:
            self._cv2_net = cv2.dnn.readNetFromONNX(model_str)
            self._cv2_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._cv2_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._active_device = "CPU (OpenCV DNN)"
            logger.info("Loaded model via OpenCV DNN CPU backend.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load detection model from {model_str}: {exc}") from exc

    @staticmethod
    def letterbox(
        image: np.ndarray,
        target_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        shape = image.shape[:2]
        target_w, target_h = target_shape

        r = min(float(target_w) / float(shape[1]), float(target_h) / float(shape[0]))
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (target_w - new_unpad[0]) / 2.0
        dh = (target_h - new_unpad[1]) / 2.0

        if shape[::-1] != new_unpad:
            resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return padded, r, (dw, dh)

    def _preprocess(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[float], float, float, float, float]:
        h, w = frame.shape[:2]
        target_w, target_h = self.settings.input_width, self.settings.input_height

        if self.settings.letterbox:
            processed_img, ratio, (pad_x, pad_y) = self.letterbox(frame, (target_w, target_h))
            scale_x = float(w) / float(target_w)
            scale_y = float(h) / float(target_h)
        else:
            processed_img = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            ratio = None
            pad_x, pad_y = 0.0, 0.0
            scale_x = float(w) / float(target_w)
            scale_y = float(h) / float(target_h)

        rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, ratio, pad_x, pad_y, scale_x, scale_y

    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        if self._ort_session is not None:
            input_name = self._ort_session.get_inputs()[0].name
            outputs = self._ort_session.run(None, {input_name: tensor})
            return outputs[0]
        elif self._cv2_net is not None:
            self._cv2_net.setInput(tensor)
            return self._cv2_net.forward()
        raise RuntimeError("No inference backend initialized.")

    def detect(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[List[Detection], DetectionMetrics]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame supplied to detect(): must be non-empty numpy array")

        if timestamp is None:
            timestamp = time.time()

        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        tensor, ratio, pad_x, pad_y, scale_x, scale_y = self._preprocess(frame)
        raw_output = self._forward(tensor)

        detections = self.postprocessor.process(
            raw_predictions=raw_output,
            frame_width=w,
            frame_height=h,
            frame_timestamp=timestamp,
            confidence_threshold=self.settings.confidence_threshold,
            filter_assistive_only=self.settings.filter_assistive_only,
            scale_x=scale_x,
            scale_y=scale_y,
            ratio=ratio,
            pad_x=pad_x,
            pad_y=pad_y,
            iou_threshold=self.settings.iou_threshold,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        if self._fps_ema == 0.0:
            self._fps_ema = fps
        else:
            self._fps_ema = (self._ema_alpha * fps) + ((1.0 - self._ema_alpha) * self._fps_ema)

        self._last_latency_ms = latency_ms

        gpu_alloc_mb = None
        gpu_res_mb = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
                gpu_res_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)
        except Exception:
            pass

        mean_conf = 0.0
        if detections:
            mean_conf = round(sum(d.confidence for d in detections) / len(detections), 4)

        metrics = DetectionMetrics(
            inference_latency_ms=round(latency_ms, 2),
            inference_fps=round(self._fps_ema, 1),
            device_used=self._active_device,
            gpu_memory_allocated_mb=gpu_alloc_mb,
            gpu_memory_reserved_mb=gpu_res_mb,
            num_detections=len(detections),
            mean_confidence=mean_conf,
        )

        return detections, metrics

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        metrics: Optional[DetectionMetrics] = None,
        draw_hud: bool = True,
    ) -> np.ndarray:
        output = frame.copy()
        h, w = output.shape[:2]

        zone_colors = {
            "left": (255, 178, 50),
            "center": (50, 220, 50),
            "right": (50, 150, 255),
        }

        col_left = int(w * 0.33)
        col_right = int(w * 0.66)
        cv2.line(output, (col_left, 0), (col_left, h), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(output, (col_right, 0), (col_right, h), (70, 70, 70), 1, cv2.LINE_AA)

        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            color = zone_colors.get(det.spatial_zone, (0, 255, 0))

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            conf_pct = int(round(det.confidence * 100))
            label_text = f"{det.class_name.capitalize()} {conf_pct}% [{det.spatial_zone}]"

            (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            banner_y1 = max(0, y1 - lh - 8)
            banner_y2 = y1
            cv2.rectangle(output, (x1, banner_y1), (x1 + lw + 8, banner_y2), color, -1)
            cv2.putText(
                output,
                label_text,
                (x1 + 4, banner_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if draw_hud and metrics is not None:
            hud_lines = [
                f"Device: {metrics.device_used}",
                f"Det Latency: {metrics.inference_latency_ms:.1f} ms ({metrics.inference_fps:.1f} FPS)",
                f"Objects: {metrics.num_detections} (Avg Conf: {int(metrics.mean_confidence * 100)}%)",
            ]
            if metrics.gpu_memory_reserved_mb is not None:
                hud_lines.append(f"VRAM: {metrics.gpu_memory_allocated_mb}M / {metrics.gpu_memory_reserved_mb}M")

            hud_y = h - 20 - (len(hud_lines) * 22)
            cv2.rectangle(output, (10, hud_y - 8), (380, h - 15), (20, 20, 20), -1)
            cv2.rectangle(output, (10, hud_y - 8), (380, h - 15), (0, 200, 255), 1)

            for i, line in enumerate(hud_lines):
                cv2.putText(
                    output,
                    line,
                    (20, hud_y + 14 + (i * 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return output
