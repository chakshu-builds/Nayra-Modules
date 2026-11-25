"""Monocular depth estimation using ONNX models with CUDA acceleration."""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config.hardware import resolve_device
from config.settings import SpatialAwarenessSettings, get_default_spatial_settings

logger = logging.getLogger(__name__)

DEPTH_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "midas_v21_small": {
        "filename": "midas_v21_small.onnx",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        "input_size": (256, 256),
        "is_disparity": True,
        "license": "MIT",
        "architecture": "EfficientNet-Lite3 + Multi-scale Fusion",
        "params": "21M",
    },
    "depth_anything_v2_vits": {
        "filename": "depth_anything_v2_vits.onnx",
        "url": "https://huggingface.co/onnx-community/Depth-Anything-V2-Small/resolve/main/onnx/model.onnx",
        "input_size": (518, 518),
        "is_disparity": False,
        "license": "Apache-2.0",
        "architecture": "DINOv2-Small ViT + DPT",
        "params": "24.8M",
    },
}


class DepthEstimator:
    """Monocular depth estimation engine using ONNX Runtime."""

    def __init__(self, settings: SpatialAwarenessSettings | None = None) -> None:
        self.settings = settings or get_default_spatial_settings()
        self._cuda_available, self._active_device, self._gpu_name = resolve_device(self.settings.device)

        model_key = self.settings.model_name.lower().replace("-", "_")
        if model_key not in DEPTH_MODEL_CONFIGS:
            logger.warning("Unknown depth model '%s', falling back to 'midas_v21_small'.", self.settings.model_name)
            model_key = "midas_v21_small"

        self.model_config = DEPTH_MODEL_CONFIGS[model_key]
        self.input_size: tuple[int, int] = self.model_config["input_size"]
        self.is_disparity: bool = self.model_config["is_disparity"]

        if self.settings.model_path and Path(self.settings.model_path).exists():
            self.model_path = Path(self.settings.model_path)
        else:
            self.model_path = self._resolve_model_path(model_key)

        self._ort_session = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._load_engine()

        self._last_latency_ms: float = 0.0
        self._fps_ema: float = 0.0
        self._ema_alpha: float = 0.15

        logger.info(
            "DepthEstimator initialized (%s, %s, %dx%d, disparity=%s)",
            self.model_path.name,
            self._active_device,
            self.input_size[0],
            self.input_size[1],
            self.is_disparity,
        )

    def _resolve_model_path(self, model_key: str) -> Path:
        models_dir = Path(self.settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        config = DEPTH_MODEL_CONFIGS[model_key]
        dest_path = models_dir / config["filename"]

        if not dest_path.exists():
            url = config["url"]
            logger.info("Downloading depth model weights from %s to %s...", url, dest_path)
            try:
                urllib.request.urlretrieve(url, str(dest_path))
                logger.info("Downloaded %s (%.1f MB)", dest_path.name, dest_path.stat().st_size / (1024 * 1024))
            except Exception as dl_err:
                logger.error("Failed to download depth model weights: %s", dl_err)
                raise FileNotFoundError(f"Could not download depth model from {url}: {dl_err}") from dl_err

        return dest_path

    def _load_engine(self) -> None:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, os.cpu_count() or 4)

        available = ort.get_available_providers()
        providers: list[str] = []
        use_cuda = (
            self._cuda_available
            and self.settings.device.lower() in ("auto", "cuda")
            and "CUDAExecutionProvider" in available
        )
        if use_cuda:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        try:
            self._ort_session = ort.InferenceSession(
                str(self.model_path.resolve()),
                sess_options=sess_options,
                providers=providers,
            )
            active_p = self._ort_session.get_providers()[0]
            if "CUDA" in active_p:
                self._active_device = f"{self._gpu_name or 'NVIDIA RTX'} (CUDA)"
            else:
                self._active_device = f"CPU ({active_p})"
            logger.info("Depth ONNX session created with provider: %s", active_p)
        except Exception as ort_err:
            logger.warning("CUDA execution failed (%s), falling back to CPU.", ort_err)
            self._ort_session = ort.InferenceSession(
                str(self.model_path.resolve()),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._active_device = "CPU (CPUExecutionProvider)"

        self._input_name = self._ort_session.get_inputs()[0].name
        self._output_name = self._ort_session.get_outputs()[0].name

        # Warm up engine
        try:
            dummy_shape = (1, 3, self.input_size[1], self.input_size[0])
            dummy_tensor = np.zeros(dummy_shape, dtype=np.float32)
            self._ort_session.run([self._output_name], {self._input_name: dummy_tensor})
        except Exception:
            pass

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize and normalize frame to ImageNet tensor (1, 3, H, W)."""
        orig_h, orig_w = frame.shape[:2]
        target_w, target_h = self.input_size

        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std

        tensor = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[np.newaxis, ...])
        return tensor, (orig_w, orig_h)

    def estimate(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Generate full-frame relative depth map and normalized proximity map."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame is invalid or empty.")

        t0 = time.perf_counter()
        tensor, (orig_w, orig_h) = self.preprocess(frame)

        raw_output = self._ort_session.run([self._output_name], {self._input_name: tensor})[0]
        t_infer = (time.perf_counter() - t0) * 1000.0

        depth_raw = np.squeeze(raw_output).astype(np.float32)
        depth_aligned = cv2.resize(depth_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        proximity_norm = self._standardize_proximity(depth_aligned)

        self._last_latency_ms = t_infer
        inst_fps = 1000.0 / max(t_infer, 1e-3)
        self._fps_ema = (
            inst_fps
            if self._fps_ema == 0.0
            else (self._ema_alpha * inst_fps + (1.0 - self._ema_alpha) * self._fps_ema)
        )

        return depth_aligned, proximity_norm, t_infer

    def _standardize_proximity(self, depth_aligned: np.ndarray) -> np.ndarray:
        """Map raw depth to uniform proximity [0.0, 1.0] where 1.0 is closest."""
        valid_mask = np.isfinite(depth_aligned)
        if not np.any(valid_mask):
            return np.zeros_like(depth_aligned, dtype=np.float32)

        d_min = float(np.min(depth_aligned[valid_mask]))
        d_max = float(np.max(depth_aligned[valid_mask]))
        diff = d_max - d_min

        if diff < 1e-6:
            return np.full_like(depth_aligned, 0.5, dtype=np.float32)

        if self.is_disparity:
            norm = (depth_aligned - d_min) / diff
        else:
            norm = 1.0 - ((depth_aligned - d_min) / diff)

        return np.clip(norm, 0.0, 1.0).astype(np.float32)

    @property
    def active_device(self) -> str:
        return self._active_device

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def fps(self) -> float:
        return self._fps_ema
