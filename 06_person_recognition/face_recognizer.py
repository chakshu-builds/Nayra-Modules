"""Pretrained SFace face embedding feature extractor with CUDA and CPU execution."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from config.hardware import resolve_device
from config.settings import PersonRecognitionSettings, get_default_person_recognition_settings

logger = logging.getLogger(__name__)

SFACE_DEFAULT_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


class FaceRecognizer:
    """Extracts 128-dimensional L2-normalized face embeddings using SFace."""

    def __init__(self, settings: PersonRecognitionSettings | None = None) -> None:
        self.settings = settings or get_default_person_recognition_settings()
        self.model_path = self._resolve_model_path()
        self._cuda_available, self._active_device, self._gpu_name = resolve_device(self.settings.device)

        self._ort_session = None
        self._cv2_recognizer: cv2.FaceRecognizerSF | None = None
        self._load_engine()

        logger.info(
            "FaceRecognizer initialized (%s, device: %s, embedding_dim: 128)",
            self.model_path.name,
            self._active_device,
        )

    def _resolve_model_path(self) -> Path:
        if self.settings.recognizer_model_path and Path(self.settings.recognizer_model_path).exists():
            return Path(self.settings.recognizer_model_path)

        models_dir = Path(self.settings.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        target_path = models_dir / f"{self.settings.recognizer_model_name}.onnx"

        if not target_path.exists():
            logger.info("Downloading SFace recognition weights to %s...", target_path)
            try:
                urllib.request.urlretrieve(SFACE_DEFAULT_URL, str(target_path))
            except Exception as dl_err:
                logger.error("Failed to download SFace model: %s", dl_err)
                raise FileNotFoundError(f"Could not download SFace from {SFACE_DEFAULT_URL}: {dl_err}") from dl_err

        return target_path

    def _load_engine(self) -> None:
        try:
            import onnxruntime as ort

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

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = max(1, os.cpu_count() or 4)

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
            logger.info("SFace ONNX session loaded with provider: %s", active_p)

            # Warmup
            dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
            self._ort_session.run(["fc1"], {"data": dummy})
        except Exception as ort_err:
            logger.warning("ONNX Runtime setup failed (%s), falling back to OpenCV FaceRecognizerSF.", ort_err)
            self._ort_session = None
            try:
                self._cv2_recognizer = cv2.FaceRecognizerSF.create(str(self.model_path.resolve()), "")
                self._active_device = "CPU (OpenCV FaceRecognizerSF)"
            except Exception as cv_err:
                logger.error("Failed to initialize FaceRecognizerSF: %s", cv_err)
                raise RuntimeError(f"Could not initialize SFace engine: {cv_err}") from cv_err

    def extract_embedding(self, aligned_crop: np.ndarray) -> np.ndarray | None:
        """Extract 128D unit L2-normalized embedding vector from aligned 112x112 face crop."""
        if aligned_crop is None or not isinstance(aligned_crop, np.ndarray) or aligned_crop.size == 0:
            return None

        if aligned_crop.shape[:2] != (112, 112):
            aligned_crop = cv2.resize(aligned_crop, (112, 112), interpolation=cv2.INTER_LINEAR)

        try:
            if self._ort_session is not None:
                # SFace expects (1, 3, 112, 112) float32 in BGR channel order
                tensor = np.transpose(aligned_crop.astype(np.float32), (2, 0, 1))[np.newaxis, ...]
                raw_emb = self._ort_session.run(["fc1"], {"data": tensor})[0][0]
            elif self._cv2_recognizer is not None:
                raw_emb = self._cv2_recognizer.feature(aligned_crop)[0]
            else:
                return None

            norm = float(np.linalg.norm(raw_emb))
            if norm < 1e-6:
                return None

            normalized_emb = (raw_emb / norm).astype(np.float32)
            return normalized_emb
        except Exception as err:
            logger.warning("Feature extraction failed: %s", err)
            return None

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two 128D normalized embeddings."""
        dot = float(np.dot(emb1, emb2))
        return float(np.clip(dot, -1.0, 1.0))

    @property
    def active_device(self) -> str:
        return self._active_device
