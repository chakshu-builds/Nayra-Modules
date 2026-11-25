"""
NAYRA - Assistive System
Module 03: Assistive Object Detection - Model Registry

Maintains model metadata, official download sources, licenses,
and class definitions for assistive object detection.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Standard 80 classes supported by the pretrained COCO model
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Everyday objects particularly relevant for blind/visually impaired navigation & assistance
ASSISTIVE_CLASSES: Set[str] = {
    "person", "chair", "couch", "dining table", "bed", "bottle", "cup",
    "cell phone", "laptop", "backpack", "handbag", "suitcase",
    "bicycle", "car", "motorcycle", "bus", "truck"
}

# Note regarding architecture & classes:
# 'door' and 'stairs' are NOT part of the standard COCO 80-class taxonomy.
# To ensure zero false labeling, they are not mapped unless specialized models are used.
UNSUPPORTED_ASSISTIVE_CLASSES: Set[str] = {"door", "stairs"}


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing a downloadable pretrained detector model."""
    name: str
    architecture: str
    license_type: str
    dataset: str
    input_shape: tuple[int, int, int, int]
    download_url: str
    filename: str
    description: str


REGISTRY: Dict[str, ModelMetadata] = {
    "yolov10n": ModelMetadata(
        name="yolov10n",
        architecture="YOLOv10-N (NMS-free dual label assignment)",
        license_type="Apache-2.0",
        dataset="COCO (80 classes)",
        input_shape=(1, 3, 640, 640),
        download_url="https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx",
        filename="yolov10n.onnx",
        description="Lightweight real-time detector (2.3M parameters, 8.2 GFLOPs, 38.5 mAP), Apache-2.0 license."
    ),
    "yolov10s": ModelMetadata(
        name="yolov10s",
        architecture="YOLOv10-S (NMS-free dual label assignment)",
        license_type="Apache-2.0",
        dataset="COCO (80 classes)",
        input_shape=(1, 3, 640, 640),
        download_url="https://huggingface.co/onnx-community/yolov10s/resolve/main/onnx/model.onnx",
        filename="yolov10s.onnx",
        description="Higher-accuracy real-time detector (7.2M parameters, 21.6 GFLOPs, 46.3 mAP), Apache-2.0 license."
    ),
}


def get_model_metadata(model_name: str = "yolov10n") -> ModelMetadata:
    """Retrieve metadata for a registered model."""
    if model_name not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise KeyError(f"Unknown model '{model_name}'. Available: {available}")
    return REGISTRY[model_name]


def ensure_model_weights(model_name: str = "yolov10n", models_dir: str = "models") -> Path:
    """Ensure that the required model weights exist locally, downloading if necessary.

    Args:
        model_name: Identifier in the model registry.
        models_dir: Directory where weights are cached.

    Returns:
        Path to local weight file.
    """
    meta = get_model_metadata(model_name)
    target_dir = Path(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    weights_path = target_dir / meta.filename
    if weights_path.exists() and weights_path.stat().st_size > 1024:
        logger.info("Found cached model weights at %s (%d bytes)", weights_path, weights_path.stat().st_size)
        return weights_path

    logger.info("Downloading %s weights from %s ...", meta.name, meta.download_url)
    try:
        req = urllib.request.Request(
            meta.download_url,
            headers={"User-Agent": "NAYRA-Assistive-Vision/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30.0) as response, open(weights_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        logger.info("Successfully downloaded %s (%d bytes)", weights_path, weights_path.stat().st_size)
    except Exception as exc:
        if weights_path.exists():
            weights_path.unlink()
        raise RuntimeError(f"Failed to download model weights from {meta.download_url}: {exc}") from exc

    return weights_path
