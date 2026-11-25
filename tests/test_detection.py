"""
NAYRA - Assistive System
Part 2, Module 03: Assistive Object Detection - Test Suite

Tests:
- Model registry and metadata validation
- Unsupported assistive classes verification (door, stairs)
- Postprocessing (bounding box clipping, confidence filtering, spatial zones, assistive hints)
- Device selection (GPU vs CPU fallback)
- Invalid frame validation
- Annotation rendering
- End-to-end detector inference (real model)
- Integration with CameraManager
"""

import importlib
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Ensure project root directory is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Register 03_object_detection module dynamically
_det_module = importlib.import_module("03_object_detection.detector")
_reg_module = importlib.import_module("03_object_detection.model_registry")
_post_module = importlib.import_module("03_object_detection.postprocessor")

ObjectDetector = _det_module.ObjectDetector
ASSISTIVE_CLASSES = _reg_module.ASSISTIVE_CLASSES
COCO_CLASSES = _reg_module.COCO_CLASSES
UNSUPPORTED_ASSISTIVE_CLASSES = _reg_module.UNSUPPORTED_ASSISTIVE_CLASSES
ensure_model_weights = _reg_module.ensure_model_weights
get_model_metadata = _reg_module.get_model_metadata
Detection = _post_module.Detection
DetectionMetrics = _post_module.DetectionMetrics
DetectionPostprocessor = _post_module.DetectionPostprocessor
from config.settings import CameraSettings, ObjectDetectionSettings


# ============================================================================
# 1. MODEL REGISTRY & METADATA TESTS
# ============================================================================

def test_model_metadata_yolov10n():
    """Verify metadata for YOLOv10-N model."""
    meta = get_model_metadata("yolov10n")
    assert meta.name == "yolov10n"
    assert meta.license_type == "Apache-2.0"
    assert meta.input_shape == (1, 3, 640, 640)
    assert meta.filename == "yolov10n.onnx"
    assert "COCO" in meta.dataset


def test_model_metadata_yolov10s():
    """Verify metadata for YOLOv10-S model."""
    meta = get_model_metadata("yolov10s")
    assert meta.name == "yolov10s"
    assert meta.license_type == "Apache-2.0"
    assert meta.input_shape == (1, 3, 640, 640)
    assert meta.filename == "yolov10s.onnx"
    assert "COCO" in meta.dataset


def test_model_metadata_unknown_raises():
    """Verify requesting an unregistered model raises KeyError."""
    with pytest.raises(KeyError):
        get_model_metadata("non_existent_model")


def test_unsupported_classes_verification():
    """Verify that 'door' and 'stairs' are explicitly identified as NOT in COCO."""
    assert "door" in UNSUPPORTED_ASSISTIVE_CLASSES
    assert "stairs" in UNSUPPORTED_ASSISTIVE_CLASSES
    assert "door" not in COCO_CLASSES
    assert "stairs" not in COCO_CLASSES


def test_ensure_model_weights_cached(tmp_path):
    """Verify that existing model weights are reused without re-downloading."""
    fake_weights = tmp_path / "yolov10n.onnx"
    fake_weights.write_bytes(b"dummy ONNX content > 1024 bytes" * 100)

    path = ensure_model_weights(model_name="yolov10n", models_dir=str(tmp_path))
    assert path == fake_weights
    assert path.exists()


# ============================================================================
# 2. DETECTION POSTPROCESSOR TESTS
# ============================================================================

class TestPostprocessor:
    @pytest.fixture
    def postprocessor(self):
        return DetectionPostprocessor(class_names=COCO_CLASSES)

    def test_spatial_zone_calculation(self):
        """Verify horizontal coordinate mapping to left, center, right."""
        # Left zone: center_x < 0.33
        zone, hint = DetectionPostprocessor.calculate_spatial_zone(0.05, 0.20)
        assert zone == "left"
        assert hint == "on the left"

        # Center zone: 0.33 <= center_x <= 0.66
        zone, hint = DetectionPostprocessor.calculate_spatial_zone(0.35, 0.60)
        assert zone == "center"
        assert hint == "ahead"

        # Right zone: center_x > 0.66
        zone, hint = DetectionPostprocessor.calculate_spatial_zone(0.70, 0.95)
        assert zone == "right"
        assert hint == "on the right"

    def test_confidence_filtering(self, postprocessor):
        """Verify predictions below confidence threshold are omitted."""
        raw_preds = np.array([
            [100, 100, 200, 200, 0.75, 0],   # person: 0.75 >= 0.40 -> keep
            [50, 50, 150, 150, 0.35, 56],    # chair: 0.35 < 0.40 -> reject
            [20, 20, 80, 80, 0.10, 39],      # bottle: 0.10 < 0.40 -> reject
        ])
        results = postprocessor.process(
            raw_preds,
            frame_width=640,
            frame_height=480,
            frame_timestamp=100.0,
            confidence_threshold=0.40,
        )
        assert len(results) == 1
        assert results[0].class_name == "person"
        assert results[0].confidence == 0.75

    def test_bounding_box_clipping_and_validation(self, postprocessor):
        """Verify coordinates outside image boundaries are properly clipped."""
        # Box extending past right and bottom borders
        raw_preds = np.array([[-50, -20, 700, 550, 0.90, 0]])
        results = postprocessor.process(
            raw_preds,
            frame_width=640,
            frame_height=480,
            frame_timestamp=0.0,
            confidence_threshold=0.40,
        )
        assert len(results) == 1
        x1, y1, x2, y2 = results[0].box_xyxy
        assert x1 == 0
        assert y1 == 0
        assert x2 == 640
        assert y2 == 480

    def test_degenerate_box_rejection(self, postprocessor):
        """Verify degenerate (zero or negative area) boxes are rejected."""
        # Inverted box (x2 < x1)
        raw_preds = np.array([[200, 200, 100, 100, 0.88, 0]])
        results = postprocessor.process(
            raw_preds,
            frame_width=640,
            frame_height=480,
            frame_timestamp=0.0,
            confidence_threshold=0.40,
        )
        assert len(results) == 0

    def test_empty_predictions_handling(self, postprocessor):
        """Verify empty or None raw predictions return empty list gracefully."""
        assert postprocessor.process(None, 640, 480, 0.0) == []
        assert postprocessor.process(np.empty((0, 6)), 640, 480, 0.0) == []

    def test_assistive_filter_only(self, postprocessor):
        """Verify filter_assistive_only excludes non-assistive objects like airplanes."""
        raw_preds = np.array([
            [100, 100, 200, 200, 0.80, 0],   # person (assistive)
            [300, 100, 400, 200, 0.85, 4],   # airplane (not everyday assistive)
            [150, 150, 250, 250, 0.70, 56],  # chair (assistive)
        ])
        results = postprocessor.process(
            raw_preds,
            frame_width=640,
            frame_height=480,
            frame_timestamp=0.0,
            confidence_threshold=0.50,
            filter_assistive_only=True,
        )
        class_names = [d.class_name for d in results]
        assert "person" in class_names
        assert "chair" in class_names
        assert "airplane" not in class_names

    def test_iou_deduplication(self, postprocessor):
        """Verify duplicate overlapping boxes of the same class are pruned."""
        # Two overlapping boxes for person (class 0) with IoU > 0.80
        raw_preds = np.array([
            [100, 100, 200, 200, 0.88, 0],   # Higher confidence box
            [102,  98, 201, 202, 0.65, 0],   # Near-duplicate box
            [300, 300, 400, 400, 0.75, 56],  # Non-overlapping chair
        ])
        results = postprocessor.process(
            raw_preds,
            frame_width=640,
            frame_height=480,
            frame_timestamp=0.0,
            confidence_threshold=0.40,
            iou_threshold=0.65,
        )
        # Should keep the higher confidence person and the chair (total 2)
        assert len(results) == 2
        assert results[0].confidence == 0.88
        assert results[0].class_name == "person"
        assert results[1].class_name == "chair"

    def test_letterbox_coordinate_unpadding(self, postprocessor):
        """Verify postprocessor accurately unletterboxes coordinates back to original frame space."""
        # Assume 1920x1080 frame letterboxed to 640x640:
        # scale ratio r = 640 / 1920 = 0.333333
        # unpadded height = 1080 * (640/1920) = 360
        # pad_y = (640 - 360) / 2 = 140
        # Box inside 640x640 model space: [0, 140, 640, 500] (spanning full unpadded region)
        raw_preds = np.array([
            [0.0, 140.0, 640.0, 500.0, 0.90, 0],
        ])
        r = 640.0 / 1920.0
        results = postprocessor.process(
            raw_preds,
            frame_width=1920,
            frame_height=1080,
            frame_timestamp=0.0,
            confidence_threshold=0.50,
            ratio=r,
            pad_x=0.0,
            pad_y=140.0,
        )
        assert len(results) == 1
        x1, y1, x2, y2 = results[0].box_xyxy
        assert x1 == 0
        assert y1 == 0
        assert x2 == 1920
        assert y2 == 1080


def test_letterbox_helper_geometry():
    """Verify ObjectDetector.letterbox preserves aspect ratio and pads symmetrically."""
    dummy_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
    padded, ratio, (pad_x, pad_y) = ObjectDetector.letterbox(dummy_1080p, (640, 640))

    assert padded.shape == (640, 640, 3)
    assert abs(ratio - (640.0 / 1920.0)) < 1e-4
    assert pad_x == 0.0
    assert abs(pad_y - 140.0) < 1.0


# ============================================================================
# 3. DETECTOR HARDWARE & VALIDATION UNIT TESTS
# ============================================================================

def test_invalid_frames_raise_error():
    """Verify detector raises ValueError on invalid input frames."""
    # Initialize in CPU mode
    settings = ObjectDetectionSettings(device="cpu")
    detector = ObjectDetector(settings)

    with pytest.raises(ValueError, match="Invalid frame"):
        detector.detect(None)

    with pytest.raises(ValueError, match="Invalid frame"):
        detector.detect(np.empty((0, 0, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="Invalid frame"):
        detector.detect("not_an_array")  # type: ignore


def test_cpu_device_selection_enforced():
    """Verify specifying device='cpu' forces CPU execution."""
    settings = ObjectDetectionSettings(device="cpu")
    detector = ObjectDetector(settings)
    assert "CPU" in detector._active_device


def test_settings_validation():
    """Verify ObjectDetectionSettings validates thresholds and dimensions."""
    with pytest.raises(ValueError):
        ObjectDetectionSettings(confidence_threshold=1.5)

    with pytest.raises(ValueError):
        ObjectDetectionSettings(confidence_threshold=-0.1)

    with pytest.raises(ValueError):
        ObjectDetectionSettings(input_width=0)

    with pytest.raises(ValueError):
        ObjectDetectionSettings(device="invalid_dev")


def test_draw_detections_annotation():
    """Verify draw_detections draws annotations on a frame copy without mutating original."""
    settings = ObjectDetectionSettings(device="cpu")
    detector = ObjectDetector(settings)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_orig = frame.copy()

    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.92,
        box_xyxy=(50, 50, 200, 300),
        box_normalized=(50 / 640, 50 / 480, 200 / 640, 300 / 480),
        spatial_zone="left",
        spatial_hint="Person detected on the left.",
        frame_timestamp=time.time(),
        frame_width=640,
        frame_height=480,
    )
    metrics = DetectionMetrics(
        inference_latency_ms=12.5,
        inference_fps=80.0,
        device_used="Test Device",
        num_detections=1,
        mean_confidence=0.92,
    )

    annotated = detector.draw_detections(frame, [detection], metrics=metrics)
    assert annotated is not frame
    assert np.array_equal(frame, frame_orig)  # Original unchanged
    assert annotated.shape == frame.shape
    # Annotated canvas should contain non-zero pixel markings
    assert np.any(annotated > 0)


# ============================================================================
# 4. REAL INFERENCE & CAMERA INTEGRATION TESTS
# ============================================================================

def test_real_model_inference_synthetic_frame():
    """Perform real model inference on a synthetic image and check metrics."""
    settings = ObjectDetectionSettings(confidence_threshold=0.30)
    detector = ObjectDetector(settings)

    # Synthetic BGR image with a white block representing an object
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (300, 200), (600, 600), (255, 255, 255), -1)

    detections, metrics = detector.detect(test_frame)

    assert isinstance(detections, list)
    assert isinstance(metrics, DetectionMetrics)
    assert metrics.inference_latency_ms > 0
    assert metrics.inference_fps > 0
    assert len(metrics.device_used) > 0


def test_integration_with_camera_manager():
    """Verify real-time integration pipeline with CameraManager."""
    import importlib
    cam_mod = importlib.import_module("02_camera_manager.camera_manager")
    CameraManager = cam_mod.CameraManager

    # Use a synthetic/test video generator or mock source
    camera_settings = CameraSettings(
        stream_url=0,
        frame_buffer_size=1,
    )
    det_settings = ObjectDetectionSettings(confidence_threshold=0.35)
    detector = ObjectDetector(det_settings)

    # Mock camera frame retrieval
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    with patch.object(CameraManager, "get_frame", return_value=(dummy_frame, time.time())):
        mgr = CameraManager(settings=camera_settings)
        frame, ts = mgr.get_frame()
        assert frame is not None

        detections, metrics = detector.detect(frame, timestamp=ts)
        assert isinstance(detections, list)
        assert metrics.inference_latency_ms > 0
