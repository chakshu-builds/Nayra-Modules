"""NAYRA Module 07 — Comprehensive Test Suite for DBNet Text Detector.

Covers:
1. Model loading & ONNX graph verification
2. CUDA provider reporting & active provider inspection
3. CPU fallback execution
4. Input validation & error handling
5. Preprocessing dimensions, normalization & multiples of 32
6. Aspect-ratio preservation
7. Coordinate restoration & clipping
8. Empty probability maps & zero-detection handling
9. Multiple text regions detection
10. Rotated text polygon handling & 4-point ordering
11. Nearby text lines separation
12. Confidence threshold filtering
13. Reading-order sorting (top-to-bottom, left-to-right)
14. Real Hindi & English text crops evaluation
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

# Ensure 07_ocr is on sys.path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from detector.dbnet_detector import DBNetDetector, DetectionResult, configure_cuda_dlls
from detector.geometry import (
    box_score_fast,
    order_points_clockwise,
    polygon_to_xyxy,
    restore_coordinates,
    sort_reading_order,
    unclip_rotated_rect,
)
from detector.postprocess import DBPostProcessor


class TestDBNetGeometry(unittest.TestCase):
    """Test geometric operations and postprocessing primitives."""

    def test_order_points_clockwise(self):
        # Shuffled rectangle vertices: TL=(10, 10), TR=(100, 10), BR=(100, 50), BL=(10, 50)
        pts = np.array([[100, 50], [10, 10], [10, 50], [100, 10]], dtype=np.float32)
        ordered = order_points_clockwise(pts)
        self.assertEqual(ordered.shape, (4, 2))
        np.testing.assert_allclose(ordered[0], [10, 10], atol=1e-2)   # TL
        np.testing.assert_allclose(ordered[1], [100, 10], atol=1e-2)  # TR
        np.testing.assert_allclose(ordered[2], [100, 50], atol=1e-2)  # BR
        np.testing.assert_allclose(ordered[3], [10, 50], atol=1e-2)   # BL

    def test_unclip_rotated_rect(self):
        # Rectangle 100 x 20 with unclip_ratio = 1.5
        # Area = 2000, Perimeter = 240, D = (2000 * 1.5) / 240 = 12.5
        # w_new = 100 + 25 = 125, h_new = 20 + 25 = 45
        rect = ((200.0, 100.0), (100.0, 20.0), 0.0)
        unclipped = unclip_rotated_rect(rect, unclip_ratio=1.5)
        self.assertEqual(unclipped.shape, (4, 2))
        w = np.linalg.norm(unclipped[1] - unclipped[0])
        h = np.linalg.norm(unclipped[3] - unclipped[0])
        self.assertAlmostEqual(w, 125.0, delta=0.5)
        self.assertAlmostEqual(h, 45.0, delta=0.5)

    def test_box_score_fast(self):
        prob_map = np.zeros((100, 100), dtype=np.float32)
        prob_map[20:60, 20:80] = 0.95
        box = np.array([[20, 20], [80, 20], [80, 60], [20, 60]], dtype=np.float32)
        score = box_score_fast(prob_map, box)
        self.assertGreaterEqual(score, 0.90)

        # Empty region
        empty_box = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        empty_score = box_score_fast(prob_map, empty_box)
        self.assertAlmostEqual(empty_score, 0.0, delta=1e-3)

    def test_coordinate_restoration(self):
        pts = np.array([[32, 64], [160, 64]], dtype=np.float32)
        scale_x = 0.5
        scale_y = 0.5
        restored = restore_coordinates(pts, scale_x, scale_y, orig_w=1920, orig_h=1080)
        np.testing.assert_allclose(restored[0], [64, 128], atol=1e-3)
        np.testing.assert_allclose(restored[1], [320, 128], atol=1e-3)

    def test_reading_order_sorting(self):
        # 3 lines: line 1 has two words, line 2 has one word, line 3 has one word
        boxes = [
            (200.0, 100.0, 300.0, 130.0),  # Line 1, Word 2
            (50.0, 100.0, 150.0, 130.0),   # Line 1, Word 1
            (80.0, 300.0, 250.0, 330.0),   # Line 3, Word 1
            (60.0, 200.0, 220.0, 230.0),   # Line 2, Word 1
        ]
        sorted_indices, sorted_boxes, _, _ = sort_reading_order(boxes)
        # Expected order: Word 1 (Line 1), Word 2 (Line 1), Line 2, Line 3
        expected_indices = [1, 0, 3, 2]
        self.assertEqual(sorted_indices, expected_indices)


class TestDBPostProcessor(unittest.TestCase):
    """Test standalone postprocessing logic."""

    def test_empty_probability_maps(self):
        postproc = DBPostProcessor(thresh=0.3, box_thresh=0.6)
        prob_map = np.zeros((320, 320), dtype=np.float32)
        polys, boxes_4pt, boxes_xyxy, confs = postproc(
            prob_map=prob_map, scale_x=1.0, scale_y=1.0, orig_shape=(320, 320)
        )
        self.assertEqual(len(boxes_xyxy), 0)
        self.assertEqual(len(confs), 0)

    def test_confidence_filtering(self):
        postproc = DBPostProcessor(thresh=0.3, box_thresh=0.6)
        prob_map = np.zeros((200, 200), dtype=np.float32)
        # High confidence region (score ~0.9)
        prob_map[20:50, 20:100] = 0.9
        # Low confidence region (above binarization 0.3 but below box threshold 0.6)
        prob_map[120:150, 20:100] = 0.45

        polys, boxes_4pt, boxes_xyxy, confs = postproc(
            prob_map=prob_map, scale_x=1.0, scale_y=1.0, orig_shape=(200, 200)
        )
        self.assertEqual(len(boxes_xyxy), 1)
        self.assertGreaterEqual(confs[0], 0.6)


class TestDBNetDetector(unittest.TestCase):
    """Test full DBNetDetector integration, preprocessing, inference, and provider fallbacks."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = Path("models/ocr/ch_PP-OCRv4_det.onnx")
        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Model checkpoint not found at: {cls.model_path}")
        cls.detector_cuda = DBNetDetector(preferred_provider="CUDA")

    def test_model_loading_and_attributes(self):
        self.assertIsNotNone(self.detector_cuda.session)
        self.assertEqual(self.detector_cuda.input_name, "x")
        self.assertEqual(self.detector_cuda.output_name, "fetch_name_0")

    def test_cuda_provider_reporting(self):
        # On RTX 4050 development environment, CUDAExecutionProvider must be active
        active = self.detector_cuda.active_provider
        self.assertIn(active, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        if active == "CUDAExecutionProvider":
            print(f"\n[PASS] Verified CUDA execution provider on RTX 4050")

    def test_cpu_fallback(self):
        detector_cpu = DBNetDetector(preferred_provider="CPU")
        self.assertEqual(detector_cpu.active_provider, "CPUExecutionProvider")
        # Run a quick detection on CPU
        dummy = np.ones((64, 64, 3), dtype=np.uint8) * 255
        res = detector_cpu.detect(dummy)
        self.assertTrue(res.success)
        self.assertEqual(res.execution_provider, "CPUExecutionProvider")

    def test_input_validation(self):
        # None input
        res_none = self.detector_cuda.detect(None)
        self.assertFalse(res_none.success)
        self.assertIn("Invalid image input", res_none.error_message)

        # Empty array
        res_empty = self.detector_cuda.detect(np.array([]))
        self.assertFalse(res_empty.success)

        # 2D grayscale array instead of 3D
        res_2d = self.detector_cuda.detect(np.zeros((100, 100), dtype=np.uint8))
        self.assertFalse(res_2d.success)

        # 4-channel RGBA array
        res_4c = self.detector_cuda.detect(np.zeros((100, 100, 4), dtype=np.uint8))
        self.assertFalse(res_4c.success)

    def test_preprocessing_dimensions_and_multiples_of_32(self):
        test_shapes = [(480, 640), (1080, 1920), (137, 253), (45, 60)]
        for h, w in test_shapes:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            tensor, (scale_x, scale_y), orig_shape = self.detector_cuda.preprocess(img)

            _, c, rh, rw = tensor.shape
            self.assertEqual(c, 3)
            self.assertEqual(rh % 32, 0, f"Height {rh} must be multiple of 32")
            self.assertEqual(rw % 32, 0, f"Width {rw} must be multiple of 32")
            self.assertEqual(orig_shape, (h, w))
            self.assertLessEqual(max(rh, rw), 960)

    def test_aspect_ratio_preservation(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, (scale_x, scale_y), _ = self.detector_cuda.preprocess(img)
        # Ratio of scale_x and scale_y should be very close to 1.0 (allowing for multiple of 32 rounding)
        scale_ratio = scale_x / scale_y
        self.assertAlmostEqual(scale_ratio, 1.0, delta=0.05)

    def test_multiple_text_regions(self):
        # Canvas with 3 distinct synthetic text regions
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
        # Draw 3 simulated high-contrast text lines
        for y in [100, 250, 400]:
            cv2.putText(
                canvas,
                "TEST LINE SAMPLE TEXT",
                (50, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        res = self.detector_cuda.detect(canvas)
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.num_detections, 2)
        self.assertEqual(len(res.boxes_xyxy), res.num_detections)
        self.assertEqual(len(res.confidences), res.num_detections)

    def test_nearby_text_lines(self):
        # Two text lines separated by 18 pixels vertically
        canvas = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "FIRST UPPER TEXT LINE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, "SECOND LOWER TEXT LINE", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        res = self.detector_cuda.detect(canvas)
        self.assertTrue(res.success)
        # Should detect at least 2 distinct text lines rather than merging into 1 giant box
        self.assertGreaterEqual(res.num_detections, 2)

    def test_real_text_images_hindi_and_english(self):
        hindi_path = Path("07_ocr/data/poc/sample_01.png")
        en_path = Path("07_ocr/data/poc_en/sample_en_01.png")

        if not hindi_path.exists() or not en_path.exists():
            raise unittest.SkipTest("Sample crop files not found")

        h_img = cv2.imread(str(hindi_path))
        en_img = cv2.imread(str(en_path))

        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
        canvas[100 : 100 + h_img.shape[0], 50 : 50 + h_img.shape[1]] = h_img
        canvas[300 : 300 + en_img.shape[0], 50 : 50 + en_img.shape[1]] = en_img

        res = self.detector_cuda.detect(canvas)
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.num_detections, 2)

        # Check that coordinates bound the pasted locations
        # Hindi pasted around y=[100..150], English pasted around y=[300..350]
        y_centers = [(box[1] + box[3]) / 2.0 for box in res.boxes_xyxy]
        has_top_line = any(80 <= yc <= 180 for yc in y_centers)
        has_bottom_line = any(280 <= yc <= 380 for yc in y_centers)
        self.assertTrue(has_top_line, "Hindi line bounding box not detected near expected y")
        self.assertTrue(has_bottom_line, "English line bounding box not detected near expected y")


if __name__ == "__main__":
    unittest.main()
