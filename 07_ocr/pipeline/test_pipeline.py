"""NAYRA Module 07 — Comprehensive Unit & Integration Test Suite for Unified OCR Pipeline.

Covers:
1. Empty frame & zero-detection handling
2. Invalid input rejection
3. Text cropper: perspective transform, degenerate polygons, auto-rotation
4. Hindi-only scene recognition
5. English-only scene recognition
6. Mixed bilingual scene recognition & language classification
7. Numeric-only text routing & recognition (SHARED_NUMERIC policy)
8. Unknown / ambiguous script handling (UNRESOLVED_SCRIPT)
9. Recognizer failure isolation (safe per-region exception containment)
10. Unicode NFC normalization guarantee
11. Reading-order sorting validation
12. Direct cross-check against standalone Hindi/English PoC outputs
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unicodedata
import unittest
from unittest.mock import patch

import cv2
import numpy as np

# Ensure 07_ocr directory is on sys.path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from pipeline.ocr_engine import OCREngine
from pipeline.reading_order import format_combined_text, sort_regions_reading_order
from pipeline.result_schema import OCRResult, PrimaryLanguage, RegionStatus, TextRegion
from pipeline.text_cropper import extract_perspective_crop
from poc_english_ocr import EnglishOCRModel
from poc_hindi_ocr import HindiOCRModel


class TestTextCropper(unittest.TestCase):
    """Test perspective crop extraction and edge-case handling."""

    def test_normal_perspective_crop(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        img[50:100, 100:300] = 200
        # Clockwise rectangle: TL=(100, 50), TR=(300, 50), BR=(300, 100), BL=(100, 100)
        pts = np.array([[100, 50], [300, 50], [300, 100], [100, 100]], dtype=np.float32)

        crop, err = extract_perspective_crop(img, pts)
        self.assertIsNone(err)
        self.assertIsNotNone(crop)
        self.assertAlmostEqual(crop.shape[1], 200, delta=2)
        self.assertAlmostEqual(crop.shape[0], 50, delta=2)

    def test_degenerate_polygons(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Sub-pixel box (width=0)
        pts_zero = np.array([[10, 10], [10, 10], [10, 20], [10, 20]], dtype=np.float32)
        crop, err = extract_perspective_crop(img, pts_zero)
        self.assertIsNone(crop)
        self.assertIn("Degenerate crop", err)

        # Non-finite (NaN)
        pts_nan = np.array([[10, 10], [np.nan, 10], [20, 20], [10, 20]], dtype=np.float32)
        crop, err = extract_perspective_crop(img, pts_nan)
        self.assertIsNone(crop)
        self.assertIn("non-finite", err)

    def test_vertical_auto_rotation(self):
        # Tall vertical box: width=30, height=120
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        pts = np.array([[50, 20], [80, 20], [80, 140], [50, 140]], dtype=np.float32)
        crop, err = extract_perspective_crop(img, pts, auto_rotate_vertical=True)
        self.assertIsNone(err)
        self.assertIsNotNone(crop)
        # After 90-deg clockwise rotation, width should exceed height
        self.assertGreater(crop.shape[1], crop.shape[0])


class TestReadingOrder(unittest.TestCase):
    """Test reading order sorting and text formatting."""

    def test_reading_order_and_formatting(self):
        poly_dummy = np.zeros((4, 2), dtype=np.float32)
        r1 = TextRegion(1, poly_dummy, (50.0, 50.0, 150.0, 80.0), (30, 100), "LATIN", 0.9, "FIRST", 0.95, RegionStatus.SUCCESS, 0)
        r2 = TextRegion(2, poly_dummy, (170.0, 50.0, 270.0, 80.0), (30, 100), "LATIN", 0.9, "LINE", 0.95, RegionStatus.SUCCESS, 0)
        r3 = TextRegion(3, poly_dummy, (60.0, 150.0, 200.0, 180.0), (30, 140), "LATIN", 0.9, "SECOND", 0.95, RegionStatus.SUCCESS, 0)

        # Pass shuffled
        sorted_regions, grouped_lines = sort_regions_reading_order([r3, r2, r1])
        self.assertEqual(len(grouped_lines), 2)
        self.assertEqual(len(grouped_lines[0]), 2)  # First line: r1, r2
        self.assertEqual(len(grouped_lines[1]), 1)  # Second line: r3
        self.assertEqual(sorted_regions[0].text, "FIRST")
        self.assertEqual(sorted_regions[1].text, "LINE")
        self.assertEqual(sorted_regions[2].text, "SECOND")

        combined = format_combined_text(grouped_lines)
        self.assertEqual(combined, "FIRST LINE\nSECOND")


class TestUnifiedOCREngine(unittest.TestCase):
    """Integration test suite for the complete OCREngine."""

    @classmethod
    def setUpClass(cls):
        # Initialize OCR engine once (reusing persistent ONNX sessions)
        cls.engine = OCREngine(preferred_provider="CUDA")

    def test_empty_frame_and_no_detection(self):
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
        res = self.engine.process_image(blank)
        self.assertTrue(res.success)
        self.assertEqual(res.detection_count, 0)
        self.assertEqual(res.recognition_count, 0)
        self.assertEqual(res.primary_language, PrimaryLanguage.NONE)
        self.assertEqual(res.combined_text, "")

    def test_invalid_inputs(self):
        res_none = self.engine.process_image(None)
        self.assertFalse(res_none.success)
        self.assertIn("Invalid image input", res_none.error_message)

        res_2d = self.engine.process_image(np.zeros((100, 100), dtype=np.uint8))
        self.assertFalse(res_2d.success)
        self.assertIn("Invalid image format", res_2d.error_message)

    def test_hindi_only_scene(self):
        h_crop = cv2.imread("07_ocr/data/poc/sample_01.png")
        if h_crop is None:
            raise unittest.SkipTest("Hindi sample_01.png not found")

        canvas = np.ones((300, 600, 3), dtype=np.uint8) * 255
        canvas[100 : 100 + h_crop.shape[0], 80 : 80 + h_crop.shape[1]] = h_crop

        res = self.engine.process_image(canvas)
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.detection_count, 1)
        self.assertGreaterEqual(res.recognition_count, 1)
        self.assertEqual(res.primary_language, PrimaryLanguage.HINDI)

        texts = [r.text for r in res.regions if r.status == RegionStatus.SUCCESS]
        self.assertTrue(any("नमस्ते" in t or "दुनिया" in t for t in texts))

    def test_english_only_scene(self):
        en_crop = cv2.imread("07_ocr/data/poc_en/sample_en_01.png")  # "EMERGENCY EXIT"
        if en_crop is None:
            raise unittest.SkipTest("English sample_en_01.png not found")

        canvas = np.ones((300, 600, 3), dtype=np.uint8) * 255
        canvas[100 : 100 + en_crop.shape[0], 80 : 80 + en_crop.shape[1]] = en_crop

        res = self.engine.process_image(canvas)
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.detection_count, 1)
        self.assertEqual(res.primary_language, PrimaryLanguage.ENGLISH)

        texts = [r.text for r in res.regions if r.status == RegionStatus.SUCCESS]
        self.assertTrue(any("EMERGENCY" in t or "EXIT" in t for t in texts))

    def test_mixed_bilingual_scene(self):
        h_crop = cv2.imread("07_ocr/data/poc/sample_01.png")  # Hindi
        en_crop = cv2.imread("07_ocr/data/poc_en/sample_en_02.png")  # English "NAYRA ASSISTIVE AI"

        canvas = np.ones((500, 700, 3), dtype=np.uint8) * 255
        canvas[80 : 80 + en_crop.shape[0], 60 : 60 + en_crop.shape[1]] = en_crop
        canvas[240 : 240 + h_crop.shape[0], 60 : 60 + h_crop.shape[1]] = h_crop

        res = self.engine.process_image(canvas)
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.detection_count, 2)
        self.assertEqual(res.primary_language, PrimaryLanguage.MIXED)

        scripts = [r.script for r in res.regions if r.status == RegionStatus.SUCCESS]
        self.assertIn("LATIN", scripts)
        self.assertIn("DEVANAGARI", scripts)

    def test_numeric_only_text(self):
        # Create a synthetic numeric crop "98214"
        canvas = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "98214", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        res = self.engine.process_image(canvas)
        self.assertTrue(res.success)
        if res.detection_count >= 1:
            r = res.regions[0]
            self.assertIn(r.script, ["SHARED_NUMERIC", "LATIN"])
            if r.status == RegionStatus.SUCCESS:
                self.assertIn("98214", r.text)

    def test_unknown_ambiguous_script(self):
        # Flat texture image (below contrast threshold) -> classified as UNKNOWN
        flat_crop = np.ones((40, 120, 3), dtype=np.uint8) * 128
        route_res = self.engine.router.route(flat_crop)
        self.assertEqual(route_res.script.value, "UNKNOWN")

    def test_recognizer_failure_isolation(self):
        """Simulate a failure in Hindi recognition to verify graceful frame degradation."""
        h_crop = cv2.imread("07_ocr/data/poc/sample_01.png")
        canvas = np.ones((300, 600, 3), dtype=np.uint8) * 255
        canvas[100 : 100 + h_crop.shape[0], 80 : 80 + h_crop.shape[1]] = h_crop

        with patch.object(self.engine.hindi_model, "predict", side_effect=RuntimeError("Simulated Device OOM")):
            res = self.engine.process_image(canvas)
            # Frame must still succeed with status recorded per region
            self.assertTrue(res.success)
            self.assertGreaterEqual(res.detection_count, 1)
            self.assertEqual(res.recognition_count, 0)
            self.assertGreaterEqual(res.unresolved_count, 1)
            failed_regions = [r for r in res.regions if r.status == RegionStatus.FAILED_RECOGNITION]
            self.assertGreaterEqual(len(failed_regions), 1)
            self.assertIn("Simulated Device OOM", failed_regions[0].error_message)

    def test_unicode_nfc_normalization(self):
        h_crop = cv2.imread("07_ocr/data/poc/sample_01.png")
        canvas = np.ones((200, 400, 3), dtype=np.uint8) * 255
        canvas[60 : 60 + h_crop.shape[0], 40 : 40 + h_crop.shape[1]] = h_crop

        res = self.engine.process_image(canvas)
        for r in res.regions:
            if r.text:
                self.assertEqual(r.text, unicodedata.normalize("NFC", r.text))
        if res.combined_text:
            self.assertEqual(res.combined_text, unicodedata.normalize("NFC", res.combined_text))

    def test_cross_check_against_standalone_pocs(self):
        """Verify integrated recognizers produce identical output to standalone PoC models."""
        h_crop = cv2.imread("07_ocr/data/poc/sample_01.png")
        en_crop = cv2.imread("07_ocr/data/poc_en/sample_en_01.png")

        # Standalone Hindi PoC
        standalone_hindi = HindiOCRModel()
        stand_h_text, stand_h_conf = standalone_hindi.predict(h_crop)

        # Integrated engine's Hindi model
        engine_h_text, engine_h_conf = self.engine.hindi_model.predict(h_crop)

        self.assertEqual(stand_h_text, engine_h_text)
        self.assertAlmostEqual(stand_h_conf, engine_h_conf, delta=1e-5)

        # Standalone English PoC
        standalone_en = EnglishOCRModel()
        stand_en_text, stand_en_conf = standalone_en.predict(en_crop)

        # Integrated engine's English model
        engine_en_text, engine_en_conf = self.engine.english_model.predict(en_crop)

        self.assertEqual(stand_en_text, engine_en_text)
        self.assertAlmostEqual(stand_en_conf, engine_en_conf, delta=1e-5)

    def test_batched_recognizer_parity(self):
        """Verify predict_batch produces exact text and close confidence to sequential predict."""
        from .batch_recognizer import predict_batch

        crops_hi = [cv2.imread(f"07_ocr/data/poc/sample_0{i}.png") for i in range(1, 4)]
        crops_en = [cv2.imread(f"07_ocr/data/poc_en/sample_en_{i:02d}.png") for i in range(1, 4)]

        # Hindi batch vs sequential
        seq_hi = [self.engine.hindi_model.predict(c) for c in crops_hi]
        batch_hi = predict_batch(self.engine.hindi_model, crops_hi)
        self.assertEqual([s[0] for s in seq_hi], [b[0] for b in batch_hi])
        for s, b in zip(seq_hi, batch_hi):
            self.assertAlmostEqual(s[1], b[1], delta=0.02)

        # English batch vs sequential
        seq_en = [self.engine.english_model.predict(c) for c in crops_en]
        batch_en = predict_batch(self.engine.english_model, crops_en)
        self.assertEqual([s[0] for s in seq_en], [b[0] for b in batch_en])
        for s, b in zip(seq_en, batch_en):
            self.assertAlmostEqual(s[1], b[1], delta=0.02)

    def test_use_batching_flag_parity(self):
        """Verify engine results with use_batching=True vs use_batching=False produce identical text."""
        h_crop = cv2.imread("07_ocr/data/poc/sample_01.png")
        en_crop = cv2.imread("07_ocr/data/poc_en/sample_en_01.png")
        canvas = np.ones((400, 600, 3), dtype=np.uint8) * 255
        canvas[50 : 50 + h_crop.shape[0], 50 : 50 + h_crop.shape[1]] = h_crop
        canvas[200 : 200 + en_crop.shape[0], 50 : 50 + en_crop.shape[1]] = en_crop

        self.engine.use_batching = False
        res_seq = self.engine.process_image(canvas)

        self.engine.use_batching = True
        res_batch = self.engine.process_image(canvas)

        self.assertEqual(res_seq.combined_text, res_batch.combined_text)
        self.assertEqual(res_seq.detection_count, res_batch.detection_count)
        self.assertEqual(res_seq.recognition_count, res_batch.recognition_count)


if __name__ == "__main__":
    unittest.main()
