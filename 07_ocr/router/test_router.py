"""Unit tests for NAYRA Module 07 Script Router.

Tests:
- Devanagari headline detection
- Latin text crop routing
- Shared numeric crop classification
- Ambiguous single-glyph handling
- Invalid, empty, and tiny inputs
- Confidence and decision-reason consistency
- Deterministic routing across identical runs
"""

import unittest
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .shirorekha_filter import ShirorekhaFilter, ShirorekhaResult
from .script_router import ScriptRouter, ScriptType, RouterResult


class TestScriptRouter(unittest.TestCase):
    """Test suite for ShirorekhaFilter and ScriptRouter."""

    @classmethod
    def setUpClass(cls):
        cls.filter = ShirorekhaFilter()
        cls.router = ScriptRouter(shirorekha_filter=cls.filter)
        try:
            cls.font_en = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 36)
        except IOError:
            cls.font_en = ImageFont.load_default()

        try:
            cls.font_hi = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc", 36)
        except IOError:
            cls.font_hi = None

    def _render_text(self, text: str, font, pad_x: int = 15, pad_y: int = 10) -> np.ndarray:
        """Helper to render synthetic text crop."""
        dummy = Image.new("RGB", (10, 10))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font)
        w = (bbox[2] - bbox[0]) + pad_x * 2
        h = max(36, (bbox[3] - bbox[1]) + pad_y * 2)
        im = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(0, 0, 0))
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    def test_invalid_and_empty_inputs(self):
        """Verify graceful handling of None, empty, and malformed images."""
        # None input
        res_none = self.router.route(None)
        self.assertEqual(res_none.script, ScriptType.UNKNOWN)
        self.assertEqual(res_none.confidence, 0.0)

        # Empty array
        res_empty = self.router.route(np.array([]))
        self.assertEqual(res_empty.script, ScriptType.UNKNOWN)
        self.assertEqual(res_empty.confidence, 0.0)

        # 0x0 image
        res_zero = self.router.route(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertEqual(res_zero.script, ScriptType.UNKNOWN)
        self.assertEqual(res_zero.confidence, 0.0)

    def test_very_small_inputs(self):
        """Verify crops smaller than minimum threshold return UNKNOWN."""
        tiny_img = np.ones((6, 6, 3), dtype=np.uint8) * 255
        res = self.router.route(tiny_img)
        self.assertEqual(res.script, ScriptType.UNKNOWN)
        self.assertEqual(res.confidence, 0.0)
        self.assertIn("minimum threshold", res.decision_reason)

    def test_low_contrast_flat_image(self):
        """Verify flat/solid color crops return UNKNOWN."""
        flat_img = np.ones((50, 100, 3), dtype=np.uint8) * 128
        res = self.router.route(flat_img)
        self.assertEqual(res.script, ScriptType.UNKNOWN)
        self.assertEqual(res.confidence, 0.0)

    def test_solid_horizontal_line_non_text_artifact(self):
        """Verify a plain horizontal line/separator rule is not classified as Devanagari."""
        line_img = np.zeros((8, 120, 3), dtype=np.uint8)
        # White line in the middle
        line_img[3:5, :] = 255
        res = self.router.route(line_img)
        self.assertEqual(res.script, ScriptType.UNKNOWN)
        self.assertNotEqual(res.script, ScriptType.DEVANAGARI)

    def test_devanagari_headline_detection(self):
        """Verify Devanagari sample produces DEVANAGARI classification with high confidence."""
        if self.font_hi is None:
            self.skipTest("Devanagari font Nirmala not available on system")

        img = self._render_text("नमस्ते भारत", self.font_hi)
        res = self.router.route(img)
        self.assertEqual(res.script, ScriptType.DEVANAGARI)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertGreater(res.shirorekha_confidence, 0.70)
        self.assertIn("Devanagari headline detected", res.decision_reason)

    def test_latin_crop_handling(self):
        """Verify Latin text crops route to LATIN with high confidence."""
        # Uppercase multi-word
        img_upper = self._render_text("EMERGENCY EXIT", self.font_en)
        res_upper = self.router.route(img_upper)
        self.assertEqual(res_upper.script, ScriptType.LATIN)
        self.assertGreaterEqual(res_upper.confidence, 0.80)

        # Lowercase sentence
        img_lower = self._render_text("please keep clear", self.font_en)
        res_lower = self.router.route(img_lower)
        self.assertEqual(res_lower.script, ScriptType.LATIN)
        self.assertGreaterEqual(res_lower.confidence, 0.85)

    def test_shared_numeric_classification(self):
        """Verify pure numeric strings route to SHARED_NUMERIC."""
        num_strings = ["12345", "9876543210", "2026", "402", "500"]
        for num_s in num_strings:
            img = self._render_text(num_s, self.font_en)
            res = self.router.route(img)
            self.assertEqual(
                res.script,
                ScriptType.SHARED_NUMERIC,
                f"Failed on numeric string: {num_s}"
            )
            self.assertGreaterEqual(res.confidence, 0.85)
            self.assertIn("digit", res.decision_reason.lower())

    def test_ambiguous_single_glyph(self):
        """Verify single isolated characters (e.g., '1', 'I', 'O', '0') return UNKNOWN."""
        for glyph in ["1", "0", "I", "O"]:
            img = self._render_text(glyph, self.font_en, pad_x=20, pad_y=15)
            res = self.router.route(img)
            self.assertEqual(
                res.script,
                ScriptType.UNKNOWN,
                f"Glyph '{glyph}' should be conservative UNKNOWN"
            )
            self.assertIn("isolated glyph", res.decision_reason.lower())

    def test_deterministic_routing(self):
        """Verify identical inputs yield strictly deterministic results."""
        img = self._render_text("Gate 4B", self.font_en)
        res1 = self.router.route(img)
        res2 = self.router.route(img)
        self.assertEqual(res1.script, res2.script)
        self.assertEqual(res1.confidence, res2.confidence)
        self.assertEqual(res1.decision_reason, res2.decision_reason)
        self.assertEqual(res1.shirorekha_confidence, res2.shirorekha_confidence)

    def test_diagnostics_consistency(self):
        """Verify output contains complete diagnostic metadata."""
        img = self._render_text("Sample Text", self.font_en)
        res = self.router.route(img)
        self.assertIsInstance(res.diagnostics, dict)
        self.assertIn("text_h", res.diagnostics)
        self.assertIn("text_w", res.diagnostics)
        self.assertIn("laplacian_variance", res.diagnostics)


if __name__ == "__main__":
    unittest.main()
