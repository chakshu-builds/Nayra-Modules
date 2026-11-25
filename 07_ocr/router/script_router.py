"""Standalone Script Router for NAYRA Module 07.

Classifies a text-line image crop into:
- DEVANAGARI
- LATIN
- SHARED_NUMERIC
- UNKNOWN

Without executing prior OCR recognition.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from .shirorekha_filter import ShirorekhaFilter, ShirorekhaResult


class ScriptType(str, Enum):
    """Supported script and character category classes."""
    DEVANAGARI = "DEVANAGARI"
    LATIN = "LATIN"
    SHARED_NUMERIC = "SHARED_NUMERIC"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RouterResult:
    """Structured decision output from ScriptRouter."""
    script: ScriptType
    confidence: float
    decision_reason: str
    shirorekha_confidence: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ScriptRouter:
    """Standalone vision-based script routing engine."""

    def __init__(
        self,
        shirorekha_filter: Optional[ShirorekhaFilter] = None,
        min_laplacian_var: float = 4.0,
        word_gap_ratio: float = 0.40,
        digit_aspect_ratio_max: float = 0.78,
        wide_cap_aspect_ratio_min: float = 0.82,
        min_lowercase_height_ratio: float = 0.35,
        max_lowercase_height_ratio: float = 0.78,
        min_components_for_numeric: int = 2,
    ):
        """Initialize script router with configurable thresholds.

        Args:
            shirorekha_filter: Optional instance of ShirorekhaFilter.
            min_laplacian_var: Minimum Laplacian variance to guard against extreme blur.
            word_gap_ratio: Minimum inter-word space gap as ratio of text height.
            digit_aspect_ratio_max: Maximum aspect ratio for standard isolated digit glyphs.
            wide_cap_aspect_ratio_min: Minimum aspect ratio for wide uppercase Latin letters.
            min_lowercase_height_ratio: Minimum height ratio for Latin lowercase x-height glyphs.
            max_lowercase_height_ratio: Maximum height ratio for Latin lowercase x-height glyphs.
            min_components_for_numeric: Minimum isolated glyphs required for SHARED_NUMERIC.
        """
        self.shirorekha_filter = shirorekha_filter or ShirorekhaFilter()
        self.min_laplacian_var = min_laplacian_var
        self.word_gap_ratio = word_gap_ratio
        self.digit_aspect_ratio_max = digit_aspect_ratio_max
        self.wide_cap_aspect_ratio_min = wide_cap_aspect_ratio_min
        self.min_lowercase_height_ratio = min_lowercase_height_ratio
        self.max_lowercase_height_ratio = max_lowercase_height_ratio
        self.min_components_for_numeric = min_components_for_numeric

    def route(self, img: np.ndarray) -> RouterResult:
        """Route an input text crop to its predicted script.

        Args:
            img: BGR or grayscale numpy image crop.

        Returns:
            RouterResult containing script classification, confidence, and diagnostics.
        """
        # 1. Base input validation
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            return RouterResult(
                script=ScriptType.UNKNOWN,
                confidence=0.0,
                decision_reason="Invalid or empty image array",
                shirorekha_confidence=0.0,
                diagnostics={"error": "invalid_input"},
            )

        h, w = img.shape[:2]
        if h < self.shirorekha_filter.min_crop_height or w < self.shirorekha_filter.min_crop_width:
            return RouterResult(
                script=ScriptType.UNKNOWN,
                confidence=0.0,
                decision_reason=f"Crop dimensions ({w}x{h}) below minimum threshold",
                shirorekha_confidence=0.0,
                diagnostics={"height": h, "width": w},
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # 2. Check contrast / flat image guard
        std_val = float(np.std(gray))
        if std_val < self.shirorekha_filter.min_contrast_std:
            return RouterResult(
                script=ScriptType.UNKNOWN,
                confidence=0.0,
                decision_reason=f"Low contrast or flat image (std={std_val:.2f} < {self.shirorekha_filter.min_contrast_std})",
                shirorekha_confidence=0.0,
                diagnostics={"std": std_val},
            )

        # 3. Check for severe image blur
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < self.min_laplacian_var:
            return RouterResult(
                script=ScriptType.UNKNOWN,
                confidence=0.0,
                decision_reason=f"Severe image blur (Laplacian variance {lap_var:.2f} < {self.min_laplacian_var})",
                shirorekha_confidence=0.0,
                diagnostics={"laplacian_variance": round(lap_var, 2)},
            )

        # 4. Evaluate Shirorekha filter for Devanagari evidence
        shiro_res: ShirorekhaResult = self.shirorekha_filter.analyze(img)

        if not shiro_res.is_valid:
            return RouterResult(
                script=ScriptType.UNKNOWN,
                confidence=0.0,
                decision_reason=f"Rejected by visual pre-filter: {shiro_res.rejection_reason}",
                shirorekha_confidence=0.0,
                diagnostics={"shirorekha_diagnostics": shiro_res.diagnostics},
            )

        # 5. If Shirorekha headline is positively detected, route to DEVANAGARI
        if shiro_res.has_headline:
            reason = (
                f"Devanagari headline detected (run_ratio={shiro_res.run_to_height_ratio:.2f}, "
                f"coverage={shiro_res.line_coverage:.2f}, mass={shiro_res.mass_ratio:.2f})"
            )
            return RouterResult(
                script=ScriptType.DEVANAGARI,
                confidence=shiro_res.confidence,
                decision_reason=reason,
                shirorekha_confidence=shiro_res.confidence,
                diagnostics={
                    "shirorekha_run_ratio": shiro_res.run_to_height_ratio,
                    "shirorekha_coverage": shiro_res.line_coverage,
                    "shirorekha_mass": shiro_res.mass_ratio,
                    "laplacian_variance": round(lap_var, 2),
                },
            )

        # 6. Non-Devanagari branch: Analyze structure to differentiate LATIN vs SHARED_NUMERIC vs UNKNOWN
        return self._classify_non_devanagari(gray, shiro_res, lap_var)

    def _classify_non_devanagari(
        self,
        gray: np.ndarray,
        shiro_res: ShirorekhaResult,
        lap_var: float,
    ) -> RouterResult:
        """Analyze non-headline crop to differentiate Latin, Shared Numeric, or Unknown."""
        mean_val = float(np.mean(gray))
        if mean_val > 127.0:
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cols = np.where(np.sum(bin_img, axis=0) > 0)[0]
        rows = np.where(np.sum(bin_img, axis=1) > 0)[0]
        if len(cols) == 0 or len(rows) == 0:
            return RouterResult(
                script=ScriptType.UNKNOWN,
                confidence=0.0,
                decision_reason="No foreground text pixels found during structure analysis",
                shirorekha_confidence=shiro_res.confidence,
                diagnostics={},
            )

        cropped = bin_img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        text_h, text_w = cropped.shape

        # Extract connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cropped)
        # Filter tiny speckle noise
        valid_stats = [
            s for s in stats[1:]
            if s[cv2.CC_STAT_AREA] >= 6 and s[cv2.CC_STAT_HEIGHT] >= 4
        ]

        if len(valid_stats) == 0:
            return RouterResult(
                script=ScriptType.UNKNOWN,
                confidence=0.0,
                decision_reason="No valid glyph components detected after noise filtering",
                shirorekha_confidence=shiro_res.confidence,
                diagnostics={"text_h": text_h, "text_w": text_w},
            )

        # Conservative guard for single isolated glyph
        if len(valid_stats) == 1:
            return RouterResult(
                script=ScriptType.UNKNOWN,
                confidence=0.40,
                decision_reason="Single isolated glyph (ambiguous between Latin letter and digit)",
                shirorekha_confidence=shiro_res.confidence,
                diagnostics={"component_count": 1, "text_h": text_h, "text_w": text_w},
            )

        sorted_boxes = sorted(valid_stats, key=lambda s: int(s[cv2.CC_STAT_LEFT]))

        # Detect Latin lowercase x-height letters (components in middle zone)
        lowercase_candidates = [
            s
            for s in sorted_boxes
            if (self.min_lowercase_height_ratio * text_h <= s[cv2.CC_STAT_HEIGHT] <= self.max_lowercase_height_ratio * text_h)
            and s[cv2.CC_STAT_TOP] >= 0.15 * text_h
        ]
        has_lowercase = len(lowercase_candidates) >= 2

        # Detect word spacing gaps (standard typographic threshold: >= 0.40 text height)
        gaps = []
        for i in range(len(sorted_boxes) - 1):
            g = int(sorted_boxes[i + 1][cv2.CC_STAT_LEFT]) - (
                int(sorted_boxes[i][cv2.CC_STAT_LEFT]) + int(sorted_boxes[i][cv2.CC_STAT_WIDTH])
            )
            gaps.append(g)
        word_spaces = [g for g in gaps if g >= text_h * self.word_gap_ratio]
        is_multi_word = len(word_spaces) >= 1 and len(sorted_boxes) >= 5

        # Cap-height main components
        main_comps = [s for s in sorted_boxes if s[cv2.CC_STAT_HEIGHT] >= 0.68 * text_h]
        aspect_ratios = [
            float(s[cv2.CC_STAT_WIDTH]) / float(s[cv2.CC_STAT_HEIGHT]) for s in main_comps
        ]

        # Wide uppercase Latin letters (e.g., M, W, G, R, Y, X with AR >= wide_cap_aspect_ratio_min)
        has_wide_cap_letters = any(ar >= self.wide_cap_aspect_ratio_min for ar in aspect_ratios)

        # Digit aspect ratio uniformity check
        all_digit_aspect_ratio = (
            all(ar <= self.digit_aspect_ratio_max for ar in aspect_ratios)
            if aspect_ratios
            else False
        )

        diag = {
            "text_h": text_h,
            "text_w": text_w,
            "glyph_count": len(sorted_boxes),
            "lowercase_glyph_count": len(lowercase_candidates),
            "word_spaces_count": len(word_spaces),
            "main_cap_glyph_count": len(main_comps),
            "max_aspect_ratio": round(max(aspect_ratios), 2) if aspect_ratios else 0.0,
            "laplacian_variance": round(lap_var, 2),
            "shirorekha_confidence": shiro_res.confidence,
        }

        # 7. SHARED_NUMERIC check:
        # Isolated uniform cap-height glyphs with digit aspect ratios, no lowercase, no wide Latin letters, no multi-word phrase
        if (
            not has_lowercase
            and not has_wide_cap_letters
            and not is_multi_word
            and all_digit_aspect_ratio
            and len(main_comps) >= self.min_components_for_numeric
        ):
            conf = 0.90
            reason = (
                f"Isolated digit/symbol glyphs detected (n={len(main_comps)}, "
                f"max_ar={max(aspect_ratios):.2f}, no lowercase or wide letters)"
            )
            return RouterResult(
                script=ScriptType.SHARED_NUMERIC,
                confidence=conf,
                decision_reason=reason,
                shirorekha_confidence=shiro_res.confidence,
                diagnostics=diag,
            )

        # 8. LATIN check:
        # Positive indicators: lowercase letters, wide uppercase letters, multi-word phrases, or extended alphabetical sequence
        if has_lowercase or has_wide_cap_letters or is_multi_word or len(sorted_boxes) >= 6:
            conf = 0.95 if (has_lowercase or is_multi_word or has_wide_cap_letters) else 0.85
            reason = (
                f"Latin alphabetic structure detected (lowercase={has_lowercase}, "
                f"wide_caps={has_wide_cap_letters}, words={len(word_spaces) + 1})"
            )
            return RouterResult(
                script=ScriptType.LATIN,
                confidence=conf,
                decision_reason=reason,
                shirorekha_confidence=shiro_res.confidence,
                diagnostics=diag,
            )

        # 9. Ambiguous / Insufficient evidence fallback
        return RouterResult(
            script=ScriptType.UNKNOWN,
            confidence=0.45,
            decision_reason="Insufficient visual evidence to distinguish Latin vs Shared Numeric",
            shirorekha_confidence=shiro_res.confidence,
            diagnostics=diag,
        )
