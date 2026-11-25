"""Morphological Shirorekha (Devanagari headline) filter for NAYRA Module 07.

Detects the presence of the continuous horizontal headline (Shirorekha)
characteristic of Devanagari script in text-line image crops.

Provides:
- Configurable thresholds (no hardcoded magic numbers)
- Measurable continuous confidence scoring
- Robust handling of empty, tiny, low-contrast, and non-text line artifacts
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class ShirorekhaResult:
    """Result of Shirorekha headline analysis."""
    has_headline: bool
    confidence: float
    run_to_height_ratio: float
    line_coverage: float
    mass_ratio: float
    is_valid: bool
    rejection_reason: Optional[str]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ShirorekhaFilter:
    """Morphological filter for detecting Devanagari Shirorekha headlines."""

    def __init__(
        self,
        min_crop_height: int = 8,
        min_crop_width: int = 4,
        min_contrast_std: float = 6.0,
        upper_zone_start: float = 0.10,
        upper_zone_end: float = 0.55,
        kernel_height_ratio: float = 0.40,
        min_kernel_width: int = 10,
        run_ratio_threshold: float = 1.30,
        line_coverage_threshold: float = 0.35,
        mass_ratio_threshold: float = 0.12,
        max_single_line_mass_ratio: float = 0.85,
    ):
        """Initialize filter with configurable thresholds.

        Args:
            min_crop_height: Minimum valid crop height in pixels.
            min_crop_width: Minimum valid crop width in pixels.
            min_contrast_std: Minimum grayscale standard deviation for contrast.
            upper_zone_start: Fractional top offset of upper headline band.
            upper_zone_end: Fractional bottom offset of upper headline band.
            kernel_height_ratio: Ratio of text height used for horizontal kernel width.
            min_kernel_width: Minimum pixel width for opening kernel.
            run_ratio_threshold: Minimum ratio of longest continuous run to text height.
            line_coverage_threshold: Minimum ratio of upper headline row coverage to text width.
            mass_ratio_threshold: Minimum ratio of upper horizontal mass to total foreground.
            max_single_line_mass_ratio: Ratio threshold to reject isolated non-text horizontal lines.
        """
        self.min_crop_height = min_crop_height
        self.min_crop_width = min_crop_width
        self.min_contrast_std = min_contrast_std
        self.upper_zone_start = upper_zone_start
        self.upper_zone_end = upper_zone_end
        self.kernel_height_ratio = kernel_height_ratio
        self.min_kernel_width = min_kernel_width
        self.run_ratio_threshold = run_ratio_threshold
        self.line_coverage_threshold = line_coverage_threshold
        self.mass_ratio_threshold = mass_ratio_threshold
        self.max_single_line_mass_ratio = max_single_line_mass_ratio

    def analyze(self, img: np.ndarray) -> ShirorekhaResult:
        """Analyze a text crop for Shirorekha headline characteristics.

        Args:
            img: BGR or grayscale numpy image crop.

        Returns:
            ShirorekhaResult with detection boolean, confidence, and metrics.
        """
        # 1. Input validation
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            return ShirorekhaResult(
                has_headline=False,
                confidence=0.0,
                run_to_height_ratio=0.0,
                line_coverage=0.0,
                mass_ratio=0.0,
                is_valid=False,
                rejection_reason="Empty or invalid image array",
                diagnostics={},
            )

        h, w = img.shape[:2]
        if h < self.min_crop_height or w < self.min_crop_width:
            return ShirorekhaResult(
                has_headline=False,
                confidence=0.0,
                run_to_height_ratio=0.0,
                line_coverage=0.0,
                mass_ratio=0.0,
                is_valid=False,
                rejection_reason=f"Crop dimensions ({w}x{h}) below minimum ({self.min_crop_width}x{self.min_crop_height})",
                diagnostics={"orig_h": h, "orig_w": w},
            )

        # 2. Grayscale conversion and contrast check
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        std_val = float(np.std(gray))
        if std_val < self.min_contrast_std:
            return ShirorekhaResult(
                has_headline=False,
                confidence=0.0,
                run_to_height_ratio=0.0,
                line_coverage=0.0,
                mass_ratio=0.0,
                is_valid=False,
                rejection_reason=f"Low contrast or flat image (std={std_val:.2f} < {self.min_contrast_std})",
                diagnostics={"std": std_val},
            )

        # 3. Adaptive binarization (foreground = white = 255)
        mean_val = float(np.mean(gray))
        if mean_val > 127.0:
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Text bounding box trimming (remove surrounding whitespace margins)
        col_proj = np.sum(bin_img, axis=0)
        cols = np.where(col_proj > 0)[0]
        row_proj = np.sum(bin_img, axis=1)
        rows = np.where(row_proj > 0)[0]

        if len(cols) == 0 or len(rows) == 0:
            return ShirorekhaResult(
                has_headline=False,
                confidence=0.0,
                run_to_height_ratio=0.0,
                line_coverage=0.0,
                mass_ratio=0.0,
                is_valid=False,
                rejection_reason="No foreground text pixels detected",
                diagnostics={"std": std_val},
            )

        y1, y2 = int(rows[0]), int(rows[-1])
        x1, x2 = int(cols[0]), int(cols[-1])
        text_h = y2 - y1 + 1
        text_w = x2 - x1 + 1

        if text_h < self.min_crop_height or text_w < self.min_crop_width:
            return ShirorekhaResult(
                has_headline=False,
                confidence=0.0,
                run_to_height_ratio=0.0,
                line_coverage=0.0,
                mass_ratio=0.0,
                is_valid=False,
                rejection_reason=f"Foreground text box ({text_w}x{text_h}) below minimum",
                diagnostics={"text_h": text_h, "text_w": text_w},
            )

        cropped_bin = bin_img[y1 : y2 + 1, x1 : x2 + 1]
        total_fg = int(np.sum(cropped_bin > 0))

        # 5. Non-text artifact guard: detect plain solid horizontal line without character body
        row_sums = np.sum(cropped_bin > 0, axis=1)
        max_row_sum = int(np.max(row_sums))
        if total_fg > 0 and (max_row_sum / float(total_fg)) > self.max_single_line_mass_ratio and text_h <= 8:
            return ShirorekhaResult(
                has_headline=False,
                confidence=0.0,
                run_to_height_ratio=0.0,
                line_coverage=0.0,
                mass_ratio=0.0,
                is_valid=False,
                rejection_reason="Artifact: solid horizontal line without vertical character strokes",
                diagnostics={"max_row_ratio": round(max_row_sum / float(total_fg), 3)},
            )

        # 6. Morphological horizontal opening
        k_w = max(self.min_kernel_width, int(round(text_h * self.kernel_height_ratio)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        opened = cv2.morphologyEx(cropped_bin, cv2.MORPH_OPEN, kernel)

        # 7. Upper zone analysis
        u1 = max(0, int(round(text_h * self.upper_zone_start)))
        u2 = min(text_h, int(round(text_h * self.upper_zone_end)))
        upper_opened = opened[u1:u2, :]
        upper_mass = int(np.sum(upper_opened > 0))

        # 8. Compute longest continuous horizontal run in upper zone
        upper_cropped = cropped_bin[u1:u2, :]
        max_run = 0
        for r in range(upper_cropped.shape[0]):
            row = upper_cropped[r, :]
            cur_run = 0
            for val in row:
                if val > 0:
                    cur_run += 1
                else:
                    if cur_run > max_run:
                        max_run = cur_run
                    cur_run = 0
            if cur_run > max_run:
                max_run = cur_run

        run_ratio = float(max_run) / float(text_h) if text_h > 0 else 0.0
        line_coverage = (
            float(np.max(np.sum(upper_opened > 0, axis=1))) / float(text_w)
            if (text_w > 0 and upper_opened.size > 0)
            else 0.0
        )
        mass_ratio = float(upper_mass) / float(total_fg) if total_fg > 0 else 0.0

        # 9. Multi-criteria headline verification
        passes_run = run_ratio >= self.run_ratio_threshold
        passes_cov = line_coverage >= self.line_coverage_threshold
        passes_mass = mass_ratio >= self.mass_ratio_threshold

        has_headline = passes_run and passes_cov and passes_mass

        # 10. Continuous confidence scoring in [0.0, 1.0]
        if has_headline:
            score = (
                0.50
                + 0.25 * min(1.0, (run_ratio - self.run_ratio_threshold) / 1.5)
                + 0.15 * min(1.0, (line_coverage - self.line_coverage_threshold) / 0.4)
                + 0.10 * min(1.0, (mass_ratio - self.mass_ratio_threshold) / 0.1)
            )
            confidence = round(min(0.99, max(0.60, score)), 4)
        else:
            score = (
                0.50
                - 0.25 * min(1.0, (self.run_ratio_threshold - run_ratio) / 0.8)
                - 0.15 * min(1.0, (self.line_coverage_threshold - line_coverage) / 0.25)
            )
            confidence = round(max(0.01, min(0.49, score)), 4)

        diagnostics = {
            "text_h": text_h,
            "text_w": text_w,
            "total_fg": total_fg,
            "max_run_px": max_run,
            "kernel_w": k_w,
            "upper_mass": upper_mass,
            "std": round(std_val, 2),
            "passes_run": passes_run,
            "passes_cov": passes_cov,
            "passes_mass": passes_mass,
        }

        return ShirorekhaResult(
            has_headline=has_headline,
            confidence=confidence,
            run_to_height_ratio=round(run_ratio, 3),
            line_coverage=round(line_coverage, 3),
            mass_ratio=round(mass_ratio, 3),
            is_valid=True,
            rejection_reason=None,
            diagnostics=diagnostics,
        )
