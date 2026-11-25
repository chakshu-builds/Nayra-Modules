"""NAYRA Module 07 — Visual HUD & Overlay Renderer for OCR.

Renders:
1. Active processing indicator when OCR is running in background.
2. Bounding polygons for detected text lines.
3. Clean, high-legibility HUD card displaying primary language, latency,
   and recognized Hindi / English Unicode text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .pipeline.result_schema import OCRResult, PrimaryLanguage, RegionStatus
except (ImportError, ValueError):
    from pipeline.result_schema import OCRResult, PrimaryLanguage, RegionStatus

logger = logging.getLogger("NAYRA-OCRVisualizer")

# Cache TrueType fonts
_FONT_DEV: Optional[ImageFont.FreeTypeFont] = None
_FONT_LATIN: Optional[ImageFont.FreeTypeFont] = None


def _get_font(font_size: int = 20) -> Optional[ImageFont.FreeTypeFont]:
    """Load TrueType font supporting Devanagari and Latin Unicode scripts."""
    global _FONT_DEV
    if _FONT_DEV is not None and _FONT_DEV.size == font_size:
        return _FONT_DEV

    candidate_fonts = [
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/nirmala.ttf",
        "C:/Windows/Fonts/mangal.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]

    for p in candidate_fonts:
        if Path(p).exists():
            try:
                _FONT_DEV = ImageFont.truetype(p, font_size)
                return _FONT_DEV
            except Exception:
                continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def render_ocr_overlay(
    frame: np.ndarray,
    result: Optional[OCRResult] = None,
    is_busy: bool = False,
) -> np.ndarray:
    """Render OCR indicators, detected region polygons, and result card on frame.

    Args:
        frame: OpenCV BGR uint8 NumPy array.
        result: OCRResult to render (if available and within display window).
        is_busy: Whether OCR inference is currently active in the background.

    Returns:
        Frame with OCR overlay elements composited.
    """
    if frame is None or frame.size == 0:
        return frame

    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # -------------------------------------------------------------------------
    # 1. Active Processing Indicator (Busy Badge)
    # -------------------------------------------------------------------------
    if is_busy:
        badge_w, badge_h = 240, 34
        badge_x = w - badge_w - 14
        badge_y = 50

        # Draw semi-transparent glowing badge
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            (20, 20, 20),
            -1,
        )
        cv2.rectangle(
            overlay,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            (0, 190, 255),
            2,
        )
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        # Pulse indicator circle
        cv2.circle(canvas, (badge_x + 18, badge_y + 17), 7, (0, 190, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "OCR READING TEXT...",
            (badge_x + 34, badge_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------------
    # 2. Result Overlay (Bounding Boxes & HUD Result Card)
    # -------------------------------------------------------------------------
    if result is None:
        return canvas

    # Draw detected text regions
    if result.success and result.regions:
        for reg in result.regions:
            poly = reg.polygon.astype(np.int32)
            if reg.status == RegionStatus.SUCCESS:
                color = (0, 220, 100) if reg.script == "DEVANAGARI" else (240, 200, 0)
                thickness = 2
            else:
                color = (0, 140, 255)
                thickness = 1

            cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

            # Draw small tag above polygon
            top_pt = poly[np.argmin(poly[:, 1])]
            tag_text = f"#{reg.reading_order + 1} {reg.script[:2]}"
            cv2.putText(
                canvas,
                tag_text,
                (max(0, top_pt[0] - 2), max(14, top_pt[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    # Draw Bottom Result Panel Card (above controls banner at h - 45)
    card_h = 120
    card_w = min(w - 24, 760)
    card_x = 12
    card_y = max(55, h - card_h - 32)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (70, 70, 70), 1)

    # Card Title Row
    if result.success:
        primary_lang = result.primary_language.value
        title_str = (
            f"[OCR RESULT] Lang: {primary_lang}  |  "
            f"Lines: {result.recognition_count}/{result.detection_count}  |  "
            f"Latency: {result.total_time_ms:.0f}ms"
        )
        title_color = (0, 220, 120)
    else:
        title_str = f"[OCR FAILED] {result.error_message[:60]}"
        title_color = (0, 90, 255)

    cv2.putText(canvas, title_str, (card_x + 14, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, title_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (card_x, card_y + 34), (card_x + card_w, card_y + 34), (50, 50, 50), 1)

    # Content Area: Unicode Hindi & English Text Rendering via PIL
    text_content = result.combined_text.strip() if result.success else ""
    if not text_content and result.success:
        text_content = "(No text detected in scene)"

    lines = [line.strip() for line in text_content.splitlines() if line.strip()][:3]
    if not lines and not result.success:
        lines = [result.error_message or "OCR processing failed."]

    font = _get_font(font_size=19)
    if font is not None:
        try:
            # Convert canvas region to PIL Image for Unicode font rasterization
            pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            y_offset = card_y + 40
            for line in lines:
                display_line = line if len(line) <= 65 else line[:62] + "..."
                draw.text(
                    (card_x + 14, y_offset),
                    display_line,
                    font=font,
                    fill=(245, 245, 245),
                )
                y_offset += 24

            # Convert back to OpenCV BGR
            canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as pil_err:
            logger.debug("PIL text rendering fallback to OpenCV: %s", pil_err)
            _render_text_fallback_cv2(canvas, lines, card_x + 14, card_y + 58)
    else:
        _render_text_fallback_cv2(canvas, lines, card_x + 14, card_y + 58)

    return canvas


def _render_text_fallback_cv2(canvas: np.ndarray, lines: list[str], start_x: int, start_y: int) -> None:
    """ASCII fallback if TrueType font rasterization is unavailable."""
    curr_y = start_y
    for line in lines:
        ascii_line = line.encode("ascii", "replace").decode("ascii")
        display_line = ascii_line if len(ascii_line) <= 60 else ascii_line[:57] + "..."
        cv2.putText(
            canvas,
            display_line,
            (start_x, curr_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        curr_y += 24
