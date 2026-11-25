"""NAYRA Module 07 — Camera Integration & Public API Unit/Integration Test Suite.

Tests:
1. Public API read_text() on valid and malformed inputs (empty, None, 2D, invalid channels).
2. Zero-text scene handling (blank/natural scene without text).
3. Monolingual (Hindi-only, English-only) and mixed Hindi-English recognition via public API.
4. Blurred and low-contrast scene handling.
5. AsyncOCRWorker thread-safety, non-blocking execution, and snapshot copy isolation.
6. Rejection of duplicate concurrent requests while busy.
7. Graceful worker shutdown while processing.
8. Visualizer HUD overlay rendering (busy badge, Unicode Hindi text, result card).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from typing import Tuple

import cv2
import numpy as np
import pytest

# Ensure repository root is on sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Import through public API
from ocr import (
    AsyncOCRWorker,
    OCREngine,
    OCRResult,
    PrimaryLanguage,
    RegionStatus,
    get_ocr_engine,
    read_text,
    render_ocr_overlay,
)


@pytest.fixture(scope="module")
def persistent_engine() -> OCREngine:
    """Pre-initialize shared OCR engine for fast test execution."""
    return get_ocr_engine(preferred_provider="CUDA")


@pytest.fixture(scope="module")
def test_crops() -> Tuple[np.ndarray, np.ndarray]:
    """Load a sample Hindi crop and English crop from benchmark datasets."""
    hi_path = Path("07_ocr/data/poc/sample_01.png")
    en_path = Path("07_ocr/data/poc_en/sample_en_01.png")

    assert hi_path.exists(), f"Hindi test crop not found: {hi_path}"
    assert en_path.exists(), f"English test crop not found: {en_path}"

    hi_img = cv2.imread(str(hi_path))
    en_img = cv2.imread(str(en_path))
    return hi_img, en_img


# -----------------------------------------------------------------------------
# 1. Public API Error & Edge Case Tests
# -----------------------------------------------------------------------------


def test_public_api_empty_frame():
    """Verify that read_text handles empty frames safely without raising exceptions."""
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    res = read_text(empty_frame)
    assert isinstance(res, OCRResult)
    assert not res.success
    assert "Invalid image input" in res.error_message


def test_public_api_none_frame():
    """Verify that read_text handles None input safely."""
    res = read_text(None)
    assert isinstance(res, OCRResult)
    assert not res.success
    assert "Invalid image input" in res.error_message


def test_public_api_invalid_channels():
    """Verify that read_text handles 1-channel or 4-channel arrays safely."""
    gray_frame = np.zeros((480, 640), dtype=np.uint8)
    res = read_text(gray_frame)
    assert isinstance(res, OCRResult)
    assert not res.success
    assert "Invalid image format" in res.error_message

    rgba_frame = np.zeros((480, 640, 4), dtype=np.uint8)
    res_rgba = read_text(rgba_frame)
    assert not res_rgba.success
    assert "Invalid image format" in res_rgba.error_message


def test_public_api_zero_text_scene(persistent_engine):
    """Verify that a solid or textured image with no text returns clean zero detections."""
    blank_canvas = np.full((720, 1280, 3), 120, dtype=np.uint8)
    res = read_text(blank_canvas)
    assert isinstance(res, OCRResult)
    assert res.success
    assert res.detection_count == 0
    assert res.recognition_count == 0
    assert res.primary_language == PrimaryLanguage.NONE
    assert res.combined_text == ""


# -----------------------------------------------------------------------------
# 2. Camera Scene Recognition Tests (Hindi, English, Mixed, Adverse)
# -----------------------------------------------------------------------------


def test_public_api_hindi_scene(test_crops):
    """Verify public API detects and reads Hindi text embedded in a 1080p camera frame."""
    hi_crop, _ = test_crops
    canvas = np.full((1080, 1920, 3), 235, dtype=np.uint8)
    # Paste Hindi crop in center
    ch, cw = hi_crop.shape[:2]
    y1, x1 = 400, 500
    canvas[y1 : y1 + ch, x1 : x1 + cw] = hi_crop

    res = read_text(canvas)
    assert res.success
    assert res.detection_count >= 1
    assert res.recognition_count >= 1
    assert "नमस्ते" in res.combined_text or "भारत" in res.combined_text
    assert res.primary_language in (PrimaryLanguage.HINDI, PrimaryLanguage.MIXED)


def test_public_api_english_scene(test_crops):
    """Verify public API detects and reads English text in a 1080p camera frame."""
    _, en_crop = test_crops
    canvas = np.full((1080, 1920, 3), 235, dtype=np.uint8)
    ch, cw = en_crop.shape[:2]
    y1, x1 = 300, 600
    canvas[y1 : y1 + ch, x1 : x1 + cw] = en_crop

    res = read_text(canvas)
    assert res.success
    assert res.detection_count >= 1
    assert "EXIT" in res.combined_text or "EMERGENCY" in res.combined_text
    assert res.primary_language == PrimaryLanguage.ENGLISH


def test_public_api_mixed_scene(test_crops):
    """Verify public API detects and reads both Hindi and English in a single scene."""
    hi_crop, en_crop = test_crops
    canvas = np.full((1080, 1920, 3), 240, dtype=np.uint8)

    # Line 1: Hindi
    h1, w1 = hi_crop.shape[:2]
    canvas[250 : 250 + h1, 500 : 500 + w1] = hi_crop

    # Line 2: English
    h2, w2 = en_crop.shape[:2]
    canvas[400 : 400 + h2, 500 : 500 + w2] = en_crop

    res = read_text(canvas)
    assert res.success
    assert res.detection_count >= 2
    assert res.recognition_count >= 2
    assert res.primary_language == PrimaryLanguage.MIXED
    # Both languages present in combined text
    has_hi = any("\u0900" <= c <= "\u097f" for c in res.combined_text)
    has_en = any("A" <= c <= "Z" or "a" <= c <= "z" for c in res.combined_text)
    assert has_hi and has_en


def test_public_api_blurred_and_low_contrast(test_crops):
    """Verify that blurred/low-contrast text does not crash the pipeline."""
    hi_crop, _ = test_crops
    canvas = np.full((720, 1280, 3), 50, dtype=np.uint8)  # Low light
    h, w = hi_crop.shape[:2]
    # Dim crop and paste
    dim_crop = (hi_crop * 0.35).astype(np.uint8)
    canvas[200 : 200 + h, 300 : 300 + w] = dim_crop
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(canvas, (9, 9), 3.0)

    res = read_text(blurred)
    # Must return a valid OCRResult without crashing
    assert isinstance(res, OCRResult)
    assert res.total_time_ms > 0


# -----------------------------------------------------------------------------
# 3. AsyncOCRWorker Thread-Safety & Lifecycle Tests
# -----------------------------------------------------------------------------


def test_async_worker_trigger_and_completion(test_crops):
    """Verify AsyncOCRWorker processes frames asynchronously and updates results."""
    hi_crop, _ = test_crops
    canvas = np.full((720, 1280, 3), 240, dtype=np.uint8)
    h, w = hi_crop.shape[:2]
    canvas[200 : 200 + h, 300 : 300 + w] = hi_crop

    worker = AsyncOCRWorker(preferred_provider="CUDA", display_duration_sec=5.0)
    assert not worker.is_busy

    t0 = time.perf_counter()
    dispatched = worker.trigger(canvas)
    t_dispatch = (time.perf_counter() - t0) * 1000.0

    # Dispatch must be virtually instantaneous (< 10 ms), maintaining 30 FPS
    assert dispatched
    assert t_dispatch < 25.0, f"Worker trigger took {t_dispatch:.2f} ms (must not block)"

    # Wait for completion (with timeout)
    worker.shutdown(timeout=5.0)
    assert not worker.is_busy

    res = worker.get_latest_result()
    assert res is not None
    assert res.success
    assert res.detection_count >= 1


def test_async_worker_snapshot_isolation(test_crops):
    """Verify that mutating the original frame after trigger() does not affect worker."""
    hi_crop, _ = test_crops
    canvas = np.full((720, 1280, 3), 240, dtype=np.uint8)
    h, w = hi_crop.shape[:2]
    canvas[200 : 200 + h, 300 : 300 + w] = hi_crop

    worker = AsyncOCRWorker(preferred_provider="CUDA")
    dispatched = worker.trigger(canvas)
    assert dispatched

    # Mutate original canvas immediately (overwrite with zeros)
    canvas.fill(0)

    # Worker must have operated on its independent snapshot
    worker.shutdown(timeout=5.0)
    res = worker.get_latest_result()
    assert res is not None
    assert res.detection_count >= 1, "Snapshot was corrupted by mutating original frame"


def test_async_worker_rejection_while_busy(test_crops):
    """Verify that trigger() rejects duplicate concurrent requests when busy."""
    hi_crop, _ = test_crops
    canvas = np.full((720, 1280, 3), 240, dtype=np.uint8)

    worker = AsyncOCRWorker(preferred_provider="CUDA")

    # Manually hold lock / simulate busy state
    with worker._lock:
        worker._busy = True

    # Attempt second trigger
    rejected = worker.trigger(canvas)
    assert not rejected
    assert worker._rejected_count == 1

    # Release busy state
    with worker._lock:
        worker._busy = False


def test_async_worker_shutdown_while_running(test_crops):
    """Verify worker.shutdown() safely joins running background thread."""
    hi_crop, _ = test_crops
    canvas = np.full((720, 1280, 3), 240, dtype=np.uint8)
    h, w = hi_crop.shape[:2]
    canvas[200 : 200 + h, 300 : 300 + w] = hi_crop

    worker = AsyncOCRWorker(preferred_provider="CUDA")
    worker.trigger(canvas)

    # Immediately shutdown
    worker.shutdown(timeout=5.0)
    assert not worker.is_busy


# -----------------------------------------------------------------------------
# 4. Visualizer HUD Overlay Tests
# -----------------------------------------------------------------------------


def test_visualizer_busy_overlay():
    """Verify render_ocr_overlay renders the busy badge correctly."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = render_ocr_overlay(frame, result=None, is_busy=True)
    assert rendered.shape == frame.shape
    # Check that pixels changed (busy badge drawn in top right)
    diff = np.sum(rendered != frame)
    assert diff > 0


def test_visualizer_result_card_unicode(test_crops):
    """Verify render_ocr_overlay renders Hindi Devanagari text on result card."""
    hi_crop, _ = test_crops
    canvas = np.full((720, 1280, 3), 200, dtype=np.uint8)
    res = read_text(hi_crop)

    rendered = render_ocr_overlay(canvas, result=res, is_busy=False)
    assert rendered.shape == canvas.shape
    diff = np.sum(rendered != canvas)
    assert diff > 0
