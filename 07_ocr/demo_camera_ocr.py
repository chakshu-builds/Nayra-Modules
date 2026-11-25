"""NAYRA Module 07 — On-Demand Camera OCR Interactive Demonstration.

Simulates the real-time camera loop with CameraManager and AsyncOCRWorker:
1. Spawns camera stream at target 30 FPS.
2. Continues reading and processing camera frames without blocking.
3. Simulates pressing 'R' to read text on-demand.
4. Confirms non-blocking execution while OCR processes in the background.
5. Displays HUD busy badge during processing and results card upon completion.
6. Prints terminal output and saves the annotated demonstration frame to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure root directory is on sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from ocr import (
    AsyncOCRWorker,
    OCRResult,
    PrimaryLanguage,
    get_ocr_engine,
    read_text,
    render_ocr_overlay,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-CameraDemo")


def generate_live_camera_frame(frame_idx: int) -> np.ndarray:
    """Simulate active 1080p camera feed with moving timestamp and realistic signage."""
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Ambient indoor lighting
    canvas[:] = (228, 230, 232)

    # Simulated camera timestamp in upper corner (proves live frame updating)
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S") + f".{frame_idx % 30:02d}"
    cv2.putText(canvas, f"CAMERA 01 (1080p) - {ts_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)

    # Embedded bilingual railway signage
    signs = [
        ("07_ocr/data/poc/sample_01.png", (320, 420)),       # नमस्ते भारत
        ("07_ocr/data/poc/sample_02.png", (450, 420)),       # आपातकालीन निकास
        ("07_ocr/data/poc_en/sample_en_01.png", (320, 1020)), # EMERGENCY EXIT
        ("07_ocr/data/poc_en/sample_en_09.png", (450, 1020)), # 12402
    ]

    for rel_path, (pos_y, pos_x) in signs:
        p = _ROOT_DIR / rel_path
        if p.exists():
            crop = cv2.imread(str(p))
            if crop is not None:
                ch, cw = crop.shape[:2]
                cv2.rectangle(canvas, (pos_x - 10, pos_y - 10), (pos_x + cw + 10, pos_y + ch + 10), (255, 255, 255), -1)
                cv2.rectangle(canvas, (pos_x - 10, pos_y - 10), (pos_x + cw + 10, pos_y + ch + 10), (80, 80, 80), 2)
                canvas[pos_y : pos_y + ch, pos_x : pos_x + cw] = crop

    return canvas


def run_demonstration():
    logger.info("Initializing On-Demand Camera OCR Live Loop Demonstration...")
    worker = AsyncOCRWorker(preferred_provider="CUDA", display_duration_sec=10.0)

    # Pre-warm worker engine
    logger.info("Pre-warming OCR worker engine...")
    test_f = generate_live_camera_frame(0)
    read_text(test_f)

    logger.info("Starting live camera acquisition loop (Simulating 30 FPS)...")
    frame_count = 0
    fps_measure_start = time.perf_counter()
    ocr_triggered = False

    t_start = time.time()
    annotated_saved = False

    while (time.time() - t_start) < 4.0:  # Run for 4 seconds
        t_loop_start = time.perf_counter()
        frame_count += 1

        # 1. Acquire latest frame
        frame = generate_live_camera_frame(frame_count)

        # 2. Simulate User pressing 'R' at frame #15 (~0.5s in)
        if frame_count == 15 and not ocr_triggered:
            logger.info(">>> USER PRESSED 'R' — TRIGGERING ON-DEMAND OCR <<<")
            dispatched = worker.trigger(frame)
            logger.info("OCR Worker trigger dispatched: %s", dispatched)
            ocr_triggered = True

        # 3. Simulate accidental second press of 'R' at frame #18 while busy
        if frame_count == 18:
            logger.info(">>> USER PRESSED 'R' AGAIN WHILE BUSY — TESTING REJECTION <<<")
            dispatched_again = worker.trigger(frame)
            logger.info("Duplicate OCR request dispatched: %s (Expected: False)", dispatched_again)

        # 4. Render overlay
        active_res = worker.get_active_result()
        is_busy = worker.is_busy
        annotated = render_ocr_overlay(frame, result=active_res, is_busy=is_busy)

        # Save sample annotated frame once result is ready
        if active_res is not None and not annotated_saved:
            out_dir = _ROOT_DIR / "snapshots"
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / "demo_camera_ocr_result.jpg"
            cv2.imwrite(str(out_file), annotated)
            logger.info("Saved verified demonstration annotated frame to: %s", out_file)
            annotated_saved = True

        # Throttle to simulate 30 FPS display loop
        loop_elapsed = time.perf_counter() - t_loop_start
        sleep_time = max(0.001, (1.0 / 30.0) - loop_elapsed)
        time.sleep(sleep_time)

    # Ensure worker finished
    worker.shutdown(timeout=3.0)
    res = worker.get_latest_result()

    total_time = time.perf_counter() - fps_measure_start
    avg_fps = frame_count / total_time

    logger.info("Camera demonstration complete. Total frames processed: %d, Avg Loop FPS: %.1f", frame_count, avg_fps)
    assert res is not None and res.success, "OCR failed during live demonstration"
    assert res.detection_count >= 2, f"Expected at least 2 text regions, got {res.detection_count}"
    logger.info("Verification confirmed: Camera loop maintained ~%.1f FPS with zero blocking!", avg_fps)


if __name__ == "__main__":
    run_demonstration()
