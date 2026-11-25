"""NAYRA Module 07 — Independent Live Camera OCR Diagnostic Tool.

Directly captures a single frame from the Android IP Webcam stream,
executes Module 07's public OCR API (ocr.read_text), prints structured
detection/recognition telemetry, and displays the image for visual verification.

Usage:
    python test_live_ocr.py
    python test_live_ocr.py --url http://192.168.0.101:8080/video
    python test_live_ocr.py --no-gui
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple
import urllib.error
import urllib.request

import cv2
import numpy as np

# Set OpenCV / FFmpeg network timeout options (in microseconds)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "timeout;5000000|stimeout;5000000",
)

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import existing Module 07 public API (Zero internal modification)
try:
    from ocr import read_text, render_ocr_overlay, OCRResult
except ImportError as exc:
    print(f"[ERROR] Failed to import Module 07 OCR public API: {exc}")
    sys.exit(1)


DEFAULT_STREAM_URL = "http://192.168.0.101:8080/video"
DEFAULT_SAVE_PATH = "live_ocr_test.jpg"
DEFAULT_ANNOTATED_PATH = "live_ocr_test_annotated.jpg"


def check_stream_reachability(url: str, timeout_sec: float = 3.0) -> Tuple[bool, Optional[str]]:
    """Verify HTTP endpoint reachability before invoking OpenCV capture backend."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NAYRA-LiveOCR-Diagnostic/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = getattr(resp, "status", 200)
            if status in (200, 206):
                return True, None
            return False, f"HTTP Status {status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP Error {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return False, f"Connection Failed: {exc.reason}"
    except TimeoutError:
        return False, "Connection Timed Out"
    except Exception as exc:
        return False, f"Network Exception: {exc}"


def capture_single_frame(url: str, save_path: Path) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
    """Connect to IP Webcam video stream and capture exactly one clear frame."""
    print(f"\n[1/3] Connecting to Android camera stream: {url} ...")
    reachable, err_reason = check_stream_reachability(url, timeout_sec=3.5)
    if not reachable:
        return False, None, err_reason

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return False, None, "cv2.VideoCapture failed to open stream."

    try:
        # Discard the initial buffer frames (typically 3-5 frames)
        # to ensure the sensor auto-focus and auto-exposure have settled
        frame = None
        for _ in range(5):
            ret, tmp_frame = cap.read()
            if ret and tmp_frame is not None and tmp_frame.size > 0:
                frame = tmp_frame

        if frame is None:
            return False, None, "Failed to retrieve a valid video frame from stream."

        # Save exact raw captured frame
        cv2.imwrite(str(save_path), frame)
        return True, frame, None

    except Exception as exc:
        return False, None, f"Exception during frame acquisition: {exc}"
    finally:
        cap.release()


def run_diagnostic(
    stream_url: str,
    save_path: Path,
    annotated_path: Path,
    device: str,
    no_gui: bool,
    gui_timeout: int,
) -> int:
    """Execute live capture, OCR inference, metric printing, and image display."""
    print("=" * 76)
    print("        NAYRA Module 07 — Live Phone Camera OCR Diagnostic")
    print("=" * 76)
    print(f"Target Stream URL     : {stream_url}")
    print(f"Raw Frame Save Path   : {save_path.resolve()}")
    print(f"Annotated Save Path   : {annotated_path.resolve()}")
    print(f"Preferred OCR Device  : {device.upper()}")
    print("=" * 76)

    # 1. Capture single frame from phone camera
    capture_ok, frame, cap_err = capture_single_frame(stream_url, save_path)

    if not capture_ok or frame is None:
        print("\n" + "!" * 76)
        print(" [CAMERA ERROR] Frame capture failed.")
        print(f" Reason: {cap_err}")
        print("!" * 76)
        print("\nTroubleshooting Checklist:")
        print("  1. Verify the Android phone and PC are connected to the same Wi-Fi network.")
        print("  2. Open the 'IP Webcam' application on your Android phone.")
        print("  3. Scroll to the very bottom and tap 'Start server'.")
        print("  4. Verify the IP address shown at the bottom of the phone screen.")
        print("     If the IP is different from 192.168.0.101, pass the actual URL:")
        print("         python test_live_ocr.py --url http://<YOUR_PHONE_IP>:8080/video")
        print("=" * 76)
        return 1

    h, w, c = frame.shape
    file_size_kb = save_path.stat().st_size / 1024.0 if save_path.exists() else 0.0

    print(f"\n[CAPTURE SUCCESS]")
    print(f"  - Status            : Frame successfully captured and written to disk")
    print(f"  - Output File       : {save_path.name} ({file_size_kb:.1f} KB)")
    print(f"  - Frame Dimensions  : {w} x {h} (Width x Height, Channels: {c})")

    # 2. Run Module 07 OCR
    print(f"\n[2/3] Executing Module 07 Unified OCR (DBNet + PP-OCRv5)...")
    t0 = time.perf_counter()
    try:
        ocr_result: OCRResult = read_text(frame, preferred_provider=device.upper())
    except Exception as exc:
        print(f"\n[OCR ERROR] read_text() threw an unhandled exception: {exc}")
        import traceback
        traceback.print_exc()
        return 2

    ocr_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Print Structured OCR Metrics
    print("\n" + "=" * 76)
    print("                        OCR DIAGNOSTIC RESULTS")
    print("=" * 76)
    print(f"Frame Capture Succeeded  : YES")
    print(f"Frame Dimensions         : {w} x {h}")
    print(f"OCR Execution Success    : {ocr_result.success}")
    print(f"OCR Total Latency        : {ocr_elapsed_ms:.1f} ms (Engine Reported: {ocr_result.total_time_ms:.1f} ms)")
    print(f"  ├── Detection Time     : {ocr_result.detection_time_ms:.1f} ms")
    print(f"  ├── Script Routing     : {ocr_result.routing_time_ms:.1f} ms")
    print(f"  └── Text Recognition   : {ocr_result.recognition_time_ms:.1f} ms")
    print(f"Active Providers         : {ocr_result.execution_providers}")

    lang_val = getattr(ocr_result.primary_language, "value", str(ocr_result.primary_language))
    print(f"Primary Language         : {lang_val}")
    print(f"Regions Summary          : Detected={ocr_result.detection_count} | Recognized={ocr_result.recognition_count} | Unresolved={ocr_result.unresolved_count}")

    print("\n" + "-" * 76)
    print("OCRResult.combined_text:")
    print("-" * 76)
    if ocr_result.combined_text and ocr_result.combined_text.strip():
        for line in ocr_result.combined_text.strip().splitlines():
            print(f"  {line}")
    else:
        print("  (No text detected in captured camera frame)")
    print("-" * 76)

    # 4. Individual Region Breakdown
    print("\nIndividual Detected Text Regions:")
    if not ocr_result.regions:
        print("  (Zero text regions detected by DBNet)")
    else:
        print(f"  {'ID':<4} | {'Order':<5} | {'Conf':<6} | {'Script':<15} | {'Status':<18} | {'Recognized Text'}")
        print("  " + "-" * 72)
        for r in ocr_result.regions:
            status_str = getattr(r.status, "value", str(r.status))
            conf_str = f"{r.confidence:.2f}"
            clean_text = r.text.replace("\n", " ").strip() if r.text else "(empty)"
            print(f"  #{r.region_id:<3} | #{r.reading_order:<4} | {conf_str:<6} | {r.script:<15} | {status_str:<18} | \"{clean_text}\"")

    print("=" * 76)

    # 5. Visual Rendering and GUI Display
    print(f"\n[3/3] Preparing visual verification...")
    annotated_frame = frame.copy()
    try:
        annotated_frame = render_ocr_overlay(frame, ocr_result, is_busy=False)
        cv2.imwrite(str(annotated_path), annotated_frame)
        print(f"  - Annotated overlay saved: {annotated_path.name}")
    except Exception as ren_err:
        print(f"  - Note: Overlay rendering failed ({ren_err}); using raw frame for display.")

    if no_gui:
        print("  - GUI display skipped (--no-gui specified).")
    else:
        window_title = "NAYRA Module 07 — Live Phone Camera OCR (Press any key to close)"
        try:
            # Scale for convenient desktop display if resolution is very large
            display_h, display_w = annotated_frame.shape[:2]
            max_w, max_h = 1280, 720
            scale = min(max_w / display_w, max_h / display_h, 1.0)
            if scale < 1.0:
                display_img = cv2.resize(annotated_frame, (int(display_w * scale), int(display_h * scale)), interpolation=cv2.INTER_AREA)
            else:
                display_img = annotated_frame

            print(f"  - Displaying interactive window. Press ANY KEY in the window to close...")
            cv2.imshow(window_title, display_img)
            cv2.waitKey(gui_timeout)
            cv2.destroyAllWindows()
            print("  - Window closed.")
        except Exception as gui_err:
            print(f"  - GUI display unavailable ({gui_err}). Image saved to {save_path.name}.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="NAYRA Module 07 Live Camera OCR Diagnostic")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_STREAM_URL,
        help=f"Android IP Webcam video endpoint (default: {DEFAULT_STREAM_URL})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=DEFAULT_SAVE_PATH,
        help=f"File path to save the captured raw frame (default: {DEFAULT_SAVE_PATH})",
    )
    parser.add_argument(
        "--annotated-path",
        type=str,
        default=DEFAULT_ANNOTATED_PATH,
        help=f"File path to save the annotated OCR overlay (default: {DEFAULT_ANNOTATED_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CUDA",
        choices=["CUDA", "CPU"],
        help="Preferred execution provider (CUDA or CPU, default: CUDA)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable interactive OpenCV window display",
    )
    parser.add_argument(
        "--gui-timeout",
        type=int,
        default=0,
        help="Milliseconds to keep display window open (0 = wait for keypress, default: 0)",
    )

    args = parser.parse_args()
    ret_code = run_diagnostic(
        stream_url=args.url,
        save_path=Path(args.save_path),
        annotated_path=Path(args.annotated_path),
        device=args.device,
        no_gui=args.no_gui,
        gui_timeout=args.gui_timeout,
    )
    sys.exit(ret_code)


if __name__ == "__main__":
    main()
