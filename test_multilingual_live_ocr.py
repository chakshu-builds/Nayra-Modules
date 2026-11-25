"""NAYRA Module 07 — Multilingual Live Camera OCR Verification Tool.

Captures sequential real-world frames from the Android IP Webcam for:
1. English printed text
2. Hindi (Devanagari) printed text
3. Hinglish (Code-Mixed) printed text

Provides a 5-second countdown between captures to allow repositioning of printed
materials, runs Module 07 public API (from ocr import read_text), saves raw and
annotated images to data/ocr_diagnostics/, and prints a final comparison report.

STRICT CONSTRAINTS:
- Zero modifications to Modules 01-07.
- No TTS, voice commands, or Read Aloud pipeline dependencies.
- No static sample images.
- No accuracy calculation without verified ground truth.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
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
DEFAULT_OUTPUT_DIR = "data/ocr_diagnostics"

STAGES = [
    {
        "key": "english",
        "title": "English Printed Text",
        "desc": "Place an English document, sign, packaging, or book page in view.",
        "filename": "english_live.jpg",
        "annotated": "english_live_annotated.jpg",
    },
    {
        "key": "hindi",
        "title": "Hindi (Devanagari) Printed Text",
        "desc": "Place a Devanagari Hindi newspaper, receipt, or printed notice in view.",
        "filename": "hindi_live.jpg",
        "annotated": "hindi_live_annotated.jpg",
    },
    {
        "key": "hinglish",
        "title": "Hinglish (Code-Mixed) Printed Text",
        "desc": "Place bilingual or code-mixed Hindi+English printed text in view.",
        "filename": "hinglish_live.jpg",
        "annotated": "hinglish_live_annotated.jpg",
    },
]


def check_stream_reachability(url: str, timeout_sec: float = 3.5) -> Tuple[bool, Optional[str]]:
    """Verify HTTP endpoint reachability before invoking OpenCV capture backend."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NAYRA-Multilingual-Diagnostic/1.0"},
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
    """Connect to IP Webcam, flush buffer frames, capture exactly one frame, and release connection."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return False, None, "cv2.VideoCapture failed to open video stream."

    try:
        frame = None
        # Discard first 5 buffer frames to let sensor exposure/focus stabilize
        for _ in range(5):
            ret, tmp_frame = cap.read()
            if ret and tmp_frame is not None and tmp_frame.size > 0:
                frame = tmp_frame

        if frame is None:
            return False, None, "Failed to retrieve a valid video frame from stream."

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), frame)
        return True, frame, None

    except Exception as exc:
        return False, None, f"Exception during frame capture: {exc}"
    finally:
        # Crucial: Immediately release camera resource so camera is not left running
        cap.release()


def run_countdown(seconds: int = 5) -> None:
    """Print an active countdown in the console."""
    print("  Countdown to capture:")
    for remaining in range(seconds, 0, -1):
        print(f"    >>> [{remaining}s] Hold steady...", flush=True)
        time.sleep(1.0)
    print("    >>> [CAPTURE NOW!]", flush=True)


def execute_multilingual_verification(
    stream_url: str,
    output_dir: Path,
    device: str,
    countdown_sec: int,
    auto_proceed: bool,
) -> int:
    """Execute the full 3-stage multilingual live OCR verification."""
    print("=" * 78)
    print("    NAYRA Module 07 — Multilingual Live Camera OCR Verification")
    print("=" * 78)
    print(f"Target Stream URL     : {stream_url}")
    print(f"Diagnostics Directory : {output_dir.resolve()}")
    print(f"Preferred OCR Device  : {device.upper()}")
    print(f"Reposition Countdown  : {countdown_sec} seconds")
    print("=" * 78)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial reachability check
    print("\n[Step 0] Probing Android IP Webcam connection...")
    reachable, err_reason = check_stream_reachability(stream_url, timeout_sec=4.0)
    if not reachable:
        print("\n" + "!" * 78)
        print(" [CAMERA ERROR] Unable to connect to Android IP Webcam stream.")
        print(f" Endpoint: {stream_url}")
        print(f" Cause   : {err_reason}")
        print("!" * 78)
        print("\nTroubleshooting Checklist:")
        print("  1. Verify the phone and PC are on the exact same local Wi-Fi network.")
        print("  2. Open the 'IP Webcam' application on your Android phone.")
        print("  3. Scroll to the bottom and tap 'Start server'.")
        print("  4. Check the streaming IP address displayed on the phone.")
        print("     If different from 192.168.0.101, pass:")
        print("         python test_multilingual_live_ocr.py --url http://<PHONE_IP>:8080/video")
        print("=" * 78)
        return 1

    print("  >>> IP Webcam video stream is online and reachable.")

    reports: List[Dict[str, Any]] = []

    for idx, stage in enumerate(STAGES, 1):
        raw_path = output_dir / stage["filename"]
        annotated_path = output_dir / stage["annotated"]

        print("\n" + "=" * 78)
        print(f" STAGE {idx}/3: [{stage['title'].upper()}]")
        print("=" * 78)
        print(f"Target Material : {stage['desc']}")
        print(f"Raw Output File : {raw_path.name}")

        if not auto_proceed:
            try:
                input(f"\nPosition your {stage['title']} in front of camera, then press [ENTER] to begin countdown...")
            except (KeyboardInterrupt, EOFError):
                print("\nVerification aborted by user.")
                return 130

        # Countdown
        run_countdown(countdown_sec)

        # Capture single frame
        capture_ok, frame, cap_err = capture_single_frame(stream_url, raw_path)
        if not capture_ok or frame is None:
            print(f" [ERROR] Frame capture failed for {stage['title']}: {cap_err}")
            reports.append({
                "stage": stage["title"],
                "key": stage["key"],
                "capture_ok": False,
                "error": cap_err,
            })
            continue

        h, w, c = frame.shape
        raw_size_kb = raw_path.stat().st_size / 1024.0 if raw_path.exists() else 0.0

        print(f"  >>> Frame captured: {w}x{h} ({raw_size_kb:.1f} KB) -> Saved as {raw_path.name}")

        # Execute OCR using public API
        print(f"  >>> Running Module 07 OCR inference ({device.upper()})...")
        t0 = time.perf_counter()
        try:
            ocr_res: OCRResult = read_text(frame, preferred_provider=device.upper())
            ocr_latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as ocr_err:
            print(f"  [ERROR] OCR inference failed: {ocr_err}")
            reports.append({
                "stage": stage["title"],
                "key": stage["key"],
                "capture_ok": True,
                "dimensions": f"{w}x{h}",
                "ocr_ok": False,
                "error": str(ocr_err),
            })
            continue

        # Render and save visual overlay
        try:
            annotated = render_ocr_overlay(frame, ocr_res, is_busy=False)
            cv2.imwrite(str(annotated_path), annotated)
            print(f"  >>> Annotated overlay rendered -> Saved as {annotated_path.name}")
        except Exception as ren_err:
            print(f"  [WARNING] Failed to render overlay: {ren_err}")

        # Print stage metrics
        lang_str = getattr(ocr_res.primary_language, "value", str(ocr_res.primary_language))
        print("\n  --- [OCR RESULTS FOR STAGE] ---")
        print(f"  Execution Latency    : {ocr_latency_ms:.1f} ms (Pipeline: {ocr_res.total_time_ms:.1f} ms)")
        print(f"  Detected Language    : {lang_str}")
        print(f"  Detected Regions     : {ocr_res.detection_count} (Recognized: {ocr_res.recognition_count}, Unresolved: {ocr_res.unresolved_count})")
        print("  Combined Text:")
        if ocr_res.combined_text and ocr_res.combined_text.strip():
            for line in ocr_res.combined_text.strip().splitlines():
                print(f"    | {line}")
        else:
            print("    | (No text recognized)")

        print("  Detected Regions Detail:")
        if not ocr_res.regions:
            print("    (No regions detected)")
        else:
            for r in ocr_res.regions:
                status_val = getattr(r.status, "value", str(r.status))
                conf_val = f"{r.confidence:.2f}"
                text_val = r.text.replace("\n", " ").strip() if r.text else "(empty)"
                print(f"    #{r.region_id:02d} [Order {r.reading_order:02d}] Conf: {conf_val} | Script: {r.script:<14s} | Status: {status_val:<18s} | Text: \"{text_val}\"")

        # Average region confidence
        valid_confs = [r.confidence for r in ocr_res.regions if r.confidence > 0]
        avg_conf = (sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0

        reports.append({
            "stage": stage["title"],
            "key": stage["key"],
            "raw_filename": stage["filename"],
            "annotated_filename": stage["annotated"],
            "capture_ok": True,
            "dimensions": f"{w}x{h}",
            "file_size_kb": raw_size_kb,
            "ocr_ok": ocr_res.success,
            "latency_ms": ocr_latency_ms,
            "primary_lang": lang_str,
            "detection_count": ocr_res.detection_count,
            "recognition_count": ocr_res.recognition_count,
            "unresolved_count": ocr_res.unresolved_count,
            "avg_confidence": avg_conf,
            "combined_text": ocr_res.combined_text.strip() if ocr_res.combined_text else "",
            "region_count": len(ocr_res.regions),
        })

    # Final Comparison Report
    print("\n" + "=" * 78)
    print("                    FINAL MULTILINGUAL COMPARISON REPORT")
    print("=" * 78)
    print(f"{'Target Language':<22} | {'Dims':<9} | {'Latency':<9} | {'Router Lang':<12} | {'Regions':<8} | {'Avg Conf'}")
    print("-" * 78)
    for rep in reports:
        if not rep.get("capture_ok", False):
            print(f"{rep['stage']:<22} | FAILED CAPTURE ({rep.get('error', 'Unknown')})")
            continue
        if not rep.get("ocr_ok", False):
            print(f"{rep['stage']:<22} | {rep.get('dimensions', 'N/A'):<9} | FAILED OCR ({rep.get('error', 'Unknown')})")
            continue

        conf_str = f"{rep['avg_confidence'] * 100:.1f}%" if rep['avg_confidence'] > 0 else "N/A"
        print(f"{rep['stage']:<22} | {rep['dimensions']:<9} | {rep['latency_ms']:>6.1f} ms | {rep['primary_lang']:<12} | {rep['recognition_count']:>2}/{rep['detection_count']:<2}    | {conf_str}")

    print("-" * 78)
    print("\nText Extraction Summary:")
    for rep in reports:
        if rep.get("ocr_ok", False):
            print(f"\n[{rep['stage']}]:")
            text = rep.get("combined_text", "")
            if text:
                for line in text.splitlines():
                    print(f"  \"{line}\"")
            else:
                print("  (No text extracted)")

    print("\n" + "=" * 78)
    print("NOTE: Per strict audit protocol, CER/WER accuracy metrics are omitted")
    print("because ground-truth text lines were not pre-annotated for live camera scenes.")
    print("=" * 78)
    print("Diagnostic session concluded. Camera connection released.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="NAYRA Module 07 Multilingual Live Camera OCR Verification")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_STREAM_URL,
        help=f"Android IP Webcam video endpoint (default: {DEFAULT_STREAM_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save captured and annotated images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CUDA",
        choices=["CUDA", "CPU"],
        help="Preferred execution provider (default: CUDA)",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="Countdown seconds before each capture to allow text positioning (default: 5)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically start countdown without waiting for user [ENTER] keypress",
    )

    args = parser.parse_args()
    ret = execute_multilingual_verification(
        stream_url=args.url,
        output_dir=Path(args.output_dir),
        device=args.device,
        countdown_sec=args.countdown,
        auto_proceed=args.auto,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
