"""Interactive test tool for Module 07 (OCR) — Real-time camera & text recognition."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "07_ocr"))
sys.path.insert(0, str(root_dir / "08_tts"))

import cv2
import numpy as np

# Import OCR Engine & Result Schemas
from ocr import get_ocr_engine, read_text, render_ocr_overlay
from tts import TTSManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-OCR-Tester")


def test_ocr_synthetic():
    """Verify OCR models with a synthetic high-contrast multi-line test image."""
    print("\n" + "=" * 60)
    print("  TEST 1: SYNTHETIC TEST CARD OCR (DBNet + CRNN CUDA)")
    print("=" * 60)

    img = np.full((400, 800, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (40, 30), (760, 110), (30, 30, 30), -1)
    cv2.putText(img, "EMERGENCY EXIT ONLY", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.putText(img, "ROOM 402 - MEDICAL WARD", (60, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    cv2.putText(img, "AUTHORIZED PERSONNEL ONLY", (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(img, "KEEP DOOR CLOSED", (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 200), 3)

    print("Running DBNet detection & CRNN recognition...")
    t0 = time.perf_counter()
    result = read_text(img)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[OCR Engine Results] (Latency: {latency_ms:.1f}ms):")
    print(f"  Success       : {result.success}")
    print(f"  Regions Found : {result.detection_count}")
    print(f"  Recognized    : {result.recognition_count}")
    print(f"  Combined Text : \"{result.combined_text}\"")

    for idx, reg in enumerate(result.regions, 1):
        script_name = getattr(reg.script, "name", str(reg.script))
        print(f"    Region #{idx}: '{reg.text}' (Conf: {reg.confidence:.2f}, Script: {script_name})")

    vis = render_ocr_overlay(img, result)
    try:
        cv2.imshow("NAYRA OCR Test 1 — Synthetic Card", vis)
        print("\nDisplaying OCR visualization window (closes in 3s or press any key)...")
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Window display skipped: {e}")

    return result.combined_text


def run_robust_ocr(frame: np.ndarray):
    """Run OCR with adaptive sensitivity for real camera streams."""
    # Attempt 1: Standard read with relaxed box threshold for camera distances
    result = read_text(frame, det_thresh=0.25, det_box_thresh=0.45, det_unclip_ratio=1.6)
    if result.recognition_count > 0 and result.combined_text.strip():
        return result

    # Attempt 2: If no text, enhance contrast using CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    
    result2 = read_text(enhanced_bgr, det_thresh=0.20, det_box_thresh=0.35, det_unclip_ratio=1.8)
    if result2.recognition_count > 0 and result2.combined_text.strip():
        return result2

    return result


def test_ocr_camera(camera_url: str):
    """Run interactive OCR camera feed with persistent TTS and on-screen HUD."""
    source_label = "Laptop Built-in Webcam" if str(camera_url).strip() == "0" else camera_url
    print("\n" + "=" * 60)
    print(f"  NAYRA CAMERA OCR SCANNER -> {source_label}")
    print("=" * 60)

    # 1. Pre-warm persistent TTS Manager
    print("Initializing NAYRA TTS Manager (Kokoro Neural Speech)...")
    tts = TTSManager()
    tts.speak("NAYRA OCR ready. Point camera at text and press spacebar.")

    # 2. Pre-warm OCR engine
    print("Pre-warming DBNet & CRNN neural models...")
    _ = read_text(np.zeros((100, 200, 3), dtype=np.uint8))

    # 3. Connect to camera
    print(f"Connecting to video stream: {source_label} ...")
    cap_source = int(camera_url) if str(camera_url).strip().isdigit() else camera_url
    cap = cv2.VideoCapture(cap_source)
    
    # Set camera resolution for crisp text if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not connect to camera at {source_label}.")
        print("Please check your camera index or IP Webcam stream.")
        tts.shutdown()
        return

    print("\n" + "*" * 60)
    print("  CAMERA CONNECTED & READY FOR VIDEO DEMO!")
    print("  -> Point camera at any book, paper, label, or screen.")
    print("  -> Press [SPACEBAR] to read text aloud.")
    print("  -> Press [Q] to quit.")
    print("*" * 60 + "\n")

    last_detected_text = ""
    status_msg = "POINT AT TEXT -> PRESS [SPACEBAR] TO READ"
    status_color = (0, 255, 255)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            hud = frame.copy()

            # Top instructions HUD
            cv2.rectangle(hud, (0, 0), (w, 55), (20, 20, 20), -1)
            cv2.line(hud, (0, 55), (w, 55), status_color, 2)
            cv2.putText(hud, status_msg, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

            # Bottom Subtitle HUD (Shows last read text on screen)
            if last_detected_text:
                cv2.rectangle(hud, (0, h - 65), (w, h), (10, 10, 10), -1)
                cv2.line(hud, (0, h - 65), (w, h - 65), (0, 255, 0), 2)
                display_sub = f"NAYRA SPOKE: \"{last_detected_text[:65]}\""
                cv2.putText(hud, display_sub, (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("NAYRA Live OCR Scanner (Press SPACE to Read | Q to Quit)", hud)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == 32:  # SPACEBAR pressed -> execute OCR
                print("\n[CAPTURE] Processing frame for OCR...")
                status_msg = "READING TEXT... PLEASE HOLD STEADY"
                status_color = (0, 165, 255)

                # Show 'READING TEXT' flash
                cv2.rectangle(hud, (0, 0), (w, 55), (0, 140, 255), -1)
                cv2.putText(hud, "PROCESSING OCR... SCANNING TEXT", (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.imshow("NAYRA Live OCR Scanner (Press SPACE to Read | Q to Quit)", hud)
                cv2.waitKey(20)

                t0 = time.perf_counter()
                result = run_robust_ocr(frame)
                lat = (time.perf_counter() - t0) * 1000

                raw_text = result.combined_text.strip()
                # Clean up newlines for speech
                clean_speech = " ".join(raw_text.split())

                print(f"[OCR RESULT] ({lat:.1f}ms): {result.detection_count} regions found")
                if clean_speech:
                    print(f"  Detected Text: \"{clean_speech}\"")
                    last_detected_text = clean_speech
                    status_msg = f"READ: \"{clean_speech[:35]}...\""
                    status_color = (0, 255, 0)

                    # Speak via persistent TTS
                    speech_phrase = f"Text reads: {clean_speech}."
                    print(f"  NAYRA Speaking: '{speech_phrase}'")
                    tts.speak(speech_phrase)
                else:
                    print("  No clear text detected in frame.")
                    last_detected_text = "No clear text detected"
                    status_msg = "NO TEXT FOUND - HOLD CLOSER & PRESS SPACE"
                    status_color = (0, 0, 255)
                    tts.speak("No readable text detected in front of you.")

                # Render overlay with bounding boxes
                vis = render_ocr_overlay(frame, result)
                # Show results window for 3.5 seconds while audio plays
                cv2.rectangle(vis, (0, 0), (w, 55), (10, 10, 10), -1)
                banner = f"DETECTED: {clean_speech[:50]}" if clean_speech else "NO TEXT FOUND"
                bcolor = (0, 255, 0) if clean_speech else (0, 0, 255)
                cv2.putText(vis, banner, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bcolor, 2)
                cv2.imshow("NAYRA Live OCR Scanner (Press SPACE to Read | Q to Quit)", vis)

                # Wait with event pump so window doesn't freeze and audio plays
                pause_end = time.time() + 3.5
                while time.time() < pause_end:
                    k = cv2.waitKey(50) & 0xFF
                    if k in (ord('q'), ord('Q'), 27):
                        break

                status_msg = "POINT AT TEXT -> PRESS [SPACEBAR] TO READ"
                status_color = (0, 255, 255)

    finally:
        print("\nShutting down OCR and Audio services...")
        cap.release()
        cv2.destroyAllWindows()
        tts.shutdown()
        print("Done. Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAYRA OCR Feature Tester")
    parser.add_argument("--camera", type=str, default="", help="Phone camera URL or '0' for laptop webcam")
    parser.add_argument("--lang", type=str, default="en", help="Language for TTS output (en or hi)")
    args, unknown = parser.parse_known_args()

    cam_url = args.camera.strip().strip('"').strip("'")
    if cam_url.startswith("http://http://"):
        cam_url = cam_url.replace("http://http://", "http://")
    elif cam_url.startswith("https://https://"):
        cam_url = cam_url.replace("https://https://", "https://")
    elif cam_url and not cam_url.startswith("http://") and not cam_url.startswith("https://") and not cam_url.isdigit():
        cam_url = "http://" + cam_url

    if cam_url.startswith("http") and ":8080" in cam_url and not cam_url.endswith("/video"):
        cam_url = cam_url.rstrip("/") + "/video"

    if cam_url != "":
        test_ocr_camera(cam_url)
    else:
        test_ocr_synthetic()
