"""Standalone Live Object Detection Runner for NAYRA.

Runs YOLOv10 deep object detection via ONNX Runtime (CUDA / CPU)
with live bounding boxes, confidence scores, latency metrics,
and real-time spoken audio announcements of detected objects.

Usage:
    python object_detection.py
    python scripts/test_object_detection.py
    python scripts/test_object_detection.py --source 0
    python scripts/test_object_detection.py --source http://192.168.1.X:8080/video
    python scripts/test_object_detection.py --no-speak
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Setup paths
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "03_object_detection"))
sys.path.insert(0, str(root_dir / "08_tts"))

import cv2

try:
    from dotenv import load_dotenv
    load_dotenv(root_dir / ".env")
except ImportError:
    pass

import importlib
obj_det = importlib.import_module("03_object_detection")
ObjectDetector = obj_det.ObjectDetector
from config.settings import ObjectDetectionSettings

try:
    from tts import TTSManager
except ImportError:
    TTSManager = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-Object-Detector")


def build_spoken_phrase(class_name: str, zone: str) -> str:
    """Format natural assistive spatial descriptions."""
    name = class_name.replace("_", " ").lower()
    if zone == "LEFT":
        return f"{name} on your left"
    elif zone == "RIGHT":
        return f"{name} on your right"
    elif zone == "CENTER":
        return f"{name} ahead"
    return f"{name} detected"


def main():
    parser = argparse.ArgumentParser(description="NAYRA Live YOLOv10 Object Detection with Voice")
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("ANDROID_STREAM_URL", "0"),
        help="Camera index (e.g. 0) or IP camera stream URL (default: from .env or 0)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence threshold (0.10 to 0.90, default: 0.30)"
    )
    parser.add_argument(
        "--assistive-only",
        action="store_true",
        help="Filter only assistive classes (chair, bottle, phone, person, etc.)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device ('cuda' or 'cpu', default: cuda)"
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Mute spoken voice announcements of detected objects"
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=3.5,
        help="Speech cooldown in seconds per object/zone (default: 3.5s)"
    )
    args = parser.parse_args()

    enable_speech = (not args.no_speak) and (TTSManager is not None)

    print("\n" + "=" * 65)
    print("  👁️ NAYRA — YOLOv10 LIVE OBJECT DETECTION")
    print("=" * 65)
    print(f"  • Source:     {args.source}")
    print(f"  • Threshold:  {args.conf * 100:.0f}%")
    print(f"  • Device:     {args.device}")
    print(f"  • Filter:     {'Assistive Only' if args.assistive_only else 'All 80 COCO Classes'}")
    print(f"  • Spoken Voice:{'Enabled (Kokoro-82M Studio Audio)' if enable_speech else 'Disabled'}")
    print("=" * 65)

    # 1. Initialize TTS Engine
    tts = None
    if enable_speech:
        try:
            print("\n[INFO] Initializing Kokoro Neural TTS for real-time speech...")
            tts = TTSManager(phone_speaker=False)
            tts.speak("Object detection active. Scanning your surroundings.")
        except Exception as e:
            logger.warning("TTS initialization failed: %s. Continuing in visual-only mode.", e)
            tts = None

    # 2. Initialize YOLOv10 Detector
    settings = ObjectDetectionSettings(
        filter_assistive_only=args.assistive_only,
        confidence_threshold=args.conf,
        device=args.device,
    )

    print("\n[INFO] Initializing YOLOv10 ONNX Engine...")
    detector = ObjectDetector(settings=settings)
    print(f"[INFO] Engine Loaded. Active Device: {detector._active_device}")

    # 3. Open Camera
    cap_source = int(args.source) if str(args.source).strip().isdigit() else args.source
    cap = cv2.VideoCapture(cap_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Fallback to webcam 0 if IP stream fails
    if not cap.isOpened() and cap_source != 0:
        print(f"[WARN] Could not connect to {args.source}. Falling back to default webcam (0)...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No camera stream could be opened! Please check your camera connection.")
        if tts:
            tts.shutdown()
        return

    print("\n" + "*" * 65)
    print("  🚀 DETECTION LIVE! Press 'q' in the window to quit.")
    print("*" * 65 + "\n")

    frame_count = 0
    last_announced: dict[str, float] = {}
    last_speech_time = 0.0
    global_speech_gap = 2.0  # At least 2.0s gap between any announcements

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            now = time.time()

            # Run detection
            detections, metrics = detector.detect(frame)

            # Voice Announcement of Detected Objects
            if tts and detections and (now - last_speech_time >= global_speech_gap):
                # Pick the highest confidence or largest object that hasn't been announced recently
                candidates = []
                for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
                    key = f"{det.class_name}_{det.spatial_zone}"
                    if now - last_announced.get(key, 0.0) >= args.cooldown:
                        candidates.append((key, det))

                if candidates:
                    chosen_key, chosen_det = candidates[0]
                    phrase = build_spoken_phrase(chosen_det.class_name, chosen_det.spatial_zone)
                    tts.speak(phrase)
                    last_announced[chosen_key] = now
                    last_speech_time = now
                    print(f"  🗣️ Spoke: \"{phrase}\"")

            # Draw bounding boxes and HUD
            annotated = detector.draw_detections(frame, detections, metrics=metrics, draw_hud=True)

            # Display
            cv2.imshow("NAYRA — YOLOv10 Object Detection", annotated)

            # Log periodically to console
            if frame_count % 30 == 0:
                detected_names = [f"{d.class_name} ({int(d.confidence*100)}%)" for d in detections]
                names_str = ", ".join(detected_names) if detected_names else "None"
                print(f"[FPS: {metrics.inference_fps:.1f} | Latency: {metrics.inference_latency_ms:.1f}ms] Detected: {names_str}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[INFO] Exiting detection.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if tts:
            tts.shutdown()


if __name__ == "__main__":
    main()
