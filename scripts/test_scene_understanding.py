"""Interactive test tool for Module 12 (Scene Understanding).

Performs real-time camera scene analysis, spatial entity summarization,
on-screen subtitle HUD rendering, and natural English spoken scene narration.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "12_scene_understanding"))
sys.path.insert(0, str(root_dir / "03_object_detection"))
sys.path.insert(0, str(root_dir / "08_tts"))

import cv2
import numpy as np

import importlib
det_mod = importlib.import_module("03_object_detection.detector")
ObjectDetector = det_mod.ObjectDetector
from config.settings import ObjectDetectionSettings

from scene_understanding import SceneService, SceneAnalyzer
from tts import TTSManager
from shared.schemas import IntentRequest, UserIntent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress phonemizer warnings
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logger = logging.getLogger("NAYRA-Scene-Tester")


def test_offline_scene():
    """Verify scene analysis logic with mock scene entities."""
    print("\n" + "=" * 65)
    print("  TEST: OFFLINE SCENE UNDERSTANDING VERIFICATION")
    print("=" * 65)

    analyzer = SceneAnalyzer()
    service = SceneService(analyzer=analyzer, preferred_language="en")
    service.initialize()

    # Create mock detections: person in center, laptop on desk, bottle on right
    mock_person = type("Detection", (), {
        "class_name": "person", "confidence": 0.95, "box_xyxy": (180, 100, 360, 440)
    })()
    mock_laptop = type("Detection", (), {
        "class_name": "laptop", "confidence": 0.91, "box_xyxy": (220, 280, 420, 460)
    })()
    mock_bottle = type("Detection", (), {
        "class_name": "bottle", "confidence": 0.86, "box_xyxy": (480, 240, 560, 420)
    })()

    service.update_scene_context([mock_person, mock_laptop, mock_bottle], frame_width=640, frame_height=480)
    req = IntentRequest(raw_query="What is in front of me?", intent=UserIntent.DESCRIBE_SCENE)
    resp = service.handle_intent(req)

    print(f"[Query]   : {req.raw_query}")
    print(f"[English] : {resp.spoken_response_en}")
    print("\n" + "=" * 65)
    print("Tip: To run the live visual scene describer, run:")
    print("python scripts/test_scene_understanding.py --camera 0")
    print("=" * 65 + "\n")


def test_live_scene(camera_source: str | int, tts: TTSManager):
    """Run live camera feed with real-time scene understanding & spoken narration."""
    source_label = "Laptop Built-in Webcam" if str(camera_source).strip() == "0" else str(camera_source)
    print("\n" + "=" * 70)
    print(f"  NAYRA SCENE UNDERSTANDING -> {source_label}")
    print("=" * 70)

    # 1. Initialize detector
    print("Loading YOLOv10 object detector on CUDA...")
    det_settings = ObjectDetectionSettings(
        filter_assistive_only=False,
        confidence_threshold=0.25,
        device="cuda",
    )
    detector = ObjectDetector(settings=det_settings)

    # 2. Initialize Scene Service
    analyzer = SceneAnalyzer()
    service = SceneService(analyzer=analyzer, preferred_language="en")
    service.initialize()

    # Pre-warm speech
    tts.speak("Scene understanding active. Press spacebar to describe the scene.")

    cap_source = int(camera_source) if str(camera_source).strip().isdigit() else camera_source
    cap = cv2.VideoCapture(cap_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not connect to camera at {source_label}")
        return

    print("\n" + "*" * 70)
    print("  READY FOR VIDEO DEMONSTRATION!")
    print("  Controls:")
    print("    [SPACE] -> Describe scene: 'What is in front of me?'")
    print("    [q]     -> Quit")
    print("*" * 70 + "\n")

    last_speech_time = 0.0
    last_speech_text = ""
    latest_detections = []
    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_counter += 1
            h, w = frame.shape[:2]
            vis = frame.copy()

            # Run detection every 2 frames
            if frame_counter % 2 == 0:
                raw_dets, _ = detector.detect(frame)
                latest_detections = raw_dets if raw_dets is not None else []
                service.update_scene_context(latest_detections, frame_width=w, frame_height=h)

            # Draw bounding boxes for all detected objects
            for det in latest_detections:
                x1, y1, x2, y2 = det.box_xyxy
                box_color = (0, 230, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                lbl = f"{det.class_name.upper()} {int(det.confidence * 100)}%"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + lw + 6, y1), box_color, -1)
                cv2.putText(vis, lbl, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

            # Top Instructions HUD
            cv2.rectangle(vis, (0, 0), (w, 45), (20, 20, 20), -1)
            cv2.line(vis, (0, 45), (w, 45), (0, 255, 255), 2)
            cv2.putText(vis, "SCENE UNDERSTANDING | PRESS [SPACEBAR] TO DESCRIBE SCENE | [Q] TO QUIT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Bottom Subtitle Banner
            if last_speech_text and (time.time() - last_speech_time < 9.0):
                cv2.rectangle(vis, (0, h - 65), (w, h), (10, 10, 10), -1)
                cv2.line(vis, (0, h - 65), (w, h - 65), (0, 255, 0), 2)
                cv2.putText(vis, f"NAYRA: \"{last_speech_text[:75]}\"", (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("NAYRA Scene Understanding", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == 32:  # SPACEBAR -> ask scene description
                print("\n[USER QUERY]: 'What is in front of me?'")
                req = IntentRequest(raw_query="What is in front of me?", intent=UserIntent.DESCRIBE_SCENE)
                resp = service.handle_intent(req)
                clean_text = resp.spoken_response_en.strip()

                print(f"[NAYRA]: \"{clean_text}\"")
                tts.speak(clean_text)
                last_speech_text = clean_text
                last_speech_time = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        service.shutdown()


def main():
    parser = argparse.ArgumentParser(description="NAYRA Module 12 Scene Understanding Tester")
    parser.add_argument("--camera", type=str, default="", help="Camera index ('0') or IP Webcam URL")
    args, _ = parser.parse_known_args()

    tts = TTSManager(phone_speaker=False)

    cam_url = args.camera.strip().strip('"').strip("'")
    if cam_url.startswith("http://http://"):
        cam_url = cam_url.replace("http://http://", "http://")
    elif cam_url and not cam_url.startswith("http://") and not cam_url.startswith("https://") and not cam_url.isdigit():
        cam_url = "http://" + cam_url

    if cam_url.startswith("http") and ":8080" in cam_url and not cam_url.endswith("/video"):
        cam_url = cam_url.rstrip("/") + "/video"

    try:
        if cam_url != "":
            test_live_scene(cam_url, tts)
        else:
            test_offline_scene()
    finally:
        tts.shutdown()


if __name__ == "__main__":
    main()
