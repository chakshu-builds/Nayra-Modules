"""Interactive test tool for Module 10 (Object Location Memory).

Real-time spatial memory tracking, natural language querying, on-screen HUD subtitles,
and Kokoro neural speech output for video demonstration.
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
sys.path.insert(0, str(root_dir / "10_object_memory"))
sys.path.insert(0, str(root_dir / "08_tts"))

import cv2
import numpy as np

from object_memory import (
    LiveCameraStream,
    ObjectMemoryDB,
    ObjectMemoryManager,
    ANCHOR_CLASSES,
)
from tts import TTSManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress phonemizer warnings
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logger = logging.getLogger("NAYRA-ObjectMemory-Tester")


def test_offline_queries(manager: ObjectMemoryManager):
    """Run verification queries against the SQLite memory database."""
    print("\n" + "=" * 65)
    print("  TEST 1: OFFLINE OBJECT LOCATION MEMORY VERIFICATION")
    print("=" * 65)

    test_queries = [
        "Where is my bottle?",
        "Where did I leave my phone?",
        "Where is my laptop?",
        "Where is my book?",
    ]

    for q in test_queries:
        res = manager.query(q, preferred_language="en", speak=False)
        print(f"\n[Query]   : {q}")
        print(f"[Target]  : {res.canonical_name} (Currently Visible: {res.is_currently_visible}, Found: {res.found})")
        print(f"[English] : {res.spoken_response_en}")

    print("\n" + "=" * 65)
    print("Tip: To run live visual tracking and voice queries, run:")
    print("python scripts/test_object_memory.py --camera 0")
    print("python scripts/test_object_memory.py --camera http://192.168.1.120:8080/video")
    print("=" * 65 + "\n")


def test_live_camera_memory(camera_source: str | int, manager: ObjectMemoryManager, tts: TTSManager):
    """Run live camera object tracking, spatial persistence, and 1-key demo queries."""
    source_label = "Laptop Built-in Webcam" if str(camera_source).strip() == "0" else str(camera_source)
    print("\n" + "=" * 70)
    print(f"  NAYRA OBJECT LOCATION MEMORY -> {source_label}")
    print("=" * 70)

    # Pre-warm speech
    tts.speak("Object memory initialized. Point camera at objects and use hotkeys to query.")

    cap_source = int(camera_source) if str(camera_source).strip().isdigit() else camera_source
    cap = cv2.VideoCapture(cap_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at {source_label}")
        return

    print("\n" + "*" * 70)
    print("  CAMERA CONNECTED & READY FOR VIDEO DEMO!")
    print("  Controls (Press key in camera window to test):")
    print("    [b] -> Ask: 'Where is my bottle?'")
    print("    [p] -> Ask: 'Where is my phone?'")
    print("    [c] -> Ask: 'Where is my cup?'")
    print("    [l] -> Ask: 'Where is my laptop?'")
    print("    [k] -> Ask: 'Where is my keyboard / keys?'")
    print("    [SPACE] -> Type any custom question in terminal")
    print("    [i] -> Print memory inventory")
    print("    [q] -> Quit")
    print("*" * 70 + "\n")

    current_query = ""
    current_answer = ""
    highlight_target = None
    query_timestamp = 0.0
    frame_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_counter += 1
            now = time.time()
            h, w = frame.shape[:2]

            # Detect & track objects every 2 frames for smooth performance
            if frame_counter % 2 == 0:
                manager.observe_frame(frame, timestamp=now)

            vis = frame.copy()

            # 1. Draw detected bounding boxes
            if hasattr(manager, "_last_frame_detections") and manager._last_frame_detections:
                for d in manager._last_frame_detections:
                    x1, y1, x2, y2 = d.box_xyxy
                    cname = d.class_name.lower()
                    is_queried = (highlight_target and cname == highlight_target.lower())

                    box_color = (0, 255, 0) if is_queried else (0, 220, 255)
                    box_thick = 4 if is_queried else 2

                    cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, box_thick)
                    label = f"{d.class_name.upper()} {int(d.confidence * 100)}%"
                    if is_queried:
                        label = f">> {label} [QUERIED] <<"

                    # Label background
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(vis, (x1, max(0, y1 - 24)), (x1 + tw + 8, y1), box_color, -1)
                    cv2.putText(vis, label, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # 2. Draw Top Control HUD
            cv2.rectangle(vis, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.line(vis, (0, 50), (w, 50), (0, 200, 255), 2)
            controls_text = "HOTKEYS: [B] Bottle | [P] Phone | [C] Cup | [L] Laptop | [K] Keys | [SPACE] Query | [Q] Quit"
            cv2.putText(vis, controls_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)

            # 3. Draw Bottom Subtitle Banner (if a query was answered recently)
            if current_answer and (now - query_timestamp < 9.0):
                # Banner height
                cv2.rectangle(vis, (0, h - 75), (w, h), (10, 10, 10), -1)
                cv2.line(vis, (0, h - 75), (w, h - 75), (0, 255, 0), 2)

                cv2.putText(vis, f"YOU: \"{current_query}\"", (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(vis, f"NAYRA: \"{current_answer[:75]}\"", (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("NAYRA Object Location Memory Demo", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break

            target_to_ask = None
            if key in (ord('b'), ord('B')):
                target_to_ask = "Where is my bottle?"
            elif key in (ord('p'), ord('P')):
                target_to_ask = "Where is my phone?"
            elif key in (ord('c'), ord('C')):
                target_to_ask = "Where is my cup?"
            elif key in (ord('l'), ord('L')):
                target_to_ask = "Where is my laptop?"
            elif key in (ord('k'), ord('K')):
                target_to_ask = "Where is my keyboard?"
            elif key in (ord('i'), ord('I')):
                objects = manager.list_known_objects()
                print("\n[ACTIVE MEMORY INVENTORY]:")
                for obj in objects:
                    print(f"  - {obj.get('canonical_name')}: {obj.get('spatial_relation')} {obj.get('anchor_object')} (Last seen: {obj.get('last_seen_relative')})")
            elif key == 32:  # SPACEBAR -> terminal custom query
                print("\n" + "-" * 50)
                try:
                    target_to_ask = input("Ask NAYRA a question (e.g. 'Where is my bottle?'): ").strip()
                except (EOFError, KeyboardInterrupt):
                    target_to_ask = None

            if target_to_ask:
                current_query = target_to_ask
                query_timestamp = time.time()
                print(f"\n[QUERY]: '{current_query}'")

                res = manager.query(current_query, current_frame=frame, preferred_language="en", speak=False)
                clean_answer = res.spoken_response_en.strip()
                current_answer = clean_answer
                highlight_target = res.canonical_name

                print(f"[NAYRA]: \"{clean_answer}\"")
                # Speak aloud through persistent Kokoro TTS in English
                tts.speak(clean_answer)

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="NAYRA Object Location Memory Tester")
    parser.add_argument("--camera", type=str, default="", help="Camera index ('0') or IP Webcam URL")
    parser.add_argument("--query", type=str, default="", help="Execute a single query directly")
    args, _ = parser.parse_known_args()

    # Initialize persistent components
    tts = TTSManager(phone_speaker=False)

    import importlib
    det_mod = importlib.import_module("03_object_detection.detector")
    ObjectDetector = det_mod.ObjectDetector
    from config.settings import ObjectDetectionSettings
    det_settings = ObjectDetectionSettings(
        filter_assistive_only=False,
        confidence_threshold=0.25,
        device="cuda",
    )
    detector_yolo = ObjectDetector(settings=det_settings)

    manager = ObjectMemoryManager(
        detector=detector_yolo,
        tts_manager=tts,
        async_db_writes=True,
        enable_motion_gating=False,
        device="cuda",
    )

    cam_url = args.camera.strip().strip('"').strip("'")
    if cam_url.startswith("http://http://"):
        cam_url = cam_url.replace("http://http://", "http://")
    elif cam_url and not cam_url.startswith("http://") and not cam_url.startswith("https://") and not cam_url.isdigit():
        cam_url = "http://" + cam_url

    if cam_url.startswith("http") and ":8080" in cam_url and not cam_url.endswith("/video"):
        cam_url = cam_url.rstrip("/") + "/video"

    try:
        if args.query:
            res = manager.query(args.query, preferred_language="en", speak=False)
            print(f"\n[Query]   : {args.query}")
            print(f"[NAYRA]   : {res.spoken_response_en}")
            tts.speak(res.spoken_response_en)
            time.sleep(3.5)
        elif cam_url != "":
            test_live_camera_memory(cam_url, manager, tts)
        else:
            test_offline_queries(manager)
    finally:
        manager.shutdown()
        tts.shutdown()


if __name__ == "__main__":
    main()
