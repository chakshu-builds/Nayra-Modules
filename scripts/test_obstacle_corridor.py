"""Interactive test tool for Module 11 (Obstacle Awareness & Navigation Corridor).

Provides real-time walking path guidance (Green = Clear, Red = Warning), proximity estimation,
spoken obstacle alerts, and obstacle queries for video demonstration.
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
sys.path.insert(0, str(root_dir / "11_obstacle_awareness"))
sys.path.insert(0, str(root_dir / "03_object_detection"))
sys.path.insert(0, str(root_dir / "08_tts"))

import cv2
import numpy as np

from obstacle_awareness import ObstacleDetector, ObstacleWarningGenerator, ObstacleService
import importlib
obj_det = importlib.import_module("03_object_detection")
ObjectDetector = obj_det.ObjectDetector
from tts import TTSManager
from shared.schemas import IntentRequest, UserIntent, ObstacleHazard

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress phonemizer warnings
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logger = logging.getLogger("NAYRA-Obstacle-Tester")


def test_offline_obstacle_simulation():
    """Verify obstacle detection logic with simulated test hazards."""
    print("\n" + "=" * 65)
    print("  TEST 1: OFFLINE OBSTACLE LOGIC & WARNING VERIFICATION")
    print("=" * 65)

    detector = ObstacleDetector()
    generator = ObstacleWarningGenerator(cooldown_sec=2.0)
    service = ObstacleService(detector=detector, warning_generator=generator, preferred_language="en")
    service.initialize()

    # Case 1: Empty scene (Clear path)
    hazards_clear = service.evaluate_frame(detections=[], frame_width=640, frame_height=480, auto_speak=False)
    query_req = IntentRequest(raw_query="Is there any obstacle in front of me?", intent=UserIntent.OBSTACLE_CHECK)
    resp_clear = service.process_query(query_req)
    print(f"\n[Case 1: Clear Path]")
    print(f"  Hazards Detected : {len(hazards_clear)}")
    print(f"  Spoken Response  : \"{resp_clear.spoken_response_en}\"")

    # Case 2: Chair right in front (Center zone, large bbox -> NEAR)
    mock_chair = type("Detection", (), {
        "class_name": "chair",
        "confidence": 0.88,
        "box_xyxy": (200, 180, 440, 460),
    })()
    hazards_chair = service.evaluate_frame(detections=[mock_chair], frame_width=640, frame_height=480, auto_speak=False)
    resp_chair = service.process_query(query_req)
    print(f"\n[Case 2: Chair in Walking Corridor]")
    print(f"  Hazards Detected : {len(hazards_chair)}")
    if hazards_chair:
        hz = hazards_chair[0]
        print(f"  Hazard Details   : {hz.class_name.upper()} | Zone: {hz.zone} | Proximity: {hz.proximity} | Urgent: {hz.is_urgent}")
        print(f"  Warning Speech   : \"{hz.warning_en}\"")
    print(f"  Query Response   : \"{resp_chair.spoken_response_en}\"")

    print("\n" + "=" * 65)
    print("Tip: To run the live visual navigation corridor, run:")
    print("python scripts/test_obstacle_corridor.py --camera 0")
    print("=" * 65 + "\n")


def test_live_navigation_corridor(camera_source: str | int, tts: TTSManager):
    """Run live camera feed with real-time dynamic navigation corridor & spoken warnings."""
    source_label = "Laptop Built-in Webcam" if str(camera_source).strip() == "0" else str(camera_source)
    print("\n" + "=" * 70)
    print(f"  NAYRA DYNAMIC NAVIGATION CORRIDOR -> {source_label}")
    print("=" * 70)

    # 1. Initialize YOLO detector with sensitive 0.25 threshold
    print("Loading YOLOv10 object detector on CUDA (Confidence: 0.25, All 80 COCO Classes)...")
    from config.settings import ObjectDetectionSettings
    det_settings = ObjectDetectionSettings(
        filter_assistive_only=False,
        confidence_threshold=0.25,
        device="cuda",
    )
    detector_yolo = ObjectDetector(settings=det_settings)

    # 2. Initialize Obstacle Service
    obs_detector = ObstacleDetector()
    obs_generator = ObstacleWarningGenerator(cooldown_sec=4.0)
    obs_service = ObstacleService(
        detector=obs_detector,
        warning_generator=obs_generator,
        preferred_language="en",
    )
    obs_service.initialize()

    # Pre-warm speech
    tts.speak("Navigation corridor active. Path guidance is green when clear and red when blocked.")

    cap_source = int(camera_source) if str(camera_source).strip().isdigit() else camera_source
    cap = cv2.VideoCapture(cap_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not connect to camera at {source_label}")
        return

    print("\n" + "*" * 70)
    print("  CORRIDOR READY FOR VIDEO DEMONSTRATION!")
    print("  Visual Corridor:")
    print("    [GREEN] = PATH CLEAR (Walk forward safely)")
    print("    [RED]   = OBSTACLE DETECTED (Stop or move aside)")
    print("  Controls:")
    print("    [SPACE] -> Ask: 'Is there any obstacle in front of me?'")
    print("    [q]     -> Quit")
    print("*" * 70 + "\n")

    last_speech_time = 0.0
    last_speech_text = ""
    frame_counter = 0
    latest_detections = []
    hazards = []

    # Auto MP4 video recorder
    recordings_dir = root_dir / "recordings"
    recordings_dir.mkdir(exist_ok=True)
    video_out_path = recordings_dir / f"obstacle_corridor_demo_{int(time.time())}.mp4"
    video_writer = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_counter += 1
            h, w = frame.shape[:2]
            vis = frame.copy()

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_out_path), fourcc, 20.0, (w, h))
                print(f"[RECORDER] Automatically recording video to: {video_out_path}")

            # Run detection every 2 frames
            if frame_counter % 2 == 0:
                raw_dets, _ = detector_yolo.detect(frame)
                latest_detections = raw_dets if raw_dets is not None else []

                # Evaluate obstacles from latest detections
                hazards = obs_service.evaluate_frame(
                    detections=latest_detections,
                    frame_width=w,
                    frame_height=h,
                    auto_speak=False,
                )

            # Check if warning should be spoken automatically
            spoken_alert = obs_generator.generate_scene_warning(hazards)
            if spoken_alert:
                _, alert_en, _ = spoken_alert
                print(f"[ALERT SPOKEN]: \"{alert_en}\"")
                tts.speak(alert_en)
                last_speech_text = alert_en
                last_speech_time = time.time()

            # Calculate Walking Corridor (Perspective trapezoid from bottom to center)
            b_left = (int(w * 0.22), h - 15)
            b_right = (int(w * 0.78), h - 15)
            m_left = (int(w * 0.38), int(h * 0.52))
            m_right = (int(w * 0.62), int(h * 0.52))
            corridor_pts = np.array([b_left, m_left, m_right, b_right], dtype=np.int32)

            center_hazards = [hz for hz in hazards if hz.zone == "CENTER" and hz.proximity in ("NEAR", "MEDIUM")]
            urgent_hazards = [hz for hz in hazards if hz.is_urgent]

            if center_hazards or urgent_hazards:
                p_color = (0, 0, 255)  # RED warning
                obs_name = (center_hazards[0].class_name if center_hazards else urgent_hazards[0].class_name).upper()
                p_status = f"WARNING: {obs_name} IN PATH"
            else:
                p_color = (0, 255, 0)  # GREEN clear
                p_status = "NAV PATH: CLEAR"

            # Draw transparent corridor overlay
            poly_overlay = vis.copy()
            cv2.fillPoly(poly_overlay, [corridor_pts], p_color)
            cv2.addWeighted(poly_overlay, 0.22, vis, 0.78, 0, vis)
            cv2.polylines(vis, [corridor_pts], isClosed=True, color=p_color, thickness=3)

            # Corridor Status Badge
            badge_x1, badge_y1 = int(w * 0.22), int(h * 0.52) - 30
            badge_x2, badge_y2 = int(w * 0.78), int(h * 0.52)
            cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 15, 15), -1)
            cv2.rectangle(vis, (badge_x1, badge_y1), (badge_x2, badge_y2), p_color, 2)
            cv2.putText(vis, p_status, (badge_x1 + 18, badge_y1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, p_color, 2)

            # Draw all detected object bounding boxes
            for det in latest_detections:
                x1, y1, x2, y2 = det.box_xyxy
                cname = det.class_name.lower()
                is_path_hazard = any(hz.class_name.lower() == cname for hz in hazards if hz.zone == "CENTER")

                box_color = (0, 0, 255) if is_path_hazard else (0, 230, 255)
                box_thick = 3 if is_path_hazard else 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, box_thick)

                prefix = "[PATH OBSTACLE] " if is_path_hazard else ""
                lbl = f"{prefix}{det.class_name.upper()} {int(det.confidence * 100)}%"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + lw + 6, y1), box_color, -1)
                cv2.putText(vis, lbl, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

            # Top instructions HUD
            cv2.rectangle(vis, (0, 0), (w, 45), (20, 20, 20), -1)
            cv2.line(vis, (0, 45), (w, 45), (0, 255, 255), 2)
            cv2.putText(vis, "NAYRA NAVIGATION | [SPACE] Query Obstacles | [Q] Quit", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Bottom Subtitle banner (shows spoken alerts)
            if last_speech_text and (time.time() - last_speech_time < 8.0):
                cv2.rectangle(vis, (0, h - 60), (w, h), (10, 10, 10), -1)
                cv2.line(vis, (0, h - 60), (w, h - 60), (0, 255, 0), 2)
                cv2.putText(vis, f"NAYRA: \"{last_speech_text[:75]}\"", (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # Record frame to output video file
            if video_writer is not None:
                video_writer.write(vis)

            cv2.imshow("NAYRA Dynamic Navigation Corridor", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (32, ord('o'), ord('O')):  # SPACEBAR or 'O' -> query obstacles
                print("\n[USER QUERY]: 'Is there any obstacle in front of me?'")
                query_req = IntentRequest(raw_query="Is there any obstacle in front of me?", intent=UserIntent.OBSTACLE_CHECK)
                resp = obs_service.process_query(query_req)
                clean_resp = resp.spoken_response_en.strip()
                print(f"[NAYRA]: \"{clean_resp}\"")
                tts.speak(clean_resp)
                last_speech_text = clean_resp
                last_speech_time = time.time()

    finally:
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"\n[RECORDER] Demo video successfully saved to: {video_out_path}")
            # Automatically convert to standard H.264 MP4 for instant Windows playback
            try:
                import shutil
                import subprocess
                if shutil.which("ffmpeg") and video_out_path.exists():
                    ready_path = video_out_path.with_name(video_out_path.stem + "_ready.mp4")
                    print(f"[RECORDER] Encoding to standard H.264 MP4 -> {ready_path.name}...")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(video_out_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(ready_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    if ready_path.exists():
                        print(f"[RECORDER] Ready-to-play video: {ready_path}")
            except Exception as e:
                logger.debug(f"Video post-processing notice: {e}")

        cap.release()
        cv2.destroyAllWindows()
        obs_service.shutdown()


def main():
    parser = argparse.ArgumentParser(description="NAYRA Module 11 Obstacle & Corridor Tester")
    parser.add_argument("--camera", type=str, default="http://192.168.0.101:8080/video", help="Camera index ('0') or IP Webcam URL")
    parser.add_argument("--offline", action="store_true", help="Run offline simulation tests")
    args, _ = parser.parse_known_args()

    if args.offline:
        test_offline_obstacle_simulation()
        return

    # Pre-warm TTS
    tts = TTSManager(phone_speaker=False)

    cam_url = args.camera.strip().strip('"').strip("'")
    if cam_url.startswith("http://http://"):
        cam_url = cam_url.replace("http://http://", "http://")
    elif cam_url and not cam_url.startswith("http://") and not cam_url.startswith("https://") and not cam_url.isdigit():
        cam_url = "http://" + cam_url

    if cam_url.startswith("http") and ":8080" in cam_url and not cam_url.endswith("/video"):
        cam_url = cam_url.rstrip("/") + "/video"

    try:
        test_live_navigation_corridor(cam_url, tts)
    finally:
        tts.shutdown()


if __name__ == "__main__":
    main()
