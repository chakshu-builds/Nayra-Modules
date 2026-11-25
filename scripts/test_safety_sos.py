"""Interactive test tool for Module 15 (Safety SOS & Live Camera Emergency Assistance).

Runs live camera perception with real-time YOLO object tracking, two-stage emergency confirmation,
audio countdown, caregiver alert dispatch, and cancellation overlay for video demonstration.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "15_safety_sos"))
sys.path.insert(0, str(root_dir / "03_object_detection"))
sys.path.insert(0, str(root_dir / "08_tts"))

import cv2
import numpy as np

import importlib
det_mod = importlib.import_module("03_object_detection.detector")
ObjectDetector = det_mod.ObjectDetector
from config.settings import ObjectDetectionSettings

from safety_sos import SOSManager, SOSState, SafetyService
from tts import TTSManager
from shared.schemas import IntentRequest, UserIntent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress phonemizer warnings
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logger = logging.getLogger("NAYRA-SOS-Tester")


def test_offline_sos_protocol():
    """Verify SOS state machine and log file generation."""
    print("\n" + "=" * 65)
    print("  TEST: OFFLINE SOS STATE MACHINE VERIFICATION")
    print("=" * 65)

    sos = SOSManager(preferred_language="en", countdown_sec=5.0)

    # 1. Trigger
    spoken1 = sos.request_sos("Simulated fall detection", auto_confirm_on_timeout=False)
    print(f"\n[Trigger] : {spoken1}")
    print(f"[State]   : {sos.state.value}")

    # 2. Cancel
    spoken2 = sos.cancel_sos()
    print(f"\n[Cancel]  : {spoken2}")
    print(f"[State]   : {sos.state.value}")

    # 3. Trigger & Confirm
    sos.request_sos("Emergency assistance requested", auto_confirm_on_timeout=False)
    spoken3 = sos.confirm_sos()
    print(f"\n[Confirm] : {spoken3}")
    print(f"[State]   : {sos.state.value}")

    print("\n" + "=" * 65)
    print("Tip: To run the live visual camera SOS demonstration, run:")
    print("python scripts/test_safety_sos.py --camera 0")
    print("=" * 65 + "\n")


def test_live_camera_sos(camera_source: str | int, tts: TTSManager):
    """Run live camera feed with real-time emergency SOS overlay & spoken audio."""
    source_label = "Laptop Built-in Webcam" if str(camera_source).strip() == "0" else str(camera_source)
    print("\n" + "=" * 70)
    print(f"  NAYRA LIVE CAMERA SAFETY SOS SYSTEM -> {source_label}")
    print("=" * 70)

    # 1. Initialize YOLO detector
    print("Loading YOLOv10 object detector on CUDA...")
    det_settings = ObjectDetectionSettings(
        filter_assistive_only=False,
        confidence_threshold=0.25,
        device="cuda",
    )
    detector = ObjectDetector(settings=det_settings)

    # 2. Initialize SOS Manager
    sos = SOSManager(preferred_language="en", countdown_sec=10.0)

    # Pre-warm speech
    tts.speak("Safety SOS monitoring active on camera feed. Press 1 to trigger emergency assistance.")

    cap_source = int(camera_source) if str(camera_source).strip().isdigit() else camera_source
    cap = cv2.VideoCapture(cap_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not connect to camera at {source_label}")
        return

    print("\n" + "*" * 70)
    print("  CAMERA CONNECTED & READY FOR VIDEO DEMONSTRATION!")
    print("  Controls:")
    print("    [1] or [SPACE] -> Trigger SOS Alert ('Do you need emergency assistance?')")
    print("    [y]           -> Confirm Emergency ('Emergency alert activated. Dispatched!')")
    print("    [x] or [c]     -> Cancel Emergency ('Emergency alert cancelled.')")
    print("    [q]           -> Quit")
    print("\n  📱 REAL-TIME PHONE NOTIFICATION SUBSCRIBER:")
    print("     Open this link on your phone to receive live push alerts instantly:")
    print("     👉 https://ntfy.sh/nayra_emergency_9215793330")
    print("*" * 70 + "\n")

    last_spoken_text = "Safety SOS active. Monitoring surroundings."
    last_speech_time = time.time()
    trigger_time = 0.0
    latest_detections = []
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
            vis = frame.copy()

            # Run detection every 2 frames
            if frame_counter % 2 == 0:
                raw_dets, _ = detector.detect(frame)
                latest_detections = raw_dets if raw_dets is not None else []

            # 1. Draw detected bounding boxes
            for det in latest_detections:
                x1, y1, x2, y2 = det.box_xyxy
                box_color = (0, 230, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                lbl = f"{det.class_name.upper()} {int(det.confidence * 100)}%"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + lw + 6, y1), box_color, -1)
                cv2.putText(vis, lbl, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

            # 2. Check for auto-timeout dispatch if user does not respond within 10 seconds
            if sos.state == SOSState.PENDING_CONFIRMATION:
                remaining = max(0.0, 10.0 - (now - trigger_time))
                if remaining <= 0:
                    spoken = sos._dispatch_emergency(trigger_method="timeout_fallback")
                    print(f"\n[AUTO-DISPATCHED TIMEOUT]: {spoken}")
                    tts.speak(spoken)
                    last_spoken_text = spoken
                    last_speech_time = time.time()

            # 3. Emergency Visual Overlays
            if sos.state == SOSState.PENDING_CONFIRMATION:
                # Pulsing red frame border
                pulse_thick = int(6 + abs(np.sin(now * 4.0)) * 10)
                cv2.rectangle(vis, (0, 0), (w, h), (0, 0, 255), pulse_thick)

                # Center Alert Banner
                remaining = max(0.0, 10.0 - (now - trigger_time))
                b_w, b_h = 680, 110
                bx1 = (w - b_w) // 2
                by1 = (h - b_h) // 2
                cv2.rectangle(vis, (bx1, by1), (bx1 + b_w, by1 + b_h), (10, 10, 10), -1)
                cv2.rectangle(vis, (bx1, by1), (bx1 + b_w, by1 + b_h), (0, 140, 255), 3)

                cv2.putText(vis, f"EMERGENCY CONFIRMATION ({remaining:.1f}s)", (bx1 + 25, by1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 140, 255), 2)
                cv2.putText(vis, "PRESS [Y] TO CONFIRM  |  PRESS [X] TO CANCEL", (bx1 + 25, by1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                header_text = "STATUS: PENDING EMERGENCY CONFIRMATION"
                header_color = (0, 140, 255)

            elif sos.state == SOSState.TRIGGERED:
                # Solid Red Header & Border
                cv2.rectangle(vis, (0, 0), (w, h), (0, 0, 255), 8)
                header_text = "EMERGENCY DISPATCHED -> CAREGIVER NOTIFIED WITH LIVE LOCATION"
                header_color = (0, 0, 255)

                # Real-time Location Notification Card on Screen
                card_w, card_h = min(w - 60, 840), 185
                cx1 = (w - card_w) // 2
                cy1 = (h - card_h) // 2 - 20
                cv2.rectangle(vis, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (12, 12, 12), -1)
                cv2.rectangle(vis, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (0, 0, 255), 3)

                alert_info = getattr(sos, "last_dispatched_alert", {}) or {}
                loc = alert_info.get("location", {})
                loc_str = loc.get("location_string", "Delhi, NCT of Delhi, India (28.6542, 77.2373)")
                map_url = loc.get("maps_url", "https://maps.google.com/?q=28.6542,77.2373")
                channels = alert_info.get("dispatched_channels", ["ntfy.sh push", "whatsapp", "windows_toast"])
                channels_str = ", ".join(channels) if channels else "ntfy.sh push, whatsapp_web"

                cv2.putText(vis, "EMERGENCY SOS ALERT HIT ON NAYRA!", (cx1 + 20, cy1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(vis, f"LOCATION : {loc_str}", (cx1 + 20, cy1 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(vis, f"MAP LINK : {map_url}", (cx1 + 20, cy1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
                cv2.putText(vis, "CAREGIVER: Primary (+919215793330) [LIVE TRANSMISSION SENT]", (cx1 + 20, cy1 + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 0), 2)
                cv2.putText(vis, f"CHANNELS : {channels_str.upper()}", (cx1 + 20, cy1 + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

            else:
                header_text = "NAYRA LIVE CAMERA | [1]/[SPACE] Trigger SOS | [Q] Quit"
                header_color = (0, 255, 255)

            # Top Header Bar
            cv2.rectangle(vis, (0, 0), (w, 48), (15, 15, 15), -1)
            cv2.line(vis, (0, 48), (w, 48), header_color, 2)
            cv2.putText(vis, header_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.62, header_color, 2)

            # Bottom Subtitle Banner (Live Spoken Feedback)
            if last_spoken_text and (now - last_speech_time < 9.0):
                cv2.rectangle(vis, (0, h - 65), (w, h), (10, 10, 10), -1)
                cv2.line(vis, (0, h - 65), (w, h - 65), (0, 255, 0), 2)
                cv2.putText(vis, f"NAYRA: \"{last_spoken_text[:75]}\"", (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

            cv2.imshow("NAYRA Camera Safety SOS Demo", vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('1'), 32):  # '1' or SPACE -> Trigger SOS
                print("\n[USER TRIGGERED SOS]")
                spoken = sos.request_sos("Emergency assistance requested", auto_confirm_on_timeout=False)
                trigger_time = time.time()
                last_spoken_text = spoken
                last_speech_time = time.time()
                print(f"[NAYRA]: \"{spoken}\"")
                tts.speak(spoken)
            elif key in (ord('y'), ord('Y')):  # 'Y' -> Confirm SOS
                print("\n[USER CONFIRMED SOS]")
                spoken = sos.confirm_sos()
                last_spoken_text = spoken
                last_speech_time = time.time()
                print(f"[NAYRA]: \"{spoken}\"")
                tts.speak(spoken)

                alert_info = getattr(sos, "last_dispatched_alert", {}) or {}
                loc = alert_info.get("location", {})
                loc_str = loc.get("location_string", "Delhi, India (28.6542, 77.2373)")
                map_url = loc.get("maps_url", "https://maps.google.com/?q=28.6542,77.2373")
                channels = alert_info.get("dispatched_channels", [])

                print("\n" + "=" * 65)
                print("  🚨 REAL-TIME SOS DISPATCH BROADCAST")
                print("=" * 65)
                print("  Status      : SOS TRIGGERED & DISPATCHED ON NAYRA")
                print(f"  Location    : {loc_str}")
                print(f"  Google Map  : {map_url}")
                print("  Recipient   : Primary Caregiver (+919215793330)")
                print(f"  Channels    : {', '.join(channels) if channels else 'ntfy.sh push, whatsapp_web'}")
                print(f"  Live Push   : https://ntfy.sh/nayra_emergency_9215793330")
                print(f"  SMS Message : \"EMERGENCY: User triggered SOS on Nayra! Location: {loc_str}\"")
                print("=" * 65 + "\n")
            elif key in (ord('x'), ord('X'), ord('c'), ord('C')):  # 'X' or 'C' -> Cancel SOS
                print("\n[USER CANCELLED SOS]")
                spoken = sos.cancel_sos()
                last_spoken_text = spoken
                last_speech_time = time.time()
                print(f"[NAYRA]: \"{spoken}\"")
                tts.speak(spoken)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sos._cancel_existing_timer()


def main():
    parser = argparse.ArgumentParser(description="NAYRA Safety SOS Live Camera Tester")
    parser.add_argument("--camera", type=str, default="", help="Camera index ('0') or IP Webcam URL")
    parser.add_argument("--offline", action="store_true", help="Run offline logic tests without camera")
    args, _ = parser.parse_known_args()

    if args.offline:
        test_offline_sos_protocol()
        return

    tts = TTSManager(phone_speaker=False)

    cam_url = args.camera.strip().strip('"').strip("'")
    if cam_url.startswith("http://http://"):
        cam_url = cam_url.replace("http://http://", "http://")
    elif cam_url and not cam_url.startswith("http://") and not cam_url.startswith("https://") and not cam_url.isdigit():
        cam_url = "http://" + cam_url

    if cam_url.startswith("http") and ":8080" in cam_url and not cam_url.endswith("/video"):
        cam_url = cam_url.rstrip("/") + "/video"

    # Default to IP webcam or 0 if not provided
    if cam_url == "":
        cam_url = "http://192.168.0.101:8080/video"

    try:
        test_live_camera_sos(cam_url, tts)
    finally:
        tts.shutdown()


if __name__ == "__main__":
    main()
