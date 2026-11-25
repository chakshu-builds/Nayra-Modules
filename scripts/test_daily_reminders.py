"""Interactive test tool for Module 14 (Daily Assistance & Medication Reminders).

Features:
- Live visual camera HUD (or standalone interactive GUI)
- Set quick 5-second & 10-second medication/hydration reminders
- Live real-time countdown timers
- Spoken voice feedback via Kokoro TTS
- Automatic audio alarm when reminder is due: "Attention! It's time to take your medicine."
- Acknowledge / dismiss reminder with spoken confirmation
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "14_daily_assistance"))
sys.path.insert(0, str(root_dir / "08_tts"))

import cv2
import numpy as np

from daily_assistance import DailyService, ReminderDB, ReminderRecord
from tts import TTSManager
from shared.schemas import IntentRequest, UserIntent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Suppress phonemizer warnings
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logger = logging.getLogger("NAYRA-DailyTester")


def test_offline_reminders():
    """Verify DB, scheduling, and callback logic offline."""
    print("\n" + "=" * 65)
    print("  TEST: OFFLINE DAILY REMINDERS VERIFICATION")
    print("=" * 65)

    test_db_path = root_dir / "data" / "test_reminders_offline.db"
    if test_db_path.exists():
        test_db_path.unlink()

    db = ReminderDB(test_db_path)
    fired_reminders = []

    class MockVoice:
        def speak(self, text, priority=None):
            print(f"  [TTS VOICE SPOKE]: \"{text}\"")

    srv = DailyService(voice_output=MockVoice(), db=db, preferred_language="en")
    srv.scheduler.reminder_callback = lambda rec: fired_reminders.append(rec)
    srv.initialize()

    # 1. Set reminder
    print("\n[Step 1] Setting 2-second medicine reminder...")
    req = IntentRequest(raw_query="Remind me to take my medicine in 2 seconds", intent=UserIntent.SET_REMINDER)
    resp = srv.process_query(req)
    print(f"  Result: {resp.spoken_response_en}")

    # 2. List reminders
    print("\n[Step 2] Listing active reminders...")
    req_list = IntentRequest(raw_query="What are my reminders?", intent=UserIntent.LIST_REMINDERS)
    resp_list = srv.process_query(req_list)
    print(f"  Result: {resp_list.spoken_response_en}")

    # 3. Wait for scheduler to trigger
    print("\n[Step 3] Waiting 3.5s for background scheduler to trigger...")
    time.sleep(3.5)

    # 4. Dismiss reminder
    print("\n[Step 4] Confirming medication taken...")
    req_del = IntentRequest(raw_query="I have taken my medicine", intent=UserIntent.DELETE_REMINDER)
    resp_del = srv.process_query(req_del)
    print(f"  Result: {resp_del.spoken_response_en}")

    srv.shutdown()
    if test_db_path.exists():
        test_db_path.unlink()

    print("\n" + "=" * 65)
    print("Tip: To run the live visual interactive reminder demo, run:")
    print("python scripts/test_daily_reminders.py --camera 0")
    print("=" * 65 + "\n")


def run_visual_reminders(camera_source: str | int, tts: TTSManager):
    """Run visual interface (with live camera or canvas) for medication & schedule reminders."""
    source_label = "Laptop Built-in Webcam" if str(camera_source).strip() == "0" else str(camera_source)
    print("\n" + "=" * 70)
    print(f"  NAYRA DAILY ASSISTANCE & MEDICATION REMINDERS -> {source_label}")
    print("=" * 70)

    # Initialize Service
    db_path = root_dir / "data" / "reminders.db"
    db = ReminderDB(db_path)
    # Clear old stale overdue reminders for clean demo start
    try:
        with db._get_connection() as conn:
            conn.execute("UPDATE reminders SET is_completed = 1 WHERE is_completed = 0 AND due_timestamp <= ?", (time.time(),))
            conn.commit()
    except Exception:
        pass

    srv = DailyService(voice_output=tts, db=db, preferred_language="en")

    due_alarm_active: list[ReminderRecord] = []
    def on_reminder_due(rec: ReminderRecord):
        due_alarm_active.append(rec)

    srv.scheduler.reminder_callback = on_reminder_due
    srv.initialize()

    # Pre-warm speech
    tts.speak("Daily assistance active. Press 1 to set a 5-second medicine reminder.")

    # Camera setup
    use_camera = True
    cap_source = int(camera_source) if str(camera_source).strip().isdigit() else camera_source
    cap = cv2.VideoCapture(cap_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[NOTE] Could not open camera at {source_label}. Using interactive canvas mode.")
        use_camera = False

    print("\n" + "*" * 70)
    print("  INTERACTIVE CONTROLS FOR VIDEO DEMO:")
    print("    [1] -> Set Quick 5-Second Medicine Reminder ('Take blood pressure medicine')")
    print("    [2] -> Set Quick 10-Second Hydration Reminder ('Drink water')")
    print("    [L] -> List Active Reminders ('What are my reminders?')")
    print("    [D] or [C] -> Confirm Medicine Taken / Dismiss Due Alarm")
    print("    [S] -> Snooze Due Reminder (by 10 minutes)")
    print("    [Q] -> Quit")
    print("*" * 70 + "\n")

    last_spoken_text = "Daily Assistance active. Ready for reminders."
    last_speech_time = time.time()

    try:
        while True:
            now = time.time()

            if use_camera:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                vis = frame.copy()
            else:
                vis = np.zeros((720, 1280, 3), dtype=np.uint8)
                vis[:] = (20, 20, 24)

            h, w = vis.shape[:2]

            # 1. Top Header Bar
            header_color = (0, 220, 255)
            cv2.rectangle(vis, (0, 0), (w, 50), (15, 15, 15), -1)
            cv2.line(vis, (0, 50), (w, 50), header_color, 2)
            cv2.putText(vis, "NAYRA DAILY ASSISTANCE | MEDICATION & HYDRATION REMINDERS", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, header_color, 2)

            # Current Clock on top right
            clk_str = datetime.datetime.now().strftime("%I:%M:%S %p")
            cv2.putText(vis, clk_str, (w - 200, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

            # 2. Query Active Reminders
            active_rems = db.list_active()
            pending_rems = [r for r in active_rems if r.due_timestamp > now]

            # 3. If any reminder is currently DUE, show prominent flashing Alert Card!
            if due_alarm_active:
                pulse = int(abs(np.sin(now * 5.0)) * 255)
                alarm_color = (0, pulse, 255)
                # Outer border
                cv2.rectangle(vis, (0, 0), (w, h), (0, 0, 255), 8)

                # Center Alarm Box
                box_w, box_h = 760, 200
                bx1 = (w - box_w) // 2
                by1 = (h - box_h) // 2 - 30
                cv2.rectangle(vis, (bx1, by1), (bx1 + box_w, by1 + box_h), (12, 12, 12), -1)
                cv2.rectangle(vis, (bx1, by1), (bx1 + box_w, by1 + box_h), (0, 0, 255), 4)

                top_rec = due_alarm_active[0]
                cv2.putText(vis, "ALARM: MEDICATION REMINDER DUE!", (bx1 + 25, by1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 140, 255), 2)
                cv2.putText(vis, f"ITEM : {top_rec.title.upper()} ({top_rec.category.upper()})", (bx1 + 25, by1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                cv2.putText(vis, "PRESS [D] -> CONFIRM TAKEN  |  PRESS [S] -> SNOOZE 10 MIN", (bx1 + 25, by1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
                cv2.putText(vis, "NAYRA: \"Attention! It's time to take your medicine.\"", (bx1 + 25, by1 + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

            else:
                # Active Reminders Card (Left side)
                panel_w, panel_h = 420, 250
                px1, py1 = 20, 70
                cv2.rectangle(vis, (px1, py1), (px1 + panel_w, py1 + panel_h), (15, 15, 15), -1)
                cv2.rectangle(vis, (px1, py1), (px1 + panel_w, py1 + panel_h), (60, 60, 60), 1)

                cv2.putText(vis, f"SCHEDULED REMINDERS ({len(pending_rems)})", (px1 + 15, py1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                cv2.line(vis, (px1 + 15, py1 + 42), (px1 + panel_w - 15, py1 + 42), (80, 80, 80), 1)

                if not pending_rems:
                    cv2.putText(vis, "No pending reminders.", (px1 + 15, py1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                    cv2.putText(vis, "Press [1] to add 5s medicine alarm", (px1 + 15, py1 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1)
                    cv2.putText(vis, "Press [2] to add 10s water reminder", (px1 + 15, py1 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1)
                else:
                    for i, r in enumerate(pending_rems[:4]):
                        rem_sec = max(0.0, r.due_timestamp - now)
                        ry = py1 + 75 + (i * 38)
                        cv2.putText(vis, f"#{r.id} {r.title}", (px1 + 15, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                        time_lbl = f"in {rem_sec:.1f}s" if rem_sec < 60 else f"in {rem_sec/60:.1f}m"
                        cv2.putText(vis, time_lbl, (px1 + panel_w - 100, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

                # Hotkey Guidance Panel (Bottom Left)
                cv2.rectangle(vis, (20, h - 170), (440, h - 80), (15, 15, 15), -1)
                cv2.rectangle(vis, (20, h - 170), (440, h - 80), (60, 60, 60), 1)
                cv2.putText(vis, "DEMO HOTKEYS:", (30, h - 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
                cv2.putText(vis, "[1] Set 5s Medicine    [2] Set 10s Water", (30, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
                cv2.putText(vis, "[L] List Reminders     [D] Confirm Taken", (30, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            # Bottom Subtitle Banner (Live Spoken Feedback)
            if last_spoken_text and (now - last_speech_time < 8.0):
                cv2.rectangle(vis, (0, h - 65), (w, h), (10, 10, 10), -1)
                cv2.line(vis, (0, h - 65), (w, h - 65), (0, 255, 0), 2)
                cv2.putText(vis, f"NAYRA: \"{last_spoken_text[:85]}\"", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

            cv2.imshow("NAYRA Daily Reminders Demo", vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == ord('1'):
                # 5-second Medicine reminder
                print("\n[USER HOTKEY 1] Setting 5s medicine reminder...")
                req = IntentRequest(raw_query="Remind me to take my blood pressure medicine in 5 seconds", intent=UserIntent.SET_REMINDER)
                resp = srv.process_query(req)
                last_spoken_text = resp.spoken_response_en
                last_speech_time = time.time()
                print(f"[NAYRA]: \"{resp.spoken_response_en}\"")
                tts.speak(resp.spoken_response_en)
            elif key == ord('2'):
                # 10-second Hydration reminder
                print("\n[USER HOTKEY 2] Setting 10s hydration reminder...")
                req = IntentRequest(raw_query="Remind me to drink water in 10 seconds", intent=UserIntent.SET_REMINDER)
                resp = srv.process_query(req)
                last_spoken_text = resp.spoken_response_en
                last_speech_time = time.time()
                print(f"[NAYRA]: \"{resp.spoken_response_en}\"")
                tts.speak(resp.spoken_response_en)
            elif key in (ord('l'), ord('L')):
                # List reminders
                print("\n[USER HOTKEY L] Listing reminders...")
                req = IntentRequest(raw_query="What are my reminders?", intent=UserIntent.LIST_REMINDERS)
                resp = srv.process_query(req)
                last_spoken_text = resp.spoken_response_en
                last_speech_time = time.time()
                print(f"[NAYRA]: \"{resp.spoken_response_en}\"")
                tts.speak(resp.spoken_response_en)
            elif key in (ord('d'), ord('D'), ord('c'), ord('C')):
                # Confirm taken / complete
                print("\n[USER HOTKEY D] Confirming medicine taken...")
                if due_alarm_active:
                    rec = due_alarm_active.pop(0)
                    db.mark_completed(rec.id)
                    spoken = f"Understood, I have marked your {rec.title} as completed."
                else:
                    req = IntentRequest(raw_query="I have taken my medicine", intent=UserIntent.DELETE_REMINDER)
                    resp = srv.process_query(req)
                    spoken = resp.spoken_response_en

                last_spoken_text = spoken
                last_speech_time = time.time()
                print(f"[NAYRA]: \"{spoken}\"")
                tts.speak(spoken)
            elif key in (ord('s'), ord('S')):
                # Snooze
                print("\n[USER HOTKEY S] Snoozing reminder...")
                if due_alarm_active:
                    rec = due_alarm_active.pop(0)
                    db.snooze(rec.id, snooze_seconds=600.0)
                    spoken = f"I have snoozed your {rec.title} reminder for 10 minutes."
                else:
                    req = IntentRequest(raw_query="Snooze reminder", intent=UserIntent.SNOOZE_REMINDER)
                    resp = srv.process_query(req)
                    spoken = resp.spoken_response_en

                last_spoken_text = spoken
                last_speech_time = time.time()
                print(f"[NAYRA]: \"{spoken}\"")
                tts.speak(spoken)

    finally:
        if use_camera:
            cap.release()
        cv2.destroyAllWindows()
        srv.shutdown()


def main():
    parser = argparse.ArgumentParser(description="NAYRA Daily Reminders & Assistance Tester")
    parser.add_argument("--camera", type=str, default="http://192.168.0.101:8080/video", help="Camera index ('0') or IP Webcam URL")
    parser.add_argument("--offline", action="store_true", help="Run offline logic tests without visual UI")
    args, _ = parser.parse_known_args()

    if args.offline:
        test_offline_reminders()
        return

    tts = TTSManager(phone_speaker=False)

    cam_url = args.camera.strip().strip('"').strip("'")
    if cam_url.startswith("http://http://"):
        cam_url = cam_url.replace("http://http://", "http://")
    elif cam_url and not cam_url.startswith("http://") and not cam_url.startswith("https://") and not cam_url.isdigit():
        cam_url = "http://" + cam_url

    if cam_url.startswith("http") and ":8080" in cam_url and not cam_url.endswith("/video"):
        cam_url = cam_url.rstrip("/") + "/video"

    try:
        run_visual_reminders(cam_url, tts)
    finally:
        tts.shutdown()


if __name__ == "__main__":
    main()
