"""NAYRA Module 10 — Standalone Object Location Memory Interactive Runner.

Provides Alzheimer's and dementia patient assistance by:
1. Observing everyday objects via Android phone camera / webcam using YOLOv10 + ByteTrack.
2. Building spatial relational context (e.g. 'phone on dining table', 'remote near couch').
3. Storing observations in local SQLite database with timestamps and latest pointers.
4. Answering natural language queries in Hindi, English, and Hinglish via terminal and Kokoro TTS.

High-Performance Live Mode:
- Persistent threaded camera stream with zero-lag 1-frame buffer.
- Decoupled capture (15–30 FPS) and asynchronous DB persistence.
- Motion-gated inference skipping redundant static frame processing.

Usage:
    # 1. Interactive terminal query mode (type questions in Hindi / English)
    python 10_object_memory/run_object_memory.py --mode interactive

    # 2. Single query directly from CLI with TTS output
    python 10_object_memory/run_object_memory.py --query "Mera phone kahan hai?" --tts

    # 3. Live observation stream from Android IP Webcam
    python 10_object_memory/run_object_memory.py --mode live --ip 192.168.0.101 --tts

    # 4. Inspect all currently remembered objects
    python 10_object_memory/run_object_memory.py --mode inspect

    # 5. Quick simulated memory test
    python 10_object_memory/run_object_memory.py --mode simulate
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional, Union

import cv2
import numpy as np

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from object_memory import (
    LiveCameraStream,
    MemoryQueryResult,
    ObjectMemoryDB,
    ObjectMemoryManager,
    ObjectObservation,
    VoiceGuidedObjectFinder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Silence phonemizer word-count discrepancy warnings (harmless whitespace tokenization difference)
logging.getLogger("phonemizer").setLevel(logging.ERROR)

logger = logging.getLogger("NAYRA-ObjectMemoryRunner")


def init_tts_manager():
    """Safely initialize Kokoro TTS manager with laptop speaker output."""
    try:
        from tts import TTSManager
        tts = TTSManager(phone_speaker=False)
        logger.info("Initialized Kokoro TTS Manager for verbal responses (Laptop Speakers).")
        return tts
    except Exception as exc:
        logger.warning("Could not initialize TTSManager: %s. Continuing with text-only responses.", exc)
        return None


def run_inspect(manager: ObjectMemoryManager) -> None:
    """Print a formatted table of all currently remembered objects."""
    objects = manager.list_known_objects()
    print("\n" + "=" * 70)
    print("             NAYRA OBJECT LOCATION MEMORY — ACTIVE INVENTORY")
    print("=" * 70)
    if not objects:
        print("  No object observations currently stored in memory.")
        print("  Run live observation mode or simulate to add memories.")
        print("=" * 70 + "\n")
        return

    print(f"{'Object':<15} | {'Hindi Name':<12} | {'Relation':<8} | {'Anchor / Zone':<16} | {'Last Seen':<12}")
    print("-" * 70)
    now = time.time()
    for obj in objects:
        canon = obj.get("canonical_name", "unknown")
        hi = obj.get("hindi_alias") or "-"
        rel = obj.get("spatial_relation") or "-"
        anc = obj.get("anchor_object") or "-"
        ts = obj.get("timestamp", 0)
        elapsed_min = int(round((now - ts) / 60.0))
        time_str = f"{elapsed_min}m ago" if elapsed_min < 60 else f"{int(elapsed_min/60)}h ago"
        print(f"{canon:<15} | {hi:<12} | {rel:<8} | {anc:<16} | {time_str:<12}")
    print("=" * 70 + "\n")


def run_interactive_queries(manager: ObjectMemoryManager, tts: Optional[Any] = None) -> None:
    """Run an interactive terminal loop answering questions in Hindi or English."""
    print("\n" + "=" * 70)
    print("  NAYRA Object Location Memory — Interactive Query Interface")
    print("  Ask questions in Hindi, English, or Hinglish.")
    print("  Examples:")
    print("    - Mera phone kahan hai?")
    print("    - Where did I last see my medicine?")
    print("    - TV ka remote kahan rakha hai?")
    print("    - Mera chashma kahan hai?")
    print("    - Meri keys kahan rakhi thi?")
    print("  Type 'list' to see all remembered objects, or 'exit' to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            query_str = input("\n[Patient / Caregiver] Ask NAYRA: ").strip()
            if not query_str:
                continue
            if query_str.lower() in ("exit", "quit", "q"):
                print("Exiting Object Memory interactive session.")
                break
            if query_str.lower() in ("list", "inventory", "status"):
                run_inspect(manager)
                continue

            result = manager.query(query_str, speak=(tts is not None))
            print("\n" + "-" * 60)
            print(f"[NAYRA Hindi]   : {result.spoken_response_hi}")
            print(f"[NAYRA English] : {result.spoken_response_en}")
            print(f"[Status]        : Found={result.found} | Category={result.time_category} | Conf={result.confidence_level}")
            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def run_simulation(manager: ObjectMemoryManager, tts: Optional[Any] = None) -> None:
    """Populate sample observations and execute clinical test queries."""
    print("\n--- Populating simulated object observations ---")
    now = time.time()

    # 1. Phone placed on dining table 5 minutes ago
    obs_phone = ObjectObservation(
        id=None,
        object_label="cell phone",
        canonical_name="phone",
        hindi_alias="फ़ोन",
        timestamp=now - 300,
        confidence=0.92,
        bbox=(200, 220, 270, 280),
        spatial_relation="on",
        anchor_object="dining table",
        horizontal_zone="CENTER",
    )
    # 2. TV remote near couch 40 minutes ago
    obs_remote = ObjectObservation(
        id=None,
        object_label="remote",
        canonical_name="remote",
        hindi_alias="रिमोट",
        timestamp=now - 2400,
        confidence=0.88,
        bbox=(120, 150, 160, 220),
        spatial_relation="near",
        anchor_object="couch",
        horizontal_zone="LEFT",
    )
    # 3. Medicine bottle on dining table 3 hours ago
    obs_bottle = ObjectObservation(
        id=None,
        object_label="bottle",
        canonical_name="bottle",
        hindi_alias="बोतल",
        timestamp=now - 10800,
        confidence=0.85,
        bbox=(300, 200, 340, 280),
        spatial_relation="on",
        anchor_object="dining table",
        horizontal_zone="RIGHT",
    )

    manager.record_manual_observation(obs_phone)
    manager.record_manual_observation(obs_remote)
    manager.record_manual_observation(obs_bottle)
    print("Recorded 3 sample observations: phone, remote, bottle.")

    test_queries = [
        "Mera phone kahan hai?",
        "Where is the TV remote?",
        "Where did I last see my medicine?",
        "Mera chashma kahan hai?",
        "Meri keys kahan rakhi thi?",
    ]

    print("\n--- Executing clinical memory test queries ---")
    for q in test_queries:
        print(f"\n>> Query: '{q}'")
        res = manager.query(q, speak=(tts is not None))
        print(f"   [HI] {res.spoken_response_hi}")
        print(f"   [EN] {res.spoken_response_en}")
        if tts:
            time.sleep(1.0)


def _draw_performance_hud(
    frame: np.ndarray,
    capture_fps: float,
    display_fps: float,
    manager: ObjectMemoryManager,
) -> None:
    """Draw telemetry overlay with capture rate, display rate, detection latency, and memory stats."""
    h, w = frame.shape[:2]
    telemetry = manager.get_telemetry()
    known_count = len(manager.list_known_objects())

    hud_lines = [
        f"Camera Stream : {capture_fps:.1f} FPS (1-Frame Buffer, Zero Lag)",
        f"Display Rate  : {display_fps:.1f} FPS",
        f"Detector (CPU): {telemetry['inference_latency_ms']:.1f} ms ({telemetry['inference_fps']:.1f} FPS)",
        f"Remembered    : {known_count} objects in memory (Pending DB: {telemetry['pending_db_writes']})",
        f"Optimizations : Motion-Gated={manager.enable_motion_gating} | Async DB={manager.async_db_writes}",
    ]

    hud_h = len(hud_lines) * 20 + 12
    hud_w = 460
    cv2.rectangle(frame, (8, 8), (8 + hud_w, 8 + hud_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (8 + hud_w, 8 + hud_h), (0, 200, 255), 1)

    for i, line in enumerate(hud_lines):
        color = (0, 255, 0) if "FPS" in line else (0, 230, 230)
        cv2.putText(
            frame,
            line,
            (16, 26 + (i * 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )


def run_live_camera(
    manager: ObjectMemoryManager,
    video_source: Union[str, int],
    tts: Optional[Any] = None,
    sample_interval: int = 2,
) -> None:
    """Connect to live camera feed, track objects, and build spatial memory with high FPS and zero lag."""
    print(f"\nConnecting persistent camera stream: {video_source}")
    stream = LiveCameraStream(source=video_source, target_fps=30.0)
    if not stream.start():
        logger.error("Could not start live camera stream from '%s'", video_source)
        return

    print("\n" + "=" * 70)
    print("  NAYRA Object Location Memory — Live Optimized Camera Mode")
    print("  Controls:")
    print("    [q] : Quit stream")
    print("    [s] : Ask question via terminal")
    print("    [i] : Inspect active object inventory")
    print("=" * 70 + "\n")

    frame_counter = 0
    t_prev_disp = time.perf_counter()
    display_fps = 0.0

    try:
        while stream.is_running:
            has_frame, frame, ts = stream.read_latest()
            if not has_frame or frame is None:
                time.sleep(0.005)
                continue

            frame_counter += 1
            t_now = time.perf_counter()
            dt = t_now - t_prev_disp
            t_prev_disp = t_now
            if dt > 0:
                inst_fps = 1.0 / dt
                display_fps = inst_fps if display_fps == 0.0 else (0.15 * inst_fps + 0.85 * display_fps)

            # Sample detection every `sample_interval` frames (motion-gated)
            if frame_counter % sample_interval == 0:
                new_obs = manager.observe_frame(frame, timestamp=ts)
                if new_obs:
                    for obs in new_obs:
                        logger.info("Persisted memory: %s (%s %s)", obs.canonical_name, obs.spatial_relation, obs.anchor_object)

            # Draw visual tracking HUD
            vis = frame.copy()
            if manager.tracker and hasattr(manager.tracker, "tracked_tracks"):
                tracks = [t for t in manager.tracker.tracked_tracks if t.consecutive_missed_frames == 0]
                vis = manager.tracker.draw_tracks(vis, tracks, draw_hud=False)

            _draw_performance_hud(vis, stream.capture_fps, display_fps, manager)

            cv2.imshow("NAYRA - Object Location Memory (Press 'q' to exit)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                q = input("\nEnter query: ")
                res = manager.query(q, speak=(tts is not None))
                print(f"[Answer]: {res.spoken_response_hi}")
            elif key == ord("i"):
                run_inspect(manager)

    finally:
        stream.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="NAYRA Module 10: Object Location Memory Runner")
    parser.add_argument("--mode", choices=["interactive", "live", "inspect", "simulate", "voice"], default="interactive")
    parser.add_argument("--query", type=str, default=None, help="Execute single query and exit")
    parser.add_argument("--ip", type=str, default=None, help="Android IP Webcam IP address (e.g. 192.168.0.101)")
    parser.add_argument("--cam", type=int, default=None, help="Local webcam index (e.g. 0)")
    parser.add_argument("--db-path", type=str, default=None, help="Path to SQLite memory database")
    parser.add_argument("--tts", action="store_true", help="Enable Kokoro TTS spoken responses")
    parser.add_argument("--save-snapshots", action="store_true", help="Save cropped object snapshot images")
    parser.add_argument("--sample-rate", type=int, default=2, help="Frame sampling interval for detection (default: 2)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu", help="Device for object detection (default: cpu to prevent cuDNN symbol collision with Whisper)")
    parser.add_argument("--feedback-cooldown", type=float, default=1.2, help="Post-TTS acoustic echo recovery cooldown in seconds (default: 1.2)")
    parser.add_argument("--no-motion-gating", action="store_true", help="Disable motion-gated inference skipping")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV camera preview GUI window in voice/live mode")
    args = parser.parse_args()

    tts = init_tts_manager() if args.tts else None
    manager = ObjectMemoryManager(
        db_path=args.db_path,
        tts_manager=tts,
        save_snapshots=args.save_snapshots,
        async_db_writes=True,
        enable_motion_gating=not args.no_motion_gating,
        detection_heartbeat_sec=1.0,
        device=args.device,
    )

    if args.query:
        result = manager.query(args.query, speak=(tts is not None))
        print(f"\n[Query]         : {args.query}")
        print(f"[NAYRA Hindi]   : {result.spoken_response_hi}")
        print(f"[NAYRA English] : {result.spoken_response_en}")
        if tts:
            time.sleep(3.0)
        manager.shutdown()
        return

    if args.mode == "inspect":
        run_inspect(manager)
    elif args.mode == "simulate":
        run_simulation(manager, tts)
    elif args.mode == "voice":
        source = f"http://{args.ip}:8080/video" if args.ip else (args.cam if args.cam is not None else 0)
        camera_stream = LiveCameraStream(source=source)
        finder = VoiceGuidedObjectFinder(
            memory_manager=manager,
            camera_stream=camera_stream,
            tts_manager=tts or init_tts_manager(),
            feedback_cooldown_sec=args.feedback_cooldown,
        )
        finder.initialize_subsystems(phone_ip=args.ip)
        print("\n" + "=" * 70)
        print("  NAYRA — Voice-Guided Object Finding Mode")
        print("  Listening for questions like:")
        print("    - 'Mera phone kahan hai?'")
        print("    - 'Remote kahan hai?'")
        print("    - 'Meri bottle kahan hai?'")
        print("  Spoken responses play automatically through laptop speakers.")
        print("=" * 70 + "\n")
        finder.start_listening()
        try:
            if not args.no_preview:
                print("  Live preview active. Point camera at objects (laptop, book, bottle, person).")
                print("  Press 'q' in the camera window to stop.\n")
                while finder._is_running:
                    has_frame, frame, _ = camera_stream.read_latest()
                    if has_frame and frame is not None and frame.size > 0:
                        vis = frame.copy()
                        # Draw detections
                        if hasattr(manager, "_last_frame_detections") and manager._last_frame_detections:
                            for d in manager._last_frame_detections:
                                x1, y1, x2, y2 = d.box_xyxy
                                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                cv2.putText(
                                    vis,
                                    f"{d.class_name.upper()} {int(d.confidence * 100)}%",
                                    (x1, max(18, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 255),
                                    2,
                                )
                        # Draw tracks
                        if manager.tracker:
                            all_tracks = getattr(manager.tracker, "tracked_tracks", []) + getattr(manager.tracker, "tentative_tracks", [])
                            for t in all_tracks:
                                x1, y1, x2, y2 = t.box_xyxy
                                color = (0, 255, 0) if t.is_confirmed else (255, 128, 0)
                                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                                prefix = "TRACK" if t.is_confirmed else "TENT"
                                cv2.putText(
                                    vis,
                                    f"[{prefix} {t.track_id}] {t.class_name}",
                                    (x1, min(vis.shape[0] - 10, y2 + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    color,
                                    2,
                                )

                        live_objs = list(manager._active_live_observations.keys())
                        hud_lines = [
                            f"Live Camera: {source} ({camera_stream.capture_fps:.1f} FPS)",
                            f"Live Memory: {live_objs}",
                            f"TTS Speaking: {tts.is_speaking() if tts and hasattr(tts, 'is_speaking') else False}",
                            "Press 'q' to exit.",
                        ]
                        cv2.rectangle(vis, (10, 10), (480, 10 + len(hud_lines) * 22 + 10), (20, 20, 20), -1)
                        cv2.rectangle(vis, (10, 10), (480, 10 + len(hud_lines) * 22 + 10), (0, 255, 0), 1)
                        for i, line in enumerate(hud_lines):
                            cv2.putText(vis, line, (18, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

                        cv2.imshow("NAYRA - Voice Guided Object Finder (Live Camera Feed)", vis)
                        key = cv2.waitKey(20) & 0xFF
                        if key == ord("q"):
                            break
                    else:
                        time.sleep(0.05)
            else:
                while True:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping voice-guided finder.")
        finally:
            finder.stop_listening()
            if not args.no_preview:
                cv2.destroyAllWindows()
    elif args.mode == "live":
        if args.ip:
            source = f"http://{args.ip}:8080/video"
        elif args.cam is not None:
            source = args.cam
        else:
            source = 0
        run_live_camera(manager, source, tts, sample_interval=args.sample_rate)
    else:
        run_interactive_queries(manager, tts)

    manager.shutdown()


if __name__ == "__main__":
    main()
