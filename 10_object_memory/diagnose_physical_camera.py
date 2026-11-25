"""NAYRA Module 10 — Physical Live-Camera Detection Diagnostic.

Standalone diagnostic tool to verify and trace the physical live-camera pipeline without voice/TTS:
1. Verifies physical camera frames (Android IP Webcam / USB webcam / video source).
2. Logs raw YOLOv10n detections, scores, bounding boxes, and latency.
3. Traces objects across YOLO -> ByteTracker -> ObservationFilter -> ObjectMemoryManager.
4. Renders live camera preview with bounding boxes, labels, and tracking telemetry.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Optional, Union

import cv2
import numpy as np

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config.settings import ObjectDetectionSettings, ObjectTrackingSettings
from object_memory import (
    LiveCameraStream,
    ObjectMemoryDB,
    ObjectMemoryManager,
    ObservationFilter,
    SpatialContextEngine,
)
import importlib

det_mod = importlib.import_module("03_object_detection.detector")
ObjectDetector = det_mod.ObjectDetector
track_mod = importlib.import_module("04_object_tracking.tracker")
ByteTracker = track_mod.ByteTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NAYRA-PhysicalCameraDiagnostic")


def run_diagnostic(
    source: Union[str, int],
    num_frames: Optional[int] = None,
    show_preview: bool = True,
    save_annotated_dir: Optional[str] = None,
    confidence_thresh: float = 0.35,
    min_hits: int = 1,
) -> None:
    """Execute live camera diagnostic tracing frames, YOLO, tracker, and memory."""
    print("\n" + "=" * 78)
    print("      NAYRA MODULE 10 — PHYSICAL LIVE-CAMERA DETECTION DIAGNOSTIC")
    print("=" * 78)
    print(f" Source                   : {source}")
    print(f" Confidence Threshold     : {confidence_thresh}")
    print(f" Tracker Min Hits         : {min_hits}")
    print(f" Show OpenCV Preview      : {show_preview}")
    print(f" Save Annotated Frames To : {save_annotated_dir or 'None'}")
    print("=" * 78 + "\n")

    if save_annotated_dir:
        out_dir = Path(save_annotated_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Detector with full 80 COCO classes on CPU
    det_settings = ObjectDetectionSettings(
        filter_assistive_only=False,
        device="cpu",
        confidence_threshold=confidence_thresh,
    )
    detector = ObjectDetector(settings=det_settings)
    print(f"[OK] ObjectDetector initialized on: {detector._active_device}")

    # 2. Initialize Tracker
    track_settings = ObjectTrackingSettings(min_hits=min_hits)
    tracker = ByteTracker(settings=track_settings)
    print(f"[OK] ByteTracker initialized (min_hits={min_hits}, buffer={track_settings.track_buffer})")

    # 3. Initialize Memory Manager
    memory_manager = ObjectMemoryManager(
        detector=detector,
        tracker=tracker,
        enable_motion_gating=False,  # Bypass motion gating during diagnostic to inspect every frame
        async_db_writes=False,
        device="cpu",
    )
    print("[OK] ObjectMemoryManager initialized.")

    # 4. Connect Camera Stream
    print(f"\n[Step 1] Connecting persistent camera stream to '{source}'...")
    stream = LiveCameraStream(source=source, target_fps=30.0)
    if not stream.start():
        print(f"[FAIL] Could not connect to camera source '{source}'.")
        print("       Check that the camera URL or index is correct and accessible.")
        return

    print(f"[OK] Camera stream connected. Waiting for initial frames...")
    time.sleep(1.0)

    frame_idx = 0
    t_prev = time.perf_counter()
    fps_measured = 0.0

    try:
        while True:
            has_frame, frame, ts = stream.read_latest()
            if not has_frame or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            frame_idx += 1
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_measured = 0.15 * (1.0 / dt) + 0.85 * fps_measured if fps_measured > 0 else (1.0 / dt)

            h, w, c = frame.shape
            mean_intensity = float(np.mean(frame))
            min_val, max_val = int(frame.min()), int(frame.max())

            # Log frame integrity (Step 1)
            is_corrupt = (min_val == max_val) or (frame.size == 0)
            status_str = "CORRUPT/BLANK" if is_corrupt else "VALID"

            # Step 2: YOLO Inference
            t_det0 = time.perf_counter()
            # Raw forward pass to inspect pre-filter raw candidate count
            tensor, ratio, pad_x, pad_y, scale_x, scale_y = detector._preprocess(frame)
            raw_output = detector._forward(tensor)
            preds = raw_output[0] if raw_output.ndim == 3 else raw_output
            raw_candidates = len(preds) if preds is not None else 0

            # Post-processed detections
            detections, _ = detector.detect(frame, timestamp=ts)
            det_lat_ms = (time.perf_counter() - t_det0) * 1000.0

            # Step 3: Tracking update
            tracks, metrics = tracker.update(detections, timestamp=ts)

            # Observe frame in MemoryManager
            committed_obs = memory_manager.observe_frame(frame, timestamp=ts, force_detection=True)

            # Stage-by-Stage Diagnostics Summary
            det_strs = [f"{d.class_name}({d.confidence:.2f})" for d in detections]
            track_strs = [f"ID{t.track_id}:{t.class_name}(h={t.hits})" for t in getattr(tracker, "tracked_tracks", [])]
            tentative_strs = [f"ID{t.track_id}:{t.class_name}" for t in getattr(tracker, "tentative_tracks", [])]
            live_cache_keys = list(memory_manager._active_live_observations.keys())

            print(
                f"[FRAME #{frame_idx:03d}] {w}x{h} ({status_str}, mean={mean_intensity:.1f}) | "
                f"FPS={fps_measured:.1f} | Latency={det_lat_ms:.1f}ms"
            )
            print(f"  Stage 1 (YOLO Raw Candidates) : {raw_candidates} candidates evaluated")
            print(f"  Stage 2 (YOLO Kept Detections): {len(detections)} {det_strs}")
            print(f"  Stage 3A (Confirmed Tracks)   : {len(tracker.tracked_tracks)} {track_strs}")
            print(f"  Stage 3B (Tentative Tracks)   : {len(tracker.tentative_tracks)} {tentative_strs}")
            print(f"  Stage 4 (Memory Live Cache)   : {len(live_cache_keys)} {live_cache_keys}")
            print(f"  Stage 5 (DB Committed New)    : {len(committed_obs)} {[o.canonical_name for o in committed_obs]}")
            print("-" * 78)

            # Step 4: Render Bounding Boxes & Preview
            vis = frame.copy()
            # Draw detections (Yellow boxes)
            for d in detections:
                bx1, by1, bx2, by2 = d.box_xyxy
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                cv2.putText(
                    vis,
                    f"{d.class_name.upper()} {int(d.confidence*100)}%",
                    (bx1, max(18, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

            # Draw tracks (Green boxes with Track ID)
            all_active_tracks = getattr(tracker, "tracked_tracks", []) + getattr(tracker, "tentative_tracks", [])
            for t in all_active_tracks:
                tx1, ty1, tx2, ty2 = t.box_xyxy
                color = (0, 255, 0) if t.is_confirmed else (255, 128, 0)
                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, 2)
                state_prefix = "TRACK" if t.is_confirmed else "TENTATIVE"
                cv2.putText(
                    vis,
                    f"[{state_prefix} {t.track_id}] {t.class_name}",
                    (tx1, min(h - 10, ty2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            # Draw Performance HUD
            hud_lines = [
                f"Source: {source} | {w}x{h} | Rate: {fps_measured:.1f} FPS",
                f"YOLO Inference: {det_lat_ms:.1f} ms | Detections: {len(detections)}",
                f"Confirmed Tracks: {len(tracker.tracked_tracks)} | Tentative: {len(tracker.tentative_tracks)}",
                f"Live Memory: {live_cache_keys}",
            ]
            cv2.rectangle(vis, (10, 10), (520, 10 + len(hud_lines) * 22 + 10), (20, 20, 20), -1)
            cv2.rectangle(vis, (10, 10), (520, 10 + len(hud_lines) * 22 + 10), (0, 255, 0), 1)
            for i, line in enumerate(hud_lines):
                cv2.putText(vis, line, (18, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

            if save_annotated_dir and frame_idx % 5 == 0:
                out_path = Path(save_annotated_dir) / f"diag_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(out_path), vis)

            if show_preview:
                cv2.imshow("NAYRA - Physical Live Camera Detection Diagnostic (Press 'q' to quit)", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nUser pressed 'q'. Stopping diagnostic.")
                    break

            if num_frames and frame_idx >= num_frames:
                print(f"\nCompleted requested {num_frames} diagnostic frames.")
                break

    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user.")
    finally:
        stream.stop()
        if show_preview:
            cv2.destroyAllWindows()
        print("\n[OK] Diagnostic session terminated cleanly.\n")


def main():
    parser = argparse.ArgumentParser(description="NAYRA Physical Camera Diagnostic")
    parser.add_argument("--source", type=str, default="0", help="Camera URL (e.g. http://192.168.0.101:8080/video) or webcam index (0)")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to evaluate (default: 30)")
    parser.add_argument("--no-preview", action="store_true", help="Disable cv2.imshow GUI window")
    parser.add_argument("--save-dir", type=str, default="data/diagnostic_frames", help="Directory to save annotated frames")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--min-hits", type=int, default=1, help="ByteTrack min_hits to confirm track")
    args = parser.parse_args()

    # Parse source as int if digit
    src = int(args.source) if args.source.strip().isdigit() else args.source

    run_diagnostic(
        source=src,
        num_frames=args.frames,
        show_preview=not args.no_preview,
        save_annotated_dir=args.save_dir,
        confidence_thresh=args.confidence,
        min_hits=args.min_hits,
    )


if __name__ == "__main__":
    main()
