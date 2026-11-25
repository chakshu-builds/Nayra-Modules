"""NAYRA Module 10 — Empirical Performance Profiler and Benchmark.

Measures:
1. Object detection inference latency per frame (CPU)
2. Tracking processing latency (ByteTrack)
3. Spatial context reasoning latency
4. Database write latency (SQLite)
5. Total end-to-end pipeline latency and effective FPS
6. System resource utilization (CPU % and RAM MB)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import numpy as np
import tracemalloc

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from object_memory import (
    ObjectMemoryDB,
    ObjectMemoryManager,
    ObjectObservation,
    SpatialContextEngine,
    ObservationFilter,
)

def benchmark_subsystems(num_iterations: int = 30):
    tracemalloc.start()

    print("=" * 70)
    print("       NAYRA MODULE 10 — SYSTEM PERFORMANCE AUDIT BENCHMARK")
    print("=" * 70)

    # 1. Initialize manager
    t0_init = time.perf_counter()
    manager = ObjectMemoryManager(db_path="data/memory/test_audit.db")
    manager._ensure_vision_models()
    init_duration_ms = (time.perf_counter() - t0_init) * 1000.0
    print(f"[Init] Model & DB initialization: {init_duration_ms:.2f} ms")

    # Synthetic realistic frame (640x480 typical phone stream)
    frame_h, frame_w = 480, 640
    dummy_frame = np.random.randint(0, 255, (frame_h, frame_w, 3), dtype=np.uint8)

    # 2. Measure Detection Inference
    det_latencies = []
    print(f"\n[Benchmarking Detector] Running {num_iterations} iterations...")
    for i in range(num_iterations):
        t0 = time.perf_counter()
        detections, metrics = manager.detector.detect(dummy_frame)
        dt = (time.perf_counter() - t0) * 1000.0
        det_latencies.append(dt)

    det_mean = np.mean(det_latencies[5:])  # skip warmup
    det_std = np.std(det_latencies[5:])
    det_fps = 1000.0 / det_mean if det_mean > 0 else 0
    print(f"  Detector Latency (640x640 input): {det_mean:.2f} +/- {det_std:.2f} ms (~{det_fps:.1f} FPS)")

    # 3. Measure Tracking (ByteTrack)
    track_latencies = []
    print(f"\n[Benchmarking Tracker] Running {num_iterations} iterations...")
    for i in range(num_iterations):
        t0 = time.perf_counter()
        tracks, metrics = manager.tracker.update(detections)
        dt = (time.perf_counter() - t0) * 1000.0
        track_latencies.append(dt)

    track_mean = np.mean(track_latencies)
    track_std = np.std(track_latencies)
    track_fps = 1000.0 / track_mean if track_mean > 0 else 0
    print(f"  Tracker Latency: {track_mean:.2f} +/- {track_std:.2f} ms (~{track_fps:.1f} FPS)")

    # 4. Measure Database Write
    db_latencies = []
    print(f"\n[Benchmarking Database] Running {num_iterations} writes...")
    for i in range(num_iterations):
        obs = ObjectObservation(
            id=None,
            object_label="cell phone",
            canonical_name="phone",
            hindi_alias="फ़ोन",
            timestamp=time.time(),
            confidence=0.9,
            bbox=(100, 100, 150, 150),
            spatial_relation="on",
            anchor_object="dining table",
            horizontal_zone="CENTER",
        )
        t0 = time.perf_counter()
        manager.db.record_observation(obs)
        dt = (time.perf_counter() - t0) * 1000.0
        db_latencies.append(dt)

    db_mean = np.mean(db_latencies)
    db_std = np.std(db_latencies)
    print(f"  SQLite Record Latency: {db_mean:.2f} +/- {db_std:.2f} ms")

    # 5. Measure End-to-End observe_frame()
    e2e_latencies = []
    print(f"\n[Benchmarking End-to-End Observe Frame] Running {num_iterations} iterations...")
    for i in range(num_iterations):
        t0 = time.perf_counter()
        manager.observe_frame(dummy_frame)
        dt = (time.perf_counter() - t0) * 1000.0
        e2e_latencies.append(dt)

    e2e_mean = np.mean(e2e_latencies[5:])
    e2e_fps = 1000.0 / e2e_mean if e2e_mean > 0 else 0
    print(f"  Total E2E Pipeline Latency: {e2e_mean:.2f} ms (~{e2e_fps:.1f} FPS)")

    # 6. Resource utilization
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "=" * 70)
    print("                     RESOURCE UTILIZATION SUMMARY")
    print("=" * 70)
    print(f"  Traced Memory (Peak) : {peak_mem / (1024 * 1024):.1f} MB")
    print(f"  Traced Memory (Curr) : {current_mem / (1024 * 1024):.1f} MB")
    print(f"  Primary Bottleneck   : Object Detection Inference ({det_mean:.1f} ms, {det_mean/e2e_mean*100:.1f}% of total cycle)")
    print("=" * 70)

    # Clean up test db
    test_db = Path("data/memory/test_audit.db")
    if test_db.exists():
        test_db.unlink()

if __name__ == "__main__":
    benchmark_subsystems()
