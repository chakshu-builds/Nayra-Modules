"""NAYRA Module 07 — Real-World Performance & Scalability Audit.

Conducts rigorous benchmarking on 1080p (1920x1080) camera frames:
1. End-to-end latency profiling across 50 warm iterations (Median, P95, Mean, Min, Max).
2. Cold-start vs warm inference measurement.
3. Subsystem latency breakdown: DBNet detection, cropping, script routing, recognition.
4. Sequential vs Batched recognition speedup.
5. Camera display FPS impact under on-demand OCR (AsyncOCRWorker).
6. Multi-module concurrent stress test: Modules 03 (YOLOv10), 04 (ByteTrack), 05 (MiDaS), 07 (OCR).
7. GPU VRAM and Host RAM telemetry.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Configure imports
sys.path.insert(0, os.path.abspath(str(Path(__file__).parent.parent)))
sys.modules["object_detection"] = importlib.import_module("03_object_detection")
sys.modules["object_tracking"] = importlib.import_module("04_object_tracking")
sys.modules["spatial_awareness"] = importlib.import_module("05_spatial_awareness")
sys.modules["ocr"] = importlib.import_module("07_ocr")

import ocr
from ocr import AsyncOCRWorker, OCREngine, OCRResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-PerformanceAudit")


# ---------------------------------------------------------------------------
# Telemetry Utilities
# ---------------------------------------------------------------------------
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_host_ram_mb() -> float:
    """Return process resident working set memory in MB."""
    try:
        psapi = ctypes.WinDLL("psapi.dll")
        kernel32 = ctypes.WinDLL("kernel32.dll")
        GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return 0.0


def get_gpu_vram_mb() -> Dict[str, float]:
    """Query NVIDIA GPU memory utilization in MB via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
        return {"total_mb": parts[0], "used_mb": parts[1], "free_mb": parts[2]}
    except Exception:
        return {"total_mb": 0.0, "used_mb": 0.0, "free_mb": 0.0}


def create_1080p_test_scene() -> np.ndarray:
    """Compose an authentic 1080p camera frame with real Hindi and English signs."""
    canvas = np.ones((1080, 1920, 3), dtype=np.uint8) * 235

    # Paste real Delhi metro sign (mixed Hindi & English)
    metro = cv2.imread("07_ocr/data/real_camera/real_01_metro_mixed.jpg")
    if metro is not None:
        m_h, m_w = metro.shape[:2]
        scaled_w = 900
        scaled_h = int(round(m_h * (scaled_w / float(m_w))))
        metro_res = cv2.resize(metro, (scaled_w, min(scaled_h, 500)))
        sh, sw = metro_res.shape[:2]
        canvas[60 : 60 + sh, 60 : 60 + sw] = metro_res

    # Paste real New Delhi railway sign (English + numerals)
    railway = cv2.imread("07_ocr/data/real_camera/real_02_railway_en.jpg")
    if railway is not None:
        r_h, r_w = railway.shape[:2]
        scaled_w = 850
        scaled_h = int(round(r_h * (scaled_w / float(r_w))))
        rail_res = cv2.resize(railway, (scaled_w, min(scaled_h, 450)))
        sh, sw = rail_res.shape[:2]
        canvas[60 : 60 + sh, 1010 : 1010 + sw] = rail_res

    # Paste real store receipt / text at bottom
    receipt = cv2.imread("07_ocr/data/real_camera/real_04_receipt.jpg")
    if receipt is not None:
        rc_h, rc_w = receipt.shape[:2]
        scaled_w = 900
        scaled_h = int(round(rc_h * (scaled_w / float(rc_w))))
        rec_res = cv2.resize(receipt, (scaled_w, min(scaled_h, 450)))
        sh, sw = rec_res.shape[:2]
        canvas[580 : 580 + sh, 60 : 60 + sw] = rec_res

    # Add realistic camera background noise / gradient
    noise = np.random.randint(-10, 10, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def run_benchmark() -> Dict[str, Any]:
    logger.info("======================================================================")
    logger.info("  NAYRA Module 07 — Real-World Performance & Scalability Audit        ")
    logger.info("======================================================================")

    vram_baseline = get_gpu_vram_mb()
    ram_baseline = get_host_ram_mb()
    logger.info("Baseline VRAM: %s MB, Host RAM: %s MB", vram_baseline["used_mb"], ram_baseline)

    test_1080p_frame = create_1080p_test_scene()
    logger.info("Created 1080p test scene: shape=%s", test_1080p_frame.shape)

    # -----------------------------------------------------------------------
    # Part 1: Cold Start vs Warm Inference
    # -----------------------------------------------------------------------
    logger.info("Measuring cold start latency (uninitialized engine)...")
    t_cold_start = time.perf_counter()
    fresh_engine = OCREngine(preferred_provider="CUDA", use_batching=True)
    cold_res = fresh_engine.process_image(test_1080p_frame)
    cold_latency_ms = (time.perf_counter() - t_cold_start) * 1000.0
    logger.info("Cold-start total latency (engine init + 1st frame): %.2f ms", cold_latency_ms)
    logger.info("Cold-start inference only: %.2f ms (detections: %d)", cold_res.total_time_ms, cold_res.detection_count)

    vram_ocr_loaded = get_gpu_vram_mb()
    logger.info("OCR Loaded VRAM: %s MB (+%.1f MB)", vram_ocr_loaded["used_mb"], vram_ocr_loaded["used_mb"] - vram_baseline["used_mb"])

    # -----------------------------------------------------------------------
    # Part 2: 50 Iterations Warm Benchmark (Batched vs Sequential)
    # -----------------------------------------------------------------------
    NUM_ITERATIONS = 50
    logger.info("Running %d warm iterations with BATCHED recognition...", NUM_ITERATIONS)
    fresh_engine.use_batching = True

    batched_totals: List[float] = []
    batched_dets: List[float] = []
    batched_routes: List[float] = []
    batched_recs: List[float] = []

    for i in range(NUM_ITERATIONS):
        res = fresh_engine.process_image(test_1080p_frame)
        batched_totals.append(res.total_time_ms)
        batched_dets.append(res.detection_time_ms)
        batched_routes.append(res.routing_time_ms)
        batched_recs.append(res.recognition_time_ms)
        if (i + 1) % 10 == 0:
            logger.info("  Batched iteration %d/%d: %.2f ms", i + 1, NUM_ITERATIONS, res.total_time_ms)

    logger.info("Running 20 warm iterations with SEQUENTIAL recognition (baseline)...")
    fresh_engine.use_batching = False
    seq_totals: List[float] = []
    seq_recs: List[float] = []

    for i in range(20):
        res = fresh_engine.process_image(test_1080p_frame)
        seq_totals.append(res.total_time_ms)
        seq_recs.append(res.recognition_time_ms)

    fresh_engine.use_batching = True  # Restore default

    # -----------------------------------------------------------------------
    # Part 3: Camera Capture & Display FPS Benchmark
    # -----------------------------------------------------------------------
    logger.info("Benchmarking camera display FPS under simulated camera loop...")
    worker = AsyncOCRWorker(engine=fresh_engine)

    def measure_display_fps(with_ocr: bool, num_frames: int = 150) -> Tuple[float, float]:
        """Simulate video display loop with or without on-demand OCR requests."""
        fps_list = []
        t0 = time.perf_counter()
        req_count = 0

        for f_idx in range(num_frames):
            t_frame_start = time.perf_counter()
            # Simulate frame arrival
            cur_frame = test_1080p_frame.copy()

            if with_ocr and f_idx % 15 == 0:
                worker.trigger(cur_frame)
                req_count += 1

            # Simulate HUD rendering and display
            ocr_res = worker.get_latest_result()
            _ = cur_frame.shape

            t_frame_end = time.perf_counter()
            # Enforce 30 FPS target pacing if under budget
            elapsed = t_frame_end - t_frame_start
            if elapsed < 0.0333:
                time.sleep(0.0333 - elapsed)
            fps_list.append(1.0 / max(time.perf_counter() - t_frame_start, 1e-4))

        total_time = time.perf_counter() - t0
        mean_fps = num_frames / total_time
        return mean_fps, round(np.percentile(fps_list, 5), 1)

    fps_baseline, fps_p5_baseline = measure_display_fps(with_ocr=False)
    fps_with_ocr, fps_p5_ocr = measure_display_fps(with_ocr=True)
    worker.shutdown()
    logger.info("Camera display FPS without OCR: Mean=%.2f FPS (P5=%.1f FPS)", fps_baseline, fps_p5_baseline)
    logger.info("Camera display FPS with on-demand OCR: Mean=%.2f FPS (P5=%.1f FPS)", fps_with_ocr, fps_p5_ocr)

    # -----------------------------------------------------------------------
    # Part 4: Multi-Module Concurrent Load Testing (03, 04, 05, 07)
    # -----------------------------------------------------------------------
    logger.info("Testing concurrent multi-module perception pipeline...")
    from config.settings import ObjectDetectionSettings, ObjectTrackingSettings, SpatialAwarenessSettings
    from object_detection import ObjectDetector
    from object_tracking import ByteTracker
    from spatial_awareness import SpatialAnalyzer

    det_settings = ObjectDetectionSettings(model_name="yolov10n", device="cuda")
    obj_detector = ObjectDetector(det_settings)
    track_settings = ObjectTrackingSettings()
    obj_tracker = ByteTracker(track_settings)
    spatial_settings = SpatialAwarenessSettings(model_name="midas_v21_small", device="cuda")
    spatial_analyzer = SpatialAnalyzer(settings=spatial_settings)

    vram_multi_loaded = get_gpu_vram_mb()
    logger.info("Multi-module Loaded VRAM: %s MB (+%.1f MB)", vram_multi_loaded["used_mb"], vram_multi_loaded["used_mb"] - vram_baseline["used_mb"])

    # Measure perception loop FPS and concurrent OCR execution
    perception_worker = AsyncOCRWorker(engine=fresh_engine)
    perception_times: List[float] = []
    pipe_fps_list: List[float] = []

    t_pipe_start = time.perf_counter()
    num_pipe_frames = 60

    for f_idx in range(num_pipe_frames):
        t_f0 = time.perf_counter()
        cur_frame = test_1080p_frame.copy()

        # Foreground Perception: YOLOv10 + ByteTrack + MiDaS
        dets, _ = obj_detector.detect(cur_frame)
        tracks, _ = obj_tracker.update(dets)
        spat = spatial_analyzer.analyze(cur_frame, tracks)

        # Trigger OCR on-demand every 15 frames
        if f_idx % 15 == 0:
            perception_worker.trigger(cur_frame)

        latest_ocr = perception_worker.get_latest_result()

        t_f1 = time.perf_counter()
        frame_elapsed = t_f1 - t_f0
        perception_times.append(frame_elapsed * 1000.0)
        pipe_fps_list.append(1.0 / max(frame_elapsed, 1e-4))

    total_pipe_elapsed = time.perf_counter() - t_pipe_start
    multi_mean_fps = num_pipe_frames / total_pipe_elapsed
    perception_worker.shutdown()

    vram_peak = get_gpu_vram_mb()
    ram_peak = get_host_ram_mb()

    # -----------------------------------------------------------------------
    # Statistics Aggregation
    # -----------------------------------------------------------------------
    stats = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)",
            "onnxruntime_providers": fresh_engine.execution_providers,
            "test_resolution": "1920x1080 (1080p Full HD)",
            "warm_iterations": NUM_ITERATIONS,
        },
        "latency_audit_1080p": {
            "cold_start_total_ms": round(cold_latency_ms, 2),
            "cold_start_inference_ms": round(cold_res.total_time_ms, 2),
            "warm_inference_ms": {
                "median": round(float(np.median(batched_totals)), 2),
                "p95": round(float(np.percentile(batched_totals, 95)), 2),
                "mean": round(float(np.mean(batched_totals)), 2),
                "min": round(float(np.min(batched_totals)), 2),
                "max": round(float(np.max(batched_totals)), 2),
            },
            "subsystem_breakdown_ms": {
                "dbnet_detection": {
                    "median": round(float(np.median(batched_dets)), 2),
                    "mean": round(float(np.mean(batched_dets)), 2),
                    "p95": round(float(np.percentile(batched_dets, 95)), 2),
                },
                "script_routing": {
                    "median": round(float(np.median(batched_routes)), 2),
                    "mean": round(float(np.mean(batched_routes)), 2),
                    "p95": round(float(np.percentile(batched_routes, 95)), 2),
                },
                "batched_recognition": {
                    "median": round(float(np.median(batched_recs)), 2),
                    "mean": round(float(np.mean(batched_recs)), 2),
                    "p95": round(float(np.percentile(batched_recs, 95)), 2),
                },
                "sequential_recognition_baseline": {
                    "median": round(float(np.median(seq_recs)), 2),
                    "mean": round(float(np.mean(seq_recs)), 2),
                    "p95": round(float(np.percentile(seq_recs, 95)), 2),
                },
                "recognition_speedup_factor": round(float(np.median(seq_recs)) / max(float(np.median(batched_recs)), 1e-3), 2),
            },
        },
        "fps_audit": {
            "camera_preview_only_fps": round(fps_baseline, 2),
            "camera_preview_with_ocr_worker_fps": round(fps_with_ocr, 2),
            "multi_module_pipeline_with_ocr_fps": round(multi_mean_fps, 2),
            "multi_module_frame_latency_ms": {
                "median": round(float(np.median(perception_times)), 2),
                "p95": round(float(np.percentile(perception_times, 95)), 2),
            },
        },
        "resource_utilization": {
            "vram_baseline_mb": vram_baseline["used_mb"],
            "vram_ocr_engine_mb": vram_ocr_loaded["used_mb"],
            "vram_multi_module_peak_mb": vram_peak["used_mb"],
            "vram_total_available_mb": vram_peak["total_mb"],
            "vram_margin_headroom_mb": vram_peak["free_mb"],
            "host_ram_working_set_peak_mb": ram_peak,
        },
    }

    report_path = Path(__file__).parent / "performance_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Performance Audit Complete! Report saved to %s", report_path)
    logger.info(
        "Warm 1080p Latency: Median=%.2f ms, P95=%.2f ms | DBNet=%.2f ms | Batch Rec=%.2f ms (Speedup: %.2fx)",
        stats["latency_audit_1080p"]["warm_inference_ms"]["median"],
        stats["latency_audit_1080p"]["warm_inference_ms"]["p95"],
        stats["latency_audit_1080p"]["subsystem_breakdown_ms"]["dbnet_detection"]["median"],
        stats["latency_audit_1080p"]["subsystem_breakdown_ms"]["batched_recognition"]["median"],
        stats["latency_audit_1080p"]["subsystem_breakdown_ms"]["recognition_speedup_factor"],
    )
    logger.info(
        "Display FPS: Preview=%.1f FPS | OCR Worker=%.1f FPS | Multi-Module=%.1f FPS | Peak VRAM=%s MB",
        stats["fps_audit"]["camera_preview_only_fps"],
        stats["fps_audit"]["camera_preview_with_ocr_worker_fps"],
        stats["fps_audit"]["multi_module_pipeline_with_ocr_fps"],
        stats["resource_utilization"]["vram_multi_module_peak_mb"],
    )
    return stats


if __name__ == "__main__":
    run_benchmark()
