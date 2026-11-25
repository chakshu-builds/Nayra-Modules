"""NAYRA Module 07 — Comprehensive DBNet Text Detection Benchmark & Real-World Evaluation.

Benchmarks:
1. End-to-end latency and breakdown (pre, infer, post, total) across:
   - 640x640
   - 960x960
   - 1920x1080 (1080p camera input)
   Under both CUDAExecutionProvider (RTX 4050) and CPUExecutionProvider.
2. Real-world text document scenarios:
   - Single-line Hindi text line
   - Single-line English text line
   - Mixed bilingual document (Hindi + English)
   - Multi-line dense document (5 lines)
   - Nearby text lines with narrow inter-line spacing
3. GPU memory utilization metrics.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Add 07_ocr to sys.path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from detector.dbnet_detector import DBNetDetector, configure_cuda_dlls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark-Det")


def get_gpu_memory_mb() -> float:
    """Query current GPU allocated memory in MB if PyTorch CUDA is active."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    return 0.0


def create_eval_scenes() -> Dict[str, Dict[str, Any]]:
    """Synthesize evaluation scenes using real cropped text lines."""
    scenes = {}

    h_crop_1 = cv2.imread("07_ocr/data/poc/sample_01.png")  # "नमस्ते दुनिया"
    h_crop_2 = cv2.imread("07_ocr/data/poc/sample_02.png")  # "भारत एक महान देश है"
    en_crop_1 = cv2.imread("07_ocr/data/poc_en/sample_en_01.png")  # "HELLO WORLD 2026"
    en_crop_2 = cv2.imread("07_ocr/data/poc_en/sample_en_02.png")  # "NAYRA ASSISTIVE AI"
    en_crop_3 = cv2.imread("07_ocr/data/poc_en/sample_en_03.png")  # "INVOICE #98214"

    # Scene 1: Single Hindi Line
    s1 = np.ones((200, 500, 3), dtype=np.uint8) * 255
    s1[70 : 70 + h_crop_1.shape[0], 50 : 50 + h_crop_1.shape[1]] = h_crop_1
    scenes["single_hindi"] = {
        "image": s1,
        "description": "Single Hindi Devanagari text line crop",
        "expected_min_lines": 1,
    }

    # Scene 2: Single English Line
    s2 = np.ones((200, 500, 3), dtype=np.uint8) * 255
    s2[70 : 70 + en_crop_1.shape[0], 50 : 50 + en_crop_1.shape[1]] = en_crop_1
    scenes["single_english"] = {
        "image": s2,
        "description": "Single English Latin text line crop",
        "expected_min_lines": 1,
    }

    # Scene 3: Mixed Bilingual Notice
    s3 = np.ones((480, 640, 3), dtype=np.uint8) * 255
    s3[80 : 80 + en_crop_2.shape[0], 60 : 60 + en_crop_2.shape[1]] = en_crop_2
    s3[200 : 200 + h_crop_1.shape[0], 60 : 60 + h_crop_1.shape[1]] = h_crop_1
    s3[320 : 320 + en_crop_1.shape[0], 60 : 60 + en_crop_1.shape[1]] = en_crop_1
    scenes["mixed_bilingual"] = {
        "image": s3,
        "description": "Mixed bilingual document with English header, Hindi body, and English footer",
        "expected_min_lines": 3,
    }

    # Scene 4: Dense Multi-Line Document
    s4 = np.ones((720, 960, 3), dtype=np.uint8) * 255
    y_offsets = [60, 160, 260, 360, 460]
    crops = [en_crop_3, h_crop_2, en_crop_1, h_crop_1, en_crop_2]
    for y_off, crop in zip(y_offsets, crops):
        s4[y_off : y_off + crop.shape[0], 80 : 80 + crop.shape[1]] = crop
    scenes["dense_multiline"] = {
        "image": s4,
        "description": "Dense 5-line mixed receipt / document layout",
        "expected_min_lines": 5,
    }

    # Scene 5: Narrow Spacing (Nearby lines 15px apart)
    s5 = np.ones((300, 600, 3), dtype=np.uint8) * 255
    s5[50 : 50 + en_crop_1.shape[0], 40 : 40 + en_crop_1.shape[1]] = en_crop_1
    s5[115 : 115 + en_crop_2.shape[0], 40 : 40 + en_crop_2.shape[1]] = en_crop_2
    scenes["nearby_lines"] = {
        "image": s5,
        "description": "Two nearby lines with tight vertical gap (15px gap)",
        "expected_min_lines": 2,
    }

    return scenes


def run_latency_benchmark(
    detector: DBNetDetector,
    provider_name: str,
    resolutions: List[Tuple[int, int]],
    warmup_runs: int = 10,
    timed_runs: int = 30,
) -> List[Dict[str, Any]]:
    """Measure latency and breakdown across resolutions."""
    results = []
    logger.info("Starting latency benchmark for Provider [%s]...", provider_name)

    for h, w in resolutions:
        # Create representative test canvas
        dummy_img = np.ones((h, w, 3), dtype=np.uint8) * 240
        # Draw some synthetic lines to trigger realistic postprocessing
        for y in range(80, min(h - 80, 800), 120):
            cv2.putText(
                dummy_img,
                "NAYRA BENCHMARK LINE 12345",
                (60, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (20, 20, 20),
                2,
            )

        # Warmup
        for _ in range(warmup_runs):
            _ = detector.detect(dummy_img)

        # Reset memory tracking if CUDA
        if provider_name == "CUDA":
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        pre_times = []
        infer_times = []
        post_times = []
        total_times = []
        detected_counts = []

        for _ in range(timed_runs):
            res = detector.detect(dummy_img)
            pre_times.append(res.pre_time_ms)
            infer_times.append(res.infer_time_ms)
            post_times.append(res.post_time_ms)
            total_times.append(res.total_time_ms)
            detected_counts.append(res.num_detections)

        peak_gpu_mb = 0.0
        if provider_name == "CUDA":
            try:
                import torch

                if torch.cuda.is_available():
                    peak_gpu_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                pass

        entry = {
            "provider": provider_name,
            "active_provider": detector.active_provider,
            "resolution": f"{w}x{h}",
            "width": w,
            "height": h,
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "detections_mean": float(np.mean(detected_counts)),
            "pre_median_ms": round(float(np.median(pre_times)), 2),
            "pre_p95_ms": round(float(np.percentile(pre_times, 95)), 2),
            "infer_median_ms": round(float(np.median(infer_times)), 2),
            "infer_p95_ms": round(float(np.percentile(infer_times, 95)), 2),
            "post_median_ms": round(float(np.median(post_times)), 2),
            "post_p95_ms": round(float(np.percentile(post_times, 95)), 2),
            "total_median_ms": round(float(np.median(total_times)), 2),
            "total_p95_ms": round(float(np.percentile(total_times, 95)), 2),
            "peak_gpu_vram_mb": round(peak_gpu_mb, 2),
        }
        results.append(entry)
        logger.info(
            "Res %dx%d (%s) -> Total Median: %.2f ms (Infer: %.2f ms, Pre: %.2f ms, Post: %.2f ms) | P95: %.2f ms",
            w,
            h,
            provider_name,
            entry["total_median_ms"],
            entry["infer_median_ms"],
            entry["pre_median_ms"],
            entry["post_median_ms"],
            entry["total_p95_ms"],
        )

    return results


def evaluate_scenes(detector: DBNetDetector) -> List[Dict[str, Any]]:
    """Evaluate detection accuracy and bounding coordinates on real scene images."""
    scenes = create_eval_scenes()
    results = []

    for name, data in scenes.items():
        img = data["image"]
        res = detector.detect(img)

        # Check detected boxes
        boxes = res.boxes_xyxy
        confs = res.confidences
        passed = (res.num_detections >= data["expected_min_lines"]) and res.success

        entry = {
            "scene_name": name,
            "description": data["description"],
            "image_shape": [int(img.shape[0]), int(img.shape[1])],
            "expected_min_lines": data["expected_min_lines"],
            "detected_lines": res.num_detections,
            "passed": passed,
            "confidences": [round(c, 3) for c in confs],
            "boxes_xyxy": [[round(v, 1) for v in b] for b in boxes],
            "latency_ms": round(res.total_time_ms, 2),
        }
        results.append(entry)
        logger.info(
            "Scene [%s]: Detected %d / >=%d lines (Passed: %s) in %.2f ms",
            name,
            res.num_detections,
            data["expected_min_lines"],
            passed,
            res.total_time_ms,
        )

    return results


def main() -> None:
    logger.info("Initializing DBNetDetector for benchmark...")
    configure_cuda_dlls()

    # Benchmark configurations
    resolutions = [(640, 640), (960, 960), (1080, 1920)]

    # 1. CUDA Detector
    cuda_detector = DBNetDetector(preferred_provider="CUDA")
    cuda_benchmarks = run_latency_benchmark(
        detector=cuda_detector,
        provider_name="CUDA",
        resolutions=resolutions,
        warmup_runs=10,
        timed_runs=30,
    )

    # 2. Real scene evaluation
    scene_eval = evaluate_scenes(cuda_detector)

    # 3. CPU Detector
    cpu_detector = DBNetDetector(preferred_provider="CPU")
    cpu_benchmarks = run_latency_benchmark(
        detector=cpu_detector,
        provider_name="CPU",
        resolutions=resolutions,
        warmup_runs=5,
        timed_runs=15,  # Fewer timed runs on CPU due to 1.5s+ latency per frame
    )

    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(cuda_detector.model_path),
        "model_sha256": "64f2f901ba4ea646f448c7df57f299bdda3e74718a3ef585d65e29eca2b7a1a2",
        "execution_providers_available": ort.get_available_providers(),
        "cuda_benchmarks": cuda_benchmarks,
        "cpu_benchmarks": cpu_benchmarks,
        "scene_evaluations": scene_eval,
    }

    report_path = Path("07_ocr/detector_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Saved complete detection benchmark report to: %s", report_path)


if __name__ == "__main__":
    import onnxruntime as ort

    main()
