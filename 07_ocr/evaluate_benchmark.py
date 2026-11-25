"""Evaluation script for NAYRA Module 07 Hindi OCR Proof-of-Concept.

Executes:
- Ground truth verification on 10 Hindi test-line crops
- Character Error Rate (CER) calculation via Levenshtein distance
- Confidence score reporting
- Exact CPU latency and CUDA latency (RTX 4050) with warmup and 50 iterations
- Node provider verification
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import POC model directly from file path
import importlib.util
poc_spec = importlib.util.spec_from_file_location("poc_hindi_ocr", Path(__file__).parent / "poc_hindi_ocr.py")
poc_module = importlib.util.module_from_spec(poc_spec)
poc_spec.loader.exec_module(poc_module)

HindiOCRModel = poc_module.HindiOCRModel
compute_cer = poc_module.compute_cer
levenshtein_distance = poc_module.levenshtein_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")


def run_benchmark():
    gt_file = Path("07_ocr/data/poc/ground_truth.json")
    if not gt_file.exists():
        logger.error(f"Ground truth catalog not found: {gt_file}")
        sys.exit(1)

    with open(gt_file, "r", encoding="utf-8") as f:
        test_cases: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(test_cases)} test cases from {gt_file}")

    # Initialize CUDA Model
    logger.info("Initializing CUDA model on NVIDIA RTX 4050...")
    cuda_model = HindiOCRModel(use_cuda=True)
    cuda_providers = cuda_model.active_providers

    # Initialize CPU Model
    logger.info("Initializing CPU model...")
    cpu_model = HindiOCRModel(use_cuda=False)
    cpu_providers = cpu_model.active_providers

    results = []

    # Warmup runs
    dummy_img = cv2.imread(test_cases[0]["image_path"])
    logger.info("Warming up CUDA model (15 passes)...")
    cuda_model.benchmark_latency(dummy_img, warmup=15, iterations=5)
    logger.info("Warming up CPU model (15 passes)...")
    cpu_model.benchmark_latency(dummy_img, warmup=15, iterations=5)

    print("\n" + "=" * 95)
    print(f"{'ID':<10} | {'Ground Truth':<28} | {'Recognized (CUDA)':<28} | {'CER':<6} | {'Conf':<6}")
    print("=" * 95)

    for tc in test_cases:
        img_path = tc["image_path"]
        gt = tc["ground_truth"]
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Could not load image: {img_path}")
            continue

        # Predictions
        pred_cuda, conf_cuda = cuda_model.predict(img)
        pred_cpu, conf_cpu = cpu_model.predict(img)

        # CER calculation
        cer_cuda = compute_cer(gt, pred_cuda)
        cer_cpu = compute_cer(gt, pred_cpu)
        edit_dist = levenshtein_distance(gt, pred_cuda)

        # Latency measurements (50 iterations)
        bench_cuda = cuda_model.benchmark_latency(img, warmup=5, iterations=50)
        bench_cpu = cpu_model.benchmark_latency(img, warmup=5, iterations=50)

        record = {
            "id": tc["id"],
            "category": tc["category"],
            "image_path": img_path,
            "ground_truth": gt,
            "predicted_cuda": pred_cuda,
            "predicted_cpu": pred_cpu,
            "edit_distance": edit_dist,
            "cer": round(cer_cuda, 4),
            "confidence": round(conf_cuda, 4),
            "cuda_latency_median_ms": bench_cuda["median_ms"],
            "cuda_latency_p95_ms": bench_cuda["p95_ms"],
            "cpu_latency_median_ms": bench_cpu["median_ms"],
            "cpu_latency_p95_ms": bench_cpu["p95_ms"],
            "exact_match": gt == pred_cuda,
        }
        results.append(record)

        print(f"{tc['id']:<10} | {gt:<28} | {pred_cuda:<28} | {cer_cuda:<6.2%} | {conf_cuda:<6.2f}")

    print("=" * 95 + "\n")

    # Aggregate metrics
    mean_cer = np.mean([r["cer"] for r in results])
    exact_matches = sum(1 for r in results if r["exact_match"])
    accuracy = exact_matches / len(results)
    mean_conf = np.mean([r["confidence"] for r in results])

    cuda_medians = [r["cuda_latency_median_ms"] for r in results]
    cuda_p95s = [r["cuda_latency_p95_ms"] for r in results]
    cpu_medians = [r["cpu_latency_median_ms"] for r in results]
    cpu_p95s = [r["cpu_latency_p95_ms"] for r in results]

    overall_cuda_median = round(float(np.mean(cuda_medians)), 2)
    overall_cuda_p95 = round(float(np.mean(cuda_p95s)), 2)
    overall_cpu_median = round(float(np.mean(cpu_medians)), 2)
    overall_cpu_p95 = round(float(np.mean(cpu_p95s)), 2)
    speedup = round(overall_cpu_median / overall_cuda_median, 2) if overall_cuda_median > 0 else 0.0

    summary = {
        "model_name": "devanagari_PP-OCRv5_mobile_rec.onnx",
        "cuda_providers": cuda_providers,
        "cpu_providers": cpu_providers,
        "total_samples": len(results),
        "exact_matches": exact_matches,
        "accuracy": round(accuracy, 4),
        "mean_cer": round(float(mean_cer), 4),
        "mean_confidence": round(float(mean_conf), 4),
        "overall_cuda_latency_median_ms": overall_cuda_median,
        "overall_cuda_latency_p95_ms": overall_cuda_p95,
        "overall_cpu_latency_median_ms": overall_cpu_median,
        "overall_cpu_latency_p95_ms": overall_cpu_p95,
        "cuda_speedup_factor": speedup,
        "sample_results": results,
    }

    report_path = Path("07_ocr/poc_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("--- BENCHMARK SUMMARY ---")
    print(f"Total Test Samples    : {len(results)}")
    print(f"Exact String Matches  : {exact_matches}/{len(results)} ({accuracy:.1%})")
    print(f"Mean Character Err Rate: {mean_cer:.2%}")
    print(f"Mean Confidence Score : {mean_conf:.4f}")
    print(f"Active CUDA Providers : {cuda_providers}")
    print(f"Active CPU Providers  : {cpu_providers}")
    print(f"CUDA Latency (Median) : {overall_cuda_median} ms")
    print(f"CUDA Latency (P95)    : {overall_cuda_p95} ms")
    print(f"CPU Latency (Median)  : {overall_cpu_median} ms")
    print(f"CPU Latency (P95)     : {overall_cpu_p95} ms")
    print(f"CUDA Speedup Factor   : {speedup}x")
    print(f"Full report saved to  : {report_path}")


if __name__ == "__main__":
    run_benchmark()
