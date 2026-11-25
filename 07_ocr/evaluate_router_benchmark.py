"""Evaluation script for NAYRA Module 07 Standalone Script Router.

Evaluates:
1. 25-Sample Benchmark:
   - 10 Hindi Devanagari crops (07_ocr/data/poc/)
   - 15 English Latin crops (07_ocr/data/poc_en/)
2. Separate Robustness Test Suite:
   - Pure numeric crops
   - Mixed digits & punctuation
   - Empty crops
   - Invalid & flat inputs
   - Very small crops (<8px)
   - Gaussian blurred text
   - Solid horizontal non-text lines

Reports:
- Routing accuracy on benchmark samples
- Confusion matrix
- Per-class Precision, Recall, and F1 score
- UNKNOWN and SHARED_NUMERIC decision counts
- Execution latency (median, P95) across 50 iterations
- Separate robustness test summary
- Saves full JSON report to 07_ocr/router_benchmark_report.json
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add 07_ocr directory to sys.path for clean local package imports
ocr_dir = Path(__file__).resolve().parent
if str(ocr_dir) not in sys.path:
    sys.path.insert(0, str(ocr_dir))

from router.script_router import ScriptRouter, ScriptType, RouterResult
from router.shirorekha_filter import ShirorekhaFilter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RouterBenchmark")


def measure_latency(router: ScriptRouter, img: np.ndarray, iterations: int = 50) -> Dict[str, float]:
    """Measure single-crop routing latency in milliseconds."""
    # Warmup
    for _ in range(10):
        _ = router.route(img)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = router.route(img)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times.sort()
    return {
        "median_ms": round(float(np.median(times)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "min_ms": round(float(np.min(times)), 3),
        "max_ms": round(float(np.max(times)), 3),
    }


def render_synthetic_crop(text: str, font, pad_x: int = 15, pad_y: int = 10) -> np.ndarray:
    """Helper to render synthetic text crop."""
    dummy = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    w = (bbox[2] - bbox[0]) + pad_x * 2
    h = max(36, (bbox[3] - bbox[1]) + pad_y * 2)
    im = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def run_benchmark():
    router = ScriptRouter()

    # Load 10 Hindi and 15 English benchmark catalogs
    hindi_gt_path = Path("07_ocr/data/poc/ground_truth.json")
    en_gt_path = Path("07_ocr/data/poc_en/ground_truth.json")

    if not hindi_gt_path.exists() or not en_gt_path.exists():
        logger.error("Benchmark ground truth files missing.")
        return

    with open(hindi_gt_path, "r", encoding="utf-8") as f:
        hindi_samples = json.load(f)
    with open(en_gt_path, "r", encoding="utf-8") as f:
        en_samples = json.load(f)

    logger.info(f"Loaded {len(hindi_samples)} Hindi samples and {len(en_samples)} English samples.")

    # Benchmark dataset evaluation
    benchmark_results = []
    classes = [ScriptType.DEVANAGARI.value, ScriptType.LATIN.value, ScriptType.SHARED_NUMERIC.value, ScriptType.UNKNOWN.value]
    confusion_matrix = {true_c: {pred_c: 0 for pred_c in classes} for true_c in classes}

    print("\n" + "=" * 105)
    print(f"{'ID':<14} | {'Ground Truth':<30} | {'Expected':<12} | {'Predicted':<14} | {'Conf':<6} | {'Status'}")
    print("=" * 105)

    all_latencies = []

    # 1. Hindi samples (Expected: DEVANAGARI)
    for s in hindi_samples:
        img = cv2.imread(s["image_path"])
        res = router.route(img)
        bench = measure_latency(router, img, iterations=50)
        all_latencies.append(bench["median_ms"])

        expected = ScriptType.DEVANAGARI.value
        predicted = res.script.value
        is_correct = (expected == predicted)
        confusion_matrix[expected][predicted] += 1

        record = {
            "id": s["id"],
            "category": s.get("category", ""),
            "ground_truth": s["ground_truth"],
            "image_path": s["image_path"],
            "expected_script": expected,
            "predicted_script": predicted,
            "confidence": res.confidence,
            "is_correct": is_correct,
            "decision_reason": res.decision_reason,
            "latency_median_ms": bench["median_ms"],
            "diagnostics": res.diagnostics,
        }
        benchmark_results.append(record)
        status_str = "PASS" if is_correct else "FAIL"
        print(f"{s['id']:<14} | {s['ground_truth']:<30} | {expected:<12} | {predicted:<14} | {res.confidence:<6.2f} | {status_str}")

    # 2. English samples (Expected: LATIN)
    for s in en_samples:
        img = cv2.imread(s["image_path"])
        res = router.route(img)
        bench = measure_latency(router, img, iterations=50)
        all_latencies.append(bench["median_ms"])

        expected = ScriptType.LATIN.value
        predicted = res.script.value
        is_correct = (expected == predicted)
        confusion_matrix[expected][predicted] += 1

        record = {
            "id": s["id"],
            "category": s.get("category", ""),
            "ground_truth": s["ground_truth"],
            "image_path": s["image_path"],
            "expected_script": expected,
            "predicted_script": predicted,
            "confidence": res.confidence,
            "is_correct": is_correct,
            "decision_reason": res.decision_reason,
            "latency_median_ms": bench["median_ms"],
            "diagnostics": res.diagnostics,
        }
        benchmark_results.append(record)
        status_str = "PASS" if is_correct else "FAIL"
        print(f"{s['id']:<14} | {s['ground_truth']:<30} | {expected:<12} | {predicted:<14} | {res.confidence:<6.2f} | {status_str}")

    print("=" * 105 + "\n")

    # Compute Benchmark Metrics
    total_benchmark = len(benchmark_results)
    correct_count = sum(1 for r in benchmark_results if r["is_correct"])
    benchmark_accuracy = correct_count / float(total_benchmark) if total_benchmark > 0 else 0.0

    # Per-class metrics
    per_class_metrics = {}
    for c in [ScriptType.DEVANAGARI.value, ScriptType.LATIN.value]:
        tp = confusion_matrix[c][c]
        fp = sum(confusion_matrix[other][c] for other in classes if other != c)
        fn = sum(confusion_matrix[c][other] for other in classes if other != c)
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / float(prec + rec) if (prec + rec) > 0 else 0.0
        per_class_metrics[c] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    overall_median_lat = round(float(np.median(all_latencies)), 3)
    overall_p95_lat = round(float(np.percentile(all_latencies, 95)), 3)

    # 3. Robustness Tests (Separated from Benchmark metrics)
    logger.info("Executing separate Robustness & Edge-Case Test Suite...")
    try:
        font_arial = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 36)
    except IOError:
        font_arial = ImageFont.load_default()

    robustness_cases = []

    # Case A: Numeric-only crop
    im_num = render_synthetic_crop("12345", font_arial)
    res_num = router.route(im_num)
    robustness_cases.append({
        "case_id": "rob_numeric_pure",
        "description": "Pure numeric string '12345'",
        "expected_behavior": "SHARED_NUMERIC",
        "actual_decision": res_num.script.value,
        "confidence": res_num.confidence,
        "passed": res_num.script == ScriptType.SHARED_NUMERIC,
        "reason": res_num.decision_reason,
    })

    # Case B: Mixed digits and punctuation
    im_phone = render_synthetic_crop("+1 (800) 555-0199", font_arial)
    res_phone = router.route(im_phone)
    robustness_cases.append({
        "case_id": "rob_numeric_symbols",
        "description": "Phone number with symbols '+1 (800) 555-0199'",
        "expected_behavior": "SHARED_NUMERIC or LATIN with symbols",
        "actual_decision": res_phone.script.value,
        "confidence": res_phone.confidence,
        "passed": res_phone.script in [ScriptType.SHARED_NUMERIC, ScriptType.LATIN],
        "reason": res_phone.decision_reason,
    })

    # Case C: Empty crop
    empty_im = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = router.route(empty_im)
    robustness_cases.append({
        "case_id": "rob_empty_array",
        "description": "Empty 0x0 image crop",
        "expected_behavior": "UNKNOWN (conf=0.0)",
        "actual_decision": res_empty.script.value,
        "confidence": res_empty.confidence,
        "passed": res_empty.script == ScriptType.UNKNOWN and res_empty.confidence == 0.0,
        "reason": res_empty.decision_reason,
    })

    # Case D: Invalid flat input
    flat_im = np.ones((50, 100, 3), dtype=np.uint8) * 128
    res_flat = router.route(flat_im)
    robustness_cases.append({
        "case_id": "rob_flat_image",
        "description": "Flat solid gray image (std=0)",
        "expected_behavior": "UNKNOWN (conf=0.0)",
        "actual_decision": res_flat.script.value,
        "confidence": res_flat.confidence,
        "passed": res_flat.script == ScriptType.UNKNOWN and res_flat.confidence == 0.0,
        "reason": res_flat.decision_reason,
    })

    # Case E: Very small crop (<8px)
    tiny_im = np.ones((5, 5, 3), dtype=np.uint8) * 255
    res_tiny = router.route(tiny_im)
    robustness_cases.append({
        "case_id": "rob_tiny_crop",
        "description": "Sub-threshold tiny crop (5x5 px)",
        "expected_behavior": "UNKNOWN (conf=0.0)",
        "actual_decision": res_tiny.script.value,
        "confidence": res_tiny.confidence,
        "passed": res_tiny.script == ScriptType.UNKNOWN and res_tiny.confidence == 0.0,
        "reason": res_tiny.decision_reason,
    })

    # Case F: Blurred text
    im_clean = render_synthetic_crop("Blurred Sign", font_arial)
    im_blurred = cv2.GaussianBlur(im_clean, (31, 31), 15.0)
    res_blur = router.route(im_blurred)
    robustness_cases.append({
        "case_id": "rob_blurred_crop",
        "description": "Heavily Gaussian-blurred text",
        "expected_behavior": "UNKNOWN",
        "actual_decision": res_blur.script.value,
        "confidence": res_blur.confidence,
        "passed": res_blur.script == ScriptType.UNKNOWN,
        "reason": res_blur.decision_reason,
    })

    # Case G: Solid horizontal line non-text artifact
    line_im = np.zeros((8, 120, 3), dtype=np.uint8)
    line_im[3:5, :] = 255
    res_line = router.route(line_im)
    robustness_cases.append({
        "case_id": "rob_horizontal_rule",
        "description": "Solid horizontal line / separator rule",
        "expected_behavior": "UNKNOWN (rejected as non-text)",
        "actual_decision": res_line.script.value,
        "confidence": res_line.confidence,
        "passed": res_line.script == ScriptType.UNKNOWN,
        "reason": res_line.decision_reason,
    })

    # Case H: Single ambiguous glyph
    im_single = render_synthetic_crop("1", font_arial, pad_x=20, pad_y=15)
    res_single = router.route(im_single)
    robustness_cases.append({
        "case_id": "rob_ambiguous_single_glyph",
        "description": "Single isolated glyph '1'",
        "expected_behavior": "UNKNOWN (conservative fallback)",
        "actual_decision": res_single.script.value,
        "confidence": res_single.confidence,
        "passed": res_single.script == ScriptType.UNKNOWN,
        "reason": res_single.decision_reason,
    })

    # Assemble summary
    summary = {
        "metadata": {
            "evaluation_title": "NAYRA Module 07 Script Router Benchmark",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "disclaimer": (
                "PRELIMINARY RESULTS: The evaluated benchmark contains 10 Hindi and 15 English samples. "
                "These figures are preliminary metrics on available development crops and are not "
                "representative of diverse real-world open-world conditions."
            ),
        },
        "benchmark_metrics": {
            "total_benchmark_samples": total_benchmark,
            "correct_classifications": correct_count,
            "incorrect_classifications": total_benchmark - correct_count,
            "routing_accuracy": round(benchmark_accuracy, 4),
            "confusion_matrix": confusion_matrix,
            "per_class_metrics": per_class_metrics,
            "decision_distribution": {
                "DEVANAGARI": sum(1 for r in benchmark_results if r["predicted_script"] == ScriptType.DEVANAGARI.value),
                "LATIN": sum(1 for r in benchmark_results if r["predicted_script"] == ScriptType.LATIN.value),
                "SHARED_NUMERIC": sum(1 for r in benchmark_results if r["predicted_script"] == ScriptType.SHARED_NUMERIC.value),
                "UNKNOWN": sum(1 for r in benchmark_results if r["predicted_script"] == ScriptType.UNKNOWN.value),
            },
            "latency_ms": {
                "overall_median_ms": overall_median_lat,
                "overall_p95_ms": overall_p95_lat,
            },
        },
        "robustness_test_suite": {
            "total_robustness_cases": len(robustness_cases),
            "passed_cases": sum(1 for c in robustness_cases if c["passed"]),
            "cases": robustness_cases,
        },
        "benchmark_sample_details": benchmark_results,
    }

    report_path = Path("07_ocr/router_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n--- SCRIPT ROUTER BENCHMARK SUMMARY ---")
    print(f"Total Benchmark Samples   : {total_benchmark}")
    print(f"Routing Accuracy          : {correct_count}/{total_benchmark} ({benchmark_accuracy:.1%})")
    print(f"Confusion Matrix          :")
    for true_c in [ScriptType.DEVANAGARI.value, ScriptType.LATIN.value]:
        row_str = " ".join([f"{pred_c}={confusion_matrix[true_c][pred_c]}" for pred_c in classes])
        print(f"  True {true_c:<10} -> {row_str}")
    print(f"Devanagari Precision/Recall: {per_class_metrics['DEVANAGARI']['precision']:.2f} / {per_class_metrics['DEVANAGARI']['recall']:.2f} (F1: {per_class_metrics['DEVANAGARI']['f1_score']:.2f})")
    print(f"Latin Precision/Recall     : {per_class_metrics['LATIN']['precision']:.2f} / {per_class_metrics['LATIN']['recall']:.2f} (F1: {per_class_metrics['LATIN']['f1_score']:.2f})")
    print(f"Latency (Median / P95)     : {overall_median_lat} ms / {overall_p95_lat} ms")
    print(f"Robustness Tests Passed    : {sum(1 for c in robustness_cases if c['passed'])}/{len(robustness_cases)}")
    print(f"Full report saved to       : {report_path}")


if __name__ == "__main__":
    run_benchmark()
