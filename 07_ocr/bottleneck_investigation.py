"""NAYRA Module 07 — Recognition Bottleneck Investigation & Batching Benchmark.

Investigates:
1. Sequential vs Batched ONNX Runtime inference for Hindi and English recognizers.
2. Dynamic tensor dimension support for batch size and width: [B, 3, 48, max_w].
3. Quality parity: Exact match rate, character-for-character comparison, confidence delta.
4. Latency scaling across 1, 2, 4, 8 text crops.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

import importlib

# Ensure root directory is on sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Import dynamically to support numeric folder naming
_poc_hi = importlib.import_module("07_ocr.poc_hindi_ocr")
_poc_en = importlib.import_module("07_ocr.poc_english_ocr")
ctc_decode_hi = _poc_hi.ctc_decode
ctc_decode_en = _poc_en.ctc_decode

from ocr import get_ocr_engine


def batch_predict(model: Any, crops: List[np.ndarray], is_hindi: bool = True) -> List[Tuple[str, float]]:
    """Execute batched inference over multiple text crops using dynamic width padding.

    Args:
        model: Initialized HindiOCRModel or EnglishOCRModel.
        crops: List of OpenCV BGR uint8 crop images.
        is_hindi: True for Hindi Devanagari, False for English Latin.

    Returns:
        List of (recognized_text, confidence_score) tuples.
    """
    if not crops:
        return []

    target_h = 48
    min_w = 320
    ratios: List[float] = []
    pad_ws: List[Tuple[int, int]] = []

    # 1. Determine scaled dimensions maintaining aspect ratios
    for c in crops:
        h, w = c.shape[:2]
        ratio = w / float(h)
        resized_w = int(round(target_h * ratio))
        pad_w = max(min_w, int(np.ceil(resized_w / 32) * 32))
        pad_ws.append((resized_w, pad_w))

    max_pad_w = max(p[1] for p in pad_ws)

    # 2. Pack batch into single tensor (B, 3, 48, max_pad_w)
    batch_tensor = np.zeros((len(crops), 3, target_h, max_pad_w), dtype=np.float32)
    for i, c in enumerate(crops):
        resized_w, _ = pad_ws[i]
        resized = cv2.resize(c, (resized_w, target_h), interpolation=cv2.INTER_LINEAR)
        norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        batch_tensor[i, :, :, :resized_w] = norm.transpose((2, 0, 1))
        ratios.append(min(1.0, resized_w / float(max_pad_w)))

    input_name = model.session.get_inputs()[0].name
    preds = model.session.run(None, {input_name: batch_tensor})[0]  # (B, T, num_classes)

    # 3. Decode each batch element with CTC
    decode_fn = ctc_decode_hi if is_hindi else ctc_decode_en
    results: List[Tuple[str, float]] = []
    char_list = getattr(model, "character_list", getattr(model, "character_dict", None))
    for b in range(len(crops)):
        text, conf, _ = decode_fn(preds[b], char_list, valid_ratio=ratios[b])
        results.append((text, float(conf)))

    return results


def run_investigation() -> Dict[str, Any]:
    print("======================================================================")
    print("   NAYRA Module 07 — Recognition Bottleneck Investigation (Phase 5)   ")
    print("======================================================================")

    engine = get_ocr_engine(preferred_provider="CUDA")

    # Load authentic Hindi and English test crops
    hi_crops = [cv2.imread(f"07_ocr/data/poc/sample_0{i}.png") for i in range(1, 9)]
    en_crops = [cv2.imread(f"07_ocr/data/poc_en/sample_en_{i:02d}.png") for i in range(1, 9)]

    # Warm up sessions
    engine.hindi_model.predict(hi_crops[0])
    engine.english_model.predict(en_crops[0])
    batch_predict(engine.hindi_model, hi_crops[:2], is_hindi=True)
    batch_predict(engine.english_model, en_crops[:2], is_hindi=False)

    results_report: Dict[str, Any] = {
        "device": "NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)",
        "hardware_provider": "CUDAExecutionProvider",
        "tests": {},
    }

    # Test Hindi scaling (1, 2, 4, 8 crops)
    for batch_size in [1, 2, 4, 8]:
        sub_crops = hi_crops[:batch_size]

        # Sequential timing
        t0 = time.perf_counter()
        for _ in range(15):
            seq_out = [engine.hindi_model.predict(c) for c in sub_crops]
        seq_ms = (time.perf_counter() - t0) * 1000.0 / 15.0

        # Batched timing
        t0 = time.perf_counter()
        for _ in range(15):
            batch_out = batch_predict(engine.hindi_model, sub_crops, is_hindi=True)
        batch_ms = (time.perf_counter() - t0) * 1000.0 / 15.0

        # Quality check
        matches = [s[0] == b[0] for s, b in zip(seq_out, batch_out)]
        exact_match = all(matches)
        conf_deltas = [abs(s[1] - b[1]) for s, b in zip(seq_out, batch_out)]
        max_conf_delta = max(conf_deltas)

        speedup = seq_ms / max(batch_ms, 0.001)

        print(f"\n[Hindi Devanagari - Batch Size {batch_size}]")
        print(f"  Sequential Latency : {seq_ms:6.2f} ms ({seq_ms/batch_size:5.2f} ms/crop)")
        print(f"  Batched Latency    : {batch_ms:6.2f} ms ({batch_ms/batch_size:5.2f} ms/crop)")
        print(f"  Speedup Factor     : {speedup:6.2f}x faster")
        print(f"  Exact Text Match   : {exact_match} ({sum(matches)}/{batch_size})")
        print(f"  Max Conf Delta     : {max_conf_delta:.6f}")

        results_report["tests"][f"hindi_b{batch_size}"] = {
            "batch_size": batch_size,
            "sequential_ms": seq_ms,
            "batched_ms": batch_ms,
            "speedup": speedup,
            "exact_match": exact_match,
            "max_conf_delta": max_conf_delta,
        }

    # Test English scaling (1, 2, 4, 8 crops)
    for batch_size in [1, 2, 4, 8]:
        sub_crops = en_crops[:batch_size]

        # Sequential timing
        t0 = time.perf_counter()
        for _ in range(15):
            seq_out = [engine.english_model.predict(c) for c in sub_crops]
        seq_ms = (time.perf_counter() - t0) * 1000.0 / 15.0

        # Batched timing
        t0 = time.perf_counter()
        for _ in range(15):
            batch_out = batch_predict(engine.english_model, sub_crops, is_hindi=False)
        batch_ms = (time.perf_counter() - t0) * 1000.0 / 15.0

        # Quality check
        matches = [s[0] == b[0] for s, b in zip(seq_out, batch_out)]
        exact_match = all(matches)
        conf_deltas = [abs(s[1] - b[1]) for s, b in zip(seq_out, batch_out)]
        max_conf_delta = max(conf_deltas)

        speedup = seq_ms / max(batch_ms, 0.001)

        print(f"\n[English Latin - Batch Size {batch_size}]")
        print(f"  Sequential Latency : {seq_ms:6.2f} ms ({seq_ms/batch_size:5.2f} ms/crop)")
        print(f"  Batched Latency    : {batch_ms:6.2f} ms ({batch_ms/batch_size:5.2f} ms/crop)")
        print(f"  Speedup Factor     : {speedup:6.2f}x faster")
        print(f"  Exact Text Match   : {exact_match} ({sum(matches)}/{batch_size})")
        print(f"  Max Conf Delta     : {max_conf_delta:.6f}")

        results_report["tests"][f"english_b{batch_size}"] = {
            "batch_size": batch_size,
            "sequential_ms": seq_ms,
            "batched_ms": batch_ms,
            "speedup": speedup,
            "exact_match": exact_match,
            "max_conf_delta": max_conf_delta,
        }

    out_file = _ROOT_DIR / "07_ocr" / "recognition_bottleneck_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2)
    print(f"\nFull report written to: {out_file}")
    return results_report


if __name__ == "__main__":
    run_investigation()
