"""NAYRA Module 07 — Comprehensive Unified OCR Pipeline Benchmark & Evaluation.

Evaluates the integrated end-to-end pipeline across:
1. Hindi-only documents
2. English-only documents
3. Mixed bilingual documents (Hindi + English)
4. Multi-line receipts and notices
5. Pure numeric / alphanumeric codes
6. Standalone PoC bitwise equivalence validation
7. End-to-end latency breakdown (Detection, Routing, Recognition, Total)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np

# Add 07_ocr to sys.path
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from pipeline.ocr_engine import OCREngine
from poc_english_ocr import EnglishOCRModel
from poc_hindi_ocr import HindiOCRModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineBenchmark")


def load_sample_crops() -> Dict[str, Dict[str, Any]]:
    """Load test crops from data directories."""
    crops = {}
    hindi_dir = Path("07_ocr/data/poc")
    en_dir = Path("07_ocr/data/poc_en")

    # Load Hindi ground truth
    with open(hindi_dir / "ground_truth.json", "r", encoding="utf-8") as f:
        h_gt = json.load(f)
    for item in h_gt:
        img_path = Path(item["image_path"])
        img = cv2.imread(str(img_path))
        if img is not None:
            crops[f"hindi_{item['id']}"] = {"image": img, "text": item["ground_truth"], "script": "DEVANAGARI"}

    # Load English ground truth
    with open(en_dir / "ground_truth.json", "r", encoding="utf-8") as f:
        en_gt = json.load(f)
    for item in en_gt:
        img_path = Path(item["image_path"])
        img = cv2.imread(str(img_path))
        if img is not None:
            crops[f"english_{item['id']}"] = {"image": img, "text": item["ground_truth"], "script": "LATIN"}

    return crops


def create_test_documents(crops: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Synthesize structured document layouts using real text crops."""
    docs = {}

    h1 = crops["hindi_sample_01"]["image"]  # "नमस्ते दुनिया"
    h2 = crops["hindi_sample_02"]["image"]  # "भारत एक महान देश है"
    en1 = crops["english_sample_en_01"]["image"]  # "EMERGENCY EXIT"
    en2 = crops["english_sample_en_02"]["image"]  # "NAYRA Assistive Vision"
    en3 = crops["english_sample_en_03"]["image"]  # "please keep this area clear."

    # Document 1: Pure Hindi Notice
    d1 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    d1[60 : 60 + h1.shape[0], 50 : 50 + h1.shape[1]] = h1
    d1[180 : 180 + h2.shape[0], 50 : 50 + h2.shape[1]] = h2
    docs["doc_hindi_pure"] = {
        "image": d1,
        "description": "Pure Hindi two-line document",
        "expected_lang": "HINDI",
        "ground_truth_lines": [crops["hindi_sample_01"]["text"], crops["hindi_sample_02"]["text"]],
    }

    # Document 2: Pure English Sign
    d2 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    d2[60 : 60 + en1.shape[0], 50 : 50 + en1.shape[1]] = en1
    d2[160 : 160 + en2.shape[0], 50 : 50 + en2.shape[1]] = en2
    d2[260 : 260 + en3.shape[0], 50 : 50 + en3.shape[1]] = en3
    docs["doc_english_pure"] = {
        "image": d2,
        "description": "Pure English 3-line office sign",
        "expected_lang": "ENGLISH",
        "ground_truth_lines": [
            crops["english_sample_en_01"]["text"],
            crops["english_sample_en_02"]["text"],
            crops["english_sample_en_03"]["text"],
        ],
    }

    # Document 3: Mixed Bilingual Notice
    d3 = np.ones((500, 700, 3), dtype=np.uint8) * 255
    d3[60 : 60 + en1.shape[0], 60 : 60 + en1.shape[1]] = en1
    d3[180 : 180 + h1.shape[0], 60 : 60 + h1.shape[1]] = h1
    d3[300 : 300 + en2.shape[0], 60 : 60 + en2.shape[1]] = en2
    docs["doc_bilingual_mixed"] = {
        "image": d3,
        "description": "Mixed bilingual notice (English Header + Hindi Body + English Footer)",
        "expected_lang": "MIXED",
        "ground_truth_lines": [
            crops["english_sample_en_01"]["text"],
            crops["hindi_sample_01"]["text"],
            crops["english_sample_en_02"]["text"],
        ],
    }

    # Document 4: Alphanumeric Receipt
    d4 = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(d4, "RECEIPT NO: 98214", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(d4, "TOTAL: $49.99", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    docs["doc_receipt_numeric"] = {
        "image": d4,
        "description": "Receipt with alphanumeric codes and prices",
        "expected_lang": "ENGLISH",
        "ground_truth_lines": ["RECEIPT NO: 98214", "TOTAL: $49.99"],
    }

    return docs


def run_benchmark(engine: OCREngine, docs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Execute end-to-end benchmark on all documents."""
    doc_results = []
    latencies = {
        "detection_ms": [],
        "routing_ms": [],
        "recognition_ms": [],
        "total_ms": [],
    }

    for name, data in docs.items():
        img = data["image"]
        # Warmup pass
        _ = engine.process_image(img)

        # Timed passes (5 iterations per document)
        runs = []
        for _ in range(5):
            t_res = engine.process_image(img)
            runs.append(t_res)

        # Pick median run
        runs.sort(key=lambda r: r.total_time_ms)
        median_res = runs[len(runs) // 2]

        latencies["detection_ms"].append(median_res.detection_time_ms)
        latencies["routing_ms"].append(median_res.routing_time_ms)
        latencies["recognition_ms"].append(median_res.recognition_time_ms)
        latencies["total_ms"].append(median_res.total_time_ms)

        recognized_lines = [r.text for r in median_res.regions if r.status.value == "SUCCESS"]
        scripts = [r.script for r in median_res.regions]

        entry = {
            "document_name": name,
            "description": data["description"],
            "expected_language": data["expected_lang"],
            "predicted_language": median_res.primary_language.value,
            "detection_count": median_res.detection_count,
            "recognition_count": median_res.recognition_count,
            "unresolved_count": median_res.unresolved_count,
            "scripts_detected": scripts,
            "recognized_lines": recognized_lines,
            "combined_text": median_res.combined_text,
            "timings_ms": {
                "detection": round(median_res.detection_time_ms, 2),
                "routing": round(median_res.routing_time_ms, 2),
                "recognition": round(median_res.recognition_time_ms, 2),
                "total": round(median_res.total_time_ms, 2),
            },
        }
        doc_results.append(entry)
        logger.info(
            "Doc [%s]: Lang=%s, Regions=%d, Total=%.2f ms (Det: %.2f ms, Route: %.2f ms, Rec: %.2f ms)",
            name,
            median_res.primary_language.value,
            median_res.recognition_count,
            median_res.total_time_ms,
            median_res.detection_time_ms,
            median_res.routing_time_ms,
            median_res.recognition_time_ms,
        )

    summary = {
        "mean_detection_ms": round(float(np.mean(latencies["detection_ms"])), 2),
        "mean_routing_ms": round(float(np.mean(latencies["routing_ms"])), 2),
        "mean_recognition_ms": round(float(np.mean(latencies["recognition_ms"])), 2),
        "mean_total_ms": round(float(np.mean(latencies["total_ms"])), 2),
        "median_total_ms": round(float(np.median(latencies["total_ms"])), 2),
        "documents_evaluated": len(docs),
    }

    return {"documents": doc_results, "summary": summary}


def verify_standalone_poc_equivalence(engine: OCREngine) -> Dict[str, Any]:
    """Test standalone PoC outputs vs integrated recognizers on all test crops."""
    logger.info("Running bitwise / string equivalence check against standalone PoCs...")
    h_poc = HindiOCRModel()
    en_poc = EnglishOCRModel()

    crops = load_sample_crops()
    checks = []

    for name, data in crops.items():
        img = data["image"]
        if data["script"] == "DEVANAGARI":
            poc_text, poc_conf = h_poc.predict(img)
            eng_text, eng_conf = engine.hindi_model.predict(img)
        else:
            poc_text, poc_conf = en_poc.predict(img)
            eng_text, eng_conf = engine.english_model.predict(img)

        exact_match = (poc_text == eng_text) and (abs(poc_conf - eng_conf) < 1e-4)
        checks.append({
            "sample": name,
            "script": data["script"],
            "poc_text": poc_text,
            "integrated_text": eng_text,
            "exact_match": exact_match,
            "poc_conf": round(float(poc_conf), 4),
            "integrated_conf": round(float(eng_conf), 4),
        })

    all_matched = all(c["exact_match"] for c in checks)
    logger.info("Standalone PoC Equivalence Check: %d / %d samples matched perfectly (All Matched: %s)",
                sum(1 for c in checks if c["exact_match"]), len(checks), all_matched)

    return {
        "all_matched": all_matched,
        "total_samples": len(checks),
        "matched_samples": sum(1 for c in checks if c["exact_match"]),
        "sample_details": checks,
    }


def main():
    logger.info("Initializing OCREngine for full benchmark...")
    engine = OCREngine(preferred_provider="CUDA")

    # 1. PoC Equivalence Check
    poc_equiv = verify_standalone_poc_equivalence(engine)

    # 2. Document Benchmarks
    crops = load_sample_crops()
    docs = create_test_documents(crops)
    benchmark_res = run_benchmark(engine, docs)

    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "active_execution_providers": engine.execution_providers,
        "standalone_poc_equivalence": poc_equiv,
        "benchmark_summary": benchmark_res["summary"],
        "document_results": benchmark_res["documents"],
    }

    out_path = Path("07_ocr/pipeline_benchmark_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    logger.info("Saved pipeline benchmark report to: %s", out_path)


if __name__ == "__main__":
    main()
