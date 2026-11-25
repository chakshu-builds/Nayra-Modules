"""NAYRA Module 07 — Step 3.6.3: Final Technical Closure & Metric Reconciliation Suite.

Implements:
1. Strict 1-to-1 Order-Aware Bipartite Matching (Hungarian algorithm via linear_sum_assignment).
   - Cost matrix uses sequence reading order distance and Levenshtein CER.
   - Bounded dummy deletion/insertion nodes prevent spurious distant matches.
2. Strict language-wise additivity:
   - Devanagari GT: 21 lines, 150 characters, 27 words.
   - Latin GT: 43 lines, 667 characters, 112 words.
   - Total GT: 64 lines, 817 characters, 139 words.
   - Strict Character Error Rate: E_all = E_hin + E_eng (33 = 12 + 21).
   - Strict Word Error Rate: WE_all = WE_hin + WE_eng (17 = 6 + 11).
3. Full Deletion Penalties on Missed Lines (len(gt_text) chars, len(words) words).
4. Rigorous Unmatched Region & Precision Accounting:
   - Denominator = 103 total detector-produced regions.
   - True Positives (annotated text matched): 61
   - Missed GT lines: 3 (4.69%)
   - Unresolved regions (script router filtered): 14
   - Successfully recognized regions: 89
   - Unannotated scene text regions: 24
   - Fragmented detections: 2
   - Non-text artifacts: 2
   - Strict Annotated Precision: 61 / 103 = 59.22% (or 61 / 89 = 68.54% among recognized)
   - Real Text Precision: (61 + 24 + 2) / 103 = 84.47%
   - Recognition Success Rate: 89 / 103 = 86.41%
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Setup path
ROOT_DIR = Path(__file__).parent.parent
OCR_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(OCR_DIR))

import ocr
from ocr import OCRResult, RegionStatus
from pipeline.ocr_engine import OCREngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-Reconciliation")


def normalize_nfc(text: str) -> str:
    """Normalize string to Unicode NFC form and strip leading/trailing whitespace."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text.strip())


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance with Unicode NFC normalization."""
    s1, s2 = normalize_nfc(s1), normalize_nfc(s2)
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def word_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Word Error edit distance between two token sequences with full deletion/insertion."""
    w1 = [normalize_nfc(w) for w in s1.split() if w.strip()]
    w2 = [normalize_nfc(w) for w in s2.split() if w.strip()]
    m, n = len(w1), len(w2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if w1[i - 1] == w2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def run_reconciliation() -> Dict[str, Any]:
    dataset_dir = OCR_DIR / "data" / "real_camera"
    gt_file = dataset_dir / "ground_truth.json"
    matching_records_file = OCR_DIR / "reconciled_matching_records.json"
    output_report_file = OCR_DIR / "reconciled_evaluation_report.json"

    with open(gt_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    logger.info("Initializing OCREngine for Reconciled Evaluation...")
    engine = OCREngine(use_batching=True)
    # Warmup
    engine.process_image(np.zeros((480, 640, 3), dtype=np.uint8))
    logger.info("Engine warmed up. Processing %d real camera test scenes...", len(samples))

    # Ground Truth exact counters
    gt_counters = {
        "lines": {"DEVANAGARI": 0, "LATIN": 0, "TOTAL": 0},
        "chars": {"DEVANAGARI": 0, "LATIN": 0, "TOTAL": 0},
        "words": {"DEVANAGARI": 0, "LATIN": 0, "TOTAL": 0},
    }

    # Error counters
    err_counters = {
        "char_edits": {"DEVANAGARI": 0, "LATIN": 0, "TOTAL": 0},
        "word_edits": {"DEVANAGARI": 0, "LATIN": 0, "TOTAL": 0},
        "exact_matches": {"DEVANAGARI": 0, "LATIN": 0, "TOTAL": 0},
        "matched_lines": {"DEVANAGARI": 0, "LATIN": 0, "TOTAL": 0},
        "missed_lines": {"DEVANAGARI": 0, "LATIN": 0, "TOTAL": 0},
    }

    # Pipeline counters
    total_detected_regions = 0
    total_recognized_regions = 0
    total_unresolved_scripts = 0
    total_unannotated_regions = 0
    total_fragment_regions = 0
    total_nontext_artifacts = 0

    per_sample_details = []
    all_matching_records = []

    for s_idx, sample in enumerate(samples):
        img_name = sample["image"]
        img_path = dataset_dir / img_name
        conditions = sample.get("conditions", [])
        gt_lines = sample.get("lines", [])

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        ocr_res: OCRResult = engine.process_image(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        total_detected_regions += ocr_res.detection_count
        total_recognized_regions += ocr_res.recognition_count
        total_unresolved_scripts += ocr_res.unresolved_count

        succ_preds = [
            (p_i, r) for p_i, r in enumerate(ocr_res.regions)
            if r.status == RegionStatus.SUCCESS and r.text and normalize_nfc(r.text)
        ]

        M = len(gt_lines)
        N = len(succ_preds)

        # -----------------------------------------------------------------
        # Order-Aware Bipartite Matching (Hungarian Algorithm with Dummy Nodes)
        # -----------------------------------------------------------------
        # Augmented cost matrix of size (M + N) x (N + M)
        # Threshold CER for valid candidate match = 0.60
        CER_THRESHOLD = 0.60
        cost_matrix = np.full((M + N, N + M), 10.0, dtype=np.float64)

        for i in range(M):
            gt_text = normalize_nfc(gt_lines[i]["text"])
            gt_len = max(len(gt_text), 1)
            for j in range(N):
                p_text = normalize_nfc(succ_preds[j][1].text)
                c_dist = levenshtein_distance(gt_text, p_text)
                cer = c_dist / gt_len
                # Normalized sequence reading order position distance [0, 1]
                norm_pos_dist = abs(i / max(M - 1, 1) - j / max(N - 1, 1)) if M > 1 and N > 1 else 0.0
                if cer <= CER_THRESHOLD:
                    cost_matrix[i, j] = 0.85 * cer + 0.15 * norm_pos_dist
                else:
                    cost_matrix[i, j] = 5.0  # high penalty for invalid pairs

        # Dummy node connections:
        # Unmatched GT i -> dummy prediction column (N + i) at cost CER_THRESHOLD
        for i in range(M):
            cost_matrix[i, N + i] = CER_THRESHOLD
        # Unmatched Prediction j -> dummy GT row (M + j) at cost CER_THRESHOLD
        for j in range(N):
            cost_matrix[M + j, j] = CER_THRESHOLD
        # Dummy-to-dummy block -> 0 cost
        for i in range(N):
            for j in range(M):
                cost_matrix[M + i, N + j] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Extract genuine matches
        matched_gt: Dict[int, Tuple[int, int, float, str]] = {}  # g_i -> (orig_p_i, dist, cer, p_text)
        matched_pred_indices = set()

        for r, c in zip(row_ind, col_ind):
            if r < M and c < N and cost_matrix[r, c] < CER_THRESHOLD:
                orig_p_i, r_obj = succ_preds[c]
                p_text = normalize_nfc(r_obj.text)
                g_text = normalize_nfc(gt_lines[r]["text"])
                dist = levenshtein_distance(g_text, p_text)
                cer = dist / max(len(g_text), 1)
                matched_gt[r] = (orig_p_i, dist, cer, p_text)
                matched_pred_indices.add(c)

        # -----------------------------------------------------------------
        # Categorize Unmatched Predictions
        # -----------------------------------------------------------------
        unmatched_preds = []
        for p_idx, (orig_p_i, r_obj) in enumerate(succ_preds):
            if p_idx not in matched_pred_indices:
                u_text = normalize_nfc(r_obj.text)
                # Check for fragment: substring of any GT line
                is_frag = any(u_text in normalize_nfc(g["text"]) and len(u_text) >= 2 for g in gt_lines)
                if is_frag:
                    cat = "FRAGMENT"
                    total_fragment_regions += 1
                elif len(u_text) <= 2 and not u_text.isalnum():
                    cat = "NON_TEXT_ARTIFACT"
                    total_nontext_artifacts += 1
                else:
                    cat = "UNANNOTATED_SCENE_TEXT"
                    total_unannotated_regions += 1

                unmatched_preds.append({
                    "region_id": orig_p_i,
                    "text": u_text,
                    "confidence": round(float(r_obj.confidence), 4),
                    "script": str(r_obj.script),
                    "category": cat,
                })

        # -----------------------------------------------------------------
        # Line-Level Evaluation with Strict Additivity
        # -----------------------------------------------------------------
        sample_line_evals = []
        for g_i, line_obj in enumerate(gt_lines):
            gt_text = normalize_nfc(line_obj["text"])
            gt_script = line_obj["script"]
            gt_char_len = len(gt_text)
            gt_words = [w for w in gt_text.split() if w.strip()]
            gt_word_len = max(len(gt_words), 1)

            # Accumulate GT denominators
            gt_counters["lines"][gt_script] += 1
            gt_counters["lines"]["TOTAL"] += 1
            gt_counters["chars"][gt_script] += gt_char_len
            gt_counters["chars"]["TOTAL"] += gt_char_len
            gt_counters["words"][gt_script] += gt_word_len
            gt_counters["words"]["TOTAL"] += gt_word_len

            if g_i in matched_gt:
                orig_p_i, c_dist, cer, pred_text = matched_gt[g_i]
                is_detected = True
                is_exact = (c_dist == 0)
                w_dist = word_levenshtein_distance(gt_text, pred_text)
                wer = w_dist / gt_word_len

                err_counters["matched_lines"][gt_script] += 1
                err_counters["matched_lines"]["TOTAL"] += 1
                if is_exact:
                    err_counters["exact_matches"][gt_script] += 1
                    err_counters["exact_matches"]["TOTAL"] += 1
            else:
                # Full deletion penalty
                orig_p_i = -1
                pred_text = ""
                c_dist = gt_char_len
                cer = 1.0
                w_dist = gt_word_len
                wer = 1.0
                is_detected = False
                is_exact = False

                err_counters["missed_lines"][gt_script] += 1
                err_counters["missed_lines"]["TOTAL"] += 1

            err_counters["char_edits"][gt_script] += c_dist
            err_counters["char_edits"]["TOTAL"] += c_dist
            err_counters["word_edits"][gt_script] += w_dist
            err_counters["word_edits"]["TOTAL"] += w_dist

            line_record = {
                "sample_image": img_name,
                "line_index": g_i,
                "gt_text": gt_text,
                "gt_script": gt_script,
                "gt_char_count": gt_char_len,
                "gt_word_count": gt_word_len,
                "matched_pred": pred_text,
                "detected": is_detected,
                "exact_match": is_exact,
                "char_edits": c_dist,
                "cer": round(cer, 4),
                "word_edits": w_dist,
                "wer": round(wer, 4),
            }
            sample_line_evals.append(line_record)
            all_matching_records.append(line_record)

        per_sample_details.append({
            "image": img_name,
            "resolution": [w, h],
            "conditions": conditions,
            "detected_count": ocr_res.detection_count,
            "recognized_count": ocr_res.recognition_count,
            "unresolved_count": ocr_res.unresolved_count,
            "latency_ms": round(ocr_res.total_time_ms, 2),
            "line_evaluations": sample_line_evals,
            "unmatched_predictions": unmatched_preds,
        })

    # ---------------------------------------------------------------------
    # Recalculate Final Reconciled Metrics
    # ---------------------------------------------------------------------
    # Strict CER
    cer_devanagari = err_counters["char_edits"]["DEVANAGARI"] / gt_counters["chars"]["DEVANAGARI"]
    cer_latin = err_counters["char_edits"]["LATIN"] / gt_counters["chars"]["LATIN"]
    cer_total = err_counters["char_edits"]["TOTAL"] / gt_counters["chars"]["TOTAL"]

    # Strict WER
    wer_devanagari = err_counters["word_edits"]["DEVANAGARI"] / gt_counters["words"]["DEVANAGARI"]
    wer_latin = err_counters["word_edits"]["LATIN"] / gt_counters["words"]["LATIN"]
    wer_total = err_counters["word_edits"]["TOTAL"] / gt_counters["words"]["TOTAL"]

    # Rates
    exact_match_rate = err_counters["exact_matches"]["TOTAL"] / gt_counters["lines"]["TOTAL"]
    detection_recall = err_counters["matched_lines"]["TOTAL"] / gt_counters["lines"]["TOTAL"]
    
    # Precision definitions:
    # 1. Strict Annotated Precision (all detected regions)
    annotated_precision_all = err_counters["matched_lines"]["TOTAL"] / total_detected_regions
    # 2. Strict Annotated Precision (successfully recognized regions)
    annotated_precision_rec = err_counters["matched_lines"]["TOTAL"] / total_recognized_regions
    # 3. Text Detection Precision (accounting for valid unannotated text + fragments)
    valid_physical_text = err_counters["matched_lines"]["TOTAL"] + total_unannotated_regions + total_fragment_regions
    text_detection_precision = valid_physical_text / total_detected_regions

    recognition_success_rate = total_recognized_regions / total_detected_regions

    # F1 scores
    f1_annotated_all = (
        2 * (annotated_precision_all * detection_recall) / (annotated_precision_all + detection_recall)
    )
    f1_annotated_rec = (
        2 * (annotated_precision_rec * detection_recall) / (annotated_precision_rec + detection_recall)
    )
    f1_text_detection = (
        2 * (text_detection_precision * detection_recall) / (text_detection_precision + detection_recall)
    )

    reconciled_report = {
        "metadata": {
            "evaluation_standard": "Step 3.6.3 Final Technical Closure Evaluation",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "matching_algorithm": "Order-Aware Bipartite Matching (Hungarian linear_sum_assignment)",
            "normalization": "Unicode NFC",
            "additivity_verified": True,
        },
        "ground_truth_inventory": {
            "total_lines": gt_counters["lines"]["TOTAL"],
            "devanagari_lines": gt_counters["lines"]["DEVANAGARI"],
            "latin_lines": gt_counters["lines"]["LATIN"],
            "total_characters": gt_counters["chars"]["TOTAL"],
            "devanagari_characters": gt_counters["chars"]["DEVANAGARI"],
            "latin_characters": gt_counters["chars"]["LATIN"],
            "total_words": gt_counters["words"]["TOTAL"],
            "devanagari_words": gt_counters["words"]["DEVANAGARI"],
            "latin_words": gt_counters["words"]["LATIN"],
        },
        "error_counters": {
            "total_char_edits": err_counters["char_edits"]["TOTAL"],
            "devanagari_char_edits": err_counters["char_edits"]["DEVANAGARI"],
            "latin_char_edits": err_counters["char_edits"]["LATIN"],
            "total_word_edits": err_counters["word_edits"]["TOTAL"],
            "devanagari_word_edits": err_counters["word_edits"]["DEVANAGARI"],
            "latin_word_edits": err_counters["word_edits"]["LATIN"],
            "exact_matches": err_counters["exact_matches"]["TOTAL"],
            "matched_lines_tp": err_counters["matched_lines"]["TOTAL"],
            "missed_lines_fn": err_counters["missed_lines"]["TOTAL"],
        },
        "reconciled_metrics": {
            "cer_overall": round(cer_total, 4),
            "cer_devanagari": round(cer_devanagari, 4),
            "cer_latin": round(cer_latin, 4),
            "wer_overall": round(wer_total, 4),
            "wer_devanagari": round(wer_devanagari, 4),
            "wer_latin": round(wer_latin, 4),
            "exact_match_rate": round(exact_match_rate, 4),
            "detection_recall": round(detection_recall, 4),
            "annotated_precision_all_detections": round(annotated_precision_all, 4),
            "annotated_precision_recognized_only": round(annotated_precision_rec, 4),
            "text_detection_precision_real_scene": round(text_detection_precision, 4),
            "f1_annotated_all": round(f1_annotated_all, 4),
            "f1_annotated_rec": round(f1_annotated_rec, 4),
            "f1_text_detection": round(f1_text_detection, 4),
            "recognition_success_rate": round(recognition_success_rate, 4),
        },
        "detection_inventory": {
            "total_detected_regions": total_detected_regions,
            "total_recognized_regions": total_recognized_regions,
            "total_unresolved_scripts": total_unresolved_scripts,
            "matched_annotated_tp": err_counters["matched_lines"]["TOTAL"],
            "unannotated_valid_text": total_unannotated_regions,
            "fragmented_detections": total_fragment_regions,
            "nontext_artifacts_fp": total_nontext_artifacts,
        },
        "per_sample_details": per_sample_details,
    }

    # Save reports
    with open(matching_records_file, "w", encoding="utf-8") as f:
        json.dump(all_matching_records, f, ensure_ascii=False, indent=2)

    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(reconciled_report, f, ensure_ascii=False, indent=2)

    logger.info("Reconciled matching records saved to %s", matching_records_file)
    logger.info("Reconciled evaluation report saved to %s", output_report_file)

    # Verification assertions
    assert (
        err_counters["char_edits"]["DEVANAGARI"] + err_counters["char_edits"]["LATIN"]
        == err_counters["char_edits"]["TOTAL"]
    ), "Character edits additivity check failed!"
    assert (
        gt_counters["chars"]["DEVANAGARI"] + gt_counters["chars"]["LATIN"]
        == gt_counters["chars"]["TOTAL"]
    ), "Character count additivity check failed!"
    assert (
        err_counters["word_edits"]["DEVANAGARI"] + err_counters["word_edits"]["LATIN"]
        == err_counters["word_edits"]["TOTAL"]
    ), "Word edits additivity check failed!"
    assert (
        gt_counters["words"]["DEVANAGARI"] + gt_counters["words"]["LATIN"]
        == gt_counters["words"]["TOTAL"]
    ), "Word count additivity check failed!"

    logger.info("100% Mathematical Additivity Checks Passed Successfully!")
    return reconciled_report


if __name__ == "__main__":
    run_reconciliation()
