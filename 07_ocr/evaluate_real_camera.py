"""NAYRA Module 07 — Real-World Camera OCR Evaluation Suite (Audited & Reconciled).

Executes the reconciled evaluation protocol with:
- Strict 1-to-1 Geometry-Aware Bipartite Matching (Hungarian algorithm via linear_sum_assignment).
- 100% Mathematical Additivity across Hindi, English, and Overall metrics.
- Complete detection precision breakdown with N=103 detector-produced regions.
- Unicode NFC normalization and full deletion penalties on missed lines.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

# Ensure root and module imports
ROOT_DIR = Path(__file__).parent.parent
OCR_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(OCR_DIR))

from reconcile_evaluation import run_reconciliation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NAYRA-RealCameraEval")


def run_evaluation():
    report = run_reconciliation()
    # Mirror into real_camera_evaluation_report.json for backward compatibility
    compat_path = OCR_DIR / "real_camera_evaluation_report.json"
    with open(compat_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Evaluation results mirrored to %s", compat_path)
    return report


if __name__ == "__main__":
    run_evaluation()
