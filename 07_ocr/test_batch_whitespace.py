"""Investigate dynamic batching whitespace regression on real_05_medicine.jpg."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from pipeline.ocr_engine import OCREngine

def main():
    img_path = Path(__file__).parent / "data" / "real_camera" / "real_05_medicine.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print("Failed to read image")
        return

    print("Initializing sequential engine...")
    engine_seq = OCREngine(use_batching=False)
    res_seq = engine_seq.process_image(img)

    print("Initializing batched engine...")
    engine_batch = OCREngine(use_batching=True)
    res_batch = engine_batch.process_image(img)

    print("\n--- SEQUENTIAL RECOGNITION RESULTS ---")
    for r in res_seq.regions:
        print(f"[{r.text}] (conf: {r.confidence:.4f}, script: {r.script})")

    print("\n--- DYNAMIC BATCHED RECOGNITION RESULTS ---")
    for r in res_batch.regions:
        print(f"[{r.text}] (conf: {r.confidence:.4f}, script: {r.script})")

if __name__ == "__main__":
    main()
