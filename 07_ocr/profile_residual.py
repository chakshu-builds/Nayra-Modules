"""Profile pipeline residual time with microsecond instrumentation across 50 warm iterations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import cv2
import numpy as np
from pipeline.ocr_engine import OCREngine
from pipeline.text_cropper import extract_perspective_crop
from pipeline.reading_order import sort_regions_reading_order, format_combined_text

def run_profiling(num_warm: int = 20):
    img_path = Path(__file__).parent / "data" / "real_camera" / "real_01_metro_mixed.jpg"
    img = cv2.imread(str(img_path))
    assert img is not None, "Failed to load image"

    print(f"Initializing OCREngine for profiling on {img.shape[1]}x{img.shape[0]} frame...")
    engine = OCREngine(use_batching=True)
    
    # 3 warmups
    for _ in range(3):
        engine.process_image(img)
    print(f"Warmup complete. Profiling {num_warm} iterations...")

    t_total = []
    t_det = []
    t_routing = []
    t_rec = []
    t_residual = []
    t_crop_only = []

    # Measure crop extraction separately on detected boxes
    det_res = engine.detector.detect(img)
    boxes = [det_res.boxes_4pt[i] for i in range(det_res.num_detections)]
    for _ in range(num_warm):
        tc0 = time.perf_counter()
        for b in boxes:
            extract_perspective_crop(img, b)
        tc1 = time.perf_counter()
        t_crop_only.append((tc1 - tc0) * 1000.0)

    for it in range(num_warm):
        res = engine.process_image(img)
        t_total.append(res.total_time_ms)
        t_det.append(res.detection_time_ms)
        t_routing.append(res.routing_time_ms)
        t_rec.append(res.recognition_time_ms)
        res_diff = res.total_time_ms - (res.detection_time_ms + res.routing_time_ms + res.recognition_time_ms)
        t_residual.append(res_diff)

    med_total = float(np.median(t_total))
    med_det = float(np.median(t_det))
    med_crop = float(np.median(t_crop_only))
    med_rout = float(np.median(t_routing))
    med_rec = float(np.median(t_rec))
    med_residual = float(np.median(t_residual))

    print("\n=== DETAILED RESIDUAL BREAKDOWN ===")
    print(f"Total Pipeline Median:        {med_total:6.2f} ms (P95: {np.percentile(t_total, 95):6.2f} ms)")
    print(f"  1. DBNet Detection:         {med_det:6.2f} ms ({med_det/med_total*100:4.1f}%)")
    print(f"  2. Script Routing:          {med_rout:6.2f} ms ({med_rout/med_total*100:4.1f}%)")
    print(f"     (Per-crop routing avg):  {med_rout/max(len(boxes), 1):6.2f} ms/crop ({len(boxes)} crops)")
    print(f"  3. Batched Recognition:     {med_rec:6.2f} ms ({med_rec/med_total*100:4.1f}%)")
    print(f"  4. Residual Overheads:      {med_residual:6.2f} ms ({med_residual/med_total*100:4.1f}%)")
    print(f"     - Perspective Warping:   {med_crop:6.2f} ms ({med_crop/med_total*100:4.1f}%)")
    print(f"     - Reading Order & Admin: {med_residual - med_crop:6.2f} ms ({(med_residual - med_crop)/med_total*100:4.1f}%)")

if __name__ == "__main__":
    run_profiling(20)
