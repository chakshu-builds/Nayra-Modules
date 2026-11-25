"""Audit all 103 regions produced by DBNet across the 12 real camera scenes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import cv2
import numpy as np
from pipeline.ocr_engine import OCREngine
from pipeline.result_schema import RegionStatus
from pipeline.text_cropper import extract_perspective_crop

def audit():
    engine = OCREngine(use_batching=False)
    gt_file = Path(__file__).parent / "data" / "real_camera" / "ground_truth.json"
    with open(gt_file, "r", encoding="utf-8") as f:
        gt = json.load(f)

    all_regions = []
    
    for sample in gt:
        img_name = sample["image"]
        img_path = Path(__file__).parent / "data" / "real_camera" / img_name
        img = cv2.imread(str(img_path))
        res = engine.process_image(img)
        
        det_res = engine.detector.detect(img)
        
        for i in range(det_res.num_detections):
            box_4pt = det_res.boxes_4pt[i]
            crop, crop_err = extract_perspective_crop(img, box_4pt)
            r_obj = next((r for r in res.regions if r.region_id == i), None)
            
            status = r_obj.status.value if r_obj else "MISSING"
            text = r_obj.text if r_obj else ""
            script = str(r_obj.script) if r_obj else "UNKNOWN"
            conf = r_obj.confidence if r_obj else 0.0
            r_conf = r_obj.routing_confidence if r_obj else 0.0
            shape = crop.shape if crop is not None else (0, 0)
            
            all_regions.append({
                "image": img_name,
                "region_id": i,
                "status": status,
                "script": script,
                "text": text,
                "confidence": conf,
                "routing_confidence": r_conf,
                "shape": shape,
                "box_xyxy": [round(float(v), 1) for v in det_res.boxes_xyxy[i]]
            })

    print(f"Total regions audited: {len(all_regions)}")
    
    # Breakdown by status
    success_regions = [r for r in all_regions if r["status"] == RegionStatus.SUCCESS.value]
    unresolved_regions = [r for r in all_regions if r["status"] != RegionStatus.SUCCESS.value]
    
    print(f"  Successfully recognized: {len(success_regions)}")
    print(f"  Unresolved / failed:     {len(unresolved_regions)}")
    
    print("\n--- ALL UNRESOLVED REGIONS (14 Regions) ---")
    for u in unresolved_regions:
        print(f"{u['image']:<30} ID:{u['region_id']:<2} | status:{u['status']:<18} | script:{u['script']:<10} | r_conf:{u['routing_confidence']:.2f} | shape:{u['shape']} | box:{u['box_xyxy']}")

    # Let's inspect what is physically in each of these 14 crops!
    print("\n--- DETAILED INSPECTION OF THE 14 UNRESOLVED CROPS ---")
    for u in unresolved_regions:
        img_path = Path(__file__).parent / "data" / "real_camera" / u["image"]
        img = cv2.imread(str(img_path))
        x1, y1, x2, y2 = [int(v) for v in u["box_xyxy"]]
        crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        h, w = crop.shape[:2]
        # Let's check mean brightness and contrast
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else np.zeros((1, 1))
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        print(f"{u['image']} [Region {u['region_id']}]: size={w}x{h}, mean_gray={mean_val:.1f}, std_gray={std_val:.1f}")

if __name__ == "__main__":
    audit()
