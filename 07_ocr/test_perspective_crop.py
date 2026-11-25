"""Check perspective crop vs axis-aligned crop and padding effect on spaces."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from pipeline.ocr_engine import OCREngine
from pipeline.text_cropper import extract_perspective_crop
from poc_hindi_ocr import ctc_decode
from pipeline.batch_recognizer import predict_batch

def main():
    engine = OCREngine(use_batching=False)
    img = cv2.imread(str(Path(__file__).parent / "data" / "real_camera" / "real_05_medicine.jpg"))
    det_res = engine.detector.detect(img)
    
    crops = []
    for i in range(det_res.num_detections):
        box_4pt = det_res.boxes_4pt[i]
        crop, err = extract_perspective_crop(img, box_4pt)
        crops.append(crop)

    model = engine.hindi_model
    
    # 1. Sequential prediction on all crops
    seq_preds = [model.predict(c) for c in crops]
    
    # 2. Batched prediction on all crops
    batch_preds = predict_batch(model, crops)
    
    print("\n=== COMPARISON FOR REAL_05 MEDICINE CROPS ===")
    for i, (c, (s_txt, s_conf), (b_txt, b_conf)) in enumerate(zip(crops, seq_preds, batch_preds)):
        h, w = c.shape[:2]
        ratio = w / float(h)
        res_w = int(round(48 * ratio))
        diff = " *** MISMATCH ***" if s_txt != b_txt else ""
        print(f"Crop {i} ({w}x{h}, res_w={res_w}):")
        print(f"   Sequential: [{s_txt}] (conf: {s_conf:.4f})")
        print(f"   Batched:    [{b_txt}] (conf: {b_conf:.4f}){diff}")

    # Width-bucketed batching evaluation:
    print("\n=== WIDTH-BUCKETED BATCHING EVALUATION ===")
    # bucket crops into aspect ratio / width bins (e.g. narrow: res_w < 350, wide: res_w >= 350)
    narrow = [c for c in crops if int(round(48 * (c.shape[1] / float(c.shape[0])))) < 350]
    wide = [c for c in crops if int(round(48 * (c.shape[1] / float(c.shape[0])))) >= 350]
    print(f"Narrow bucket ({len(narrow)} crops):")
    narrow_preds = predict_batch(model, narrow)
    for c, (t, cf) in zip(narrow, narrow_preds):
        print(f"   [{t}] (conf: {cf:.4f})")
    print(f"Wide bucket ({len(wide)} crops):")
    wide_preds = predict_batch(model, wide)
    for c, (t, cf) in zip(wide, wide_preds):
        print(f"   [{t}] (conf: {cf:.4f})")

if __name__ == "__main__":
    main()
