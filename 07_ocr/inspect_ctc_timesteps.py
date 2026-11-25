"""Inspect exact timestep argmax and probability distributions for the whitespace token."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from pipeline.ocr_engine import OCREngine
from pipeline.text_cropper import extract_perspective_crop

def inspect_tokens():
    engine = OCREngine(use_batching=False)
    img = cv2.imread(str(Path(__file__).parent / "data" / "real_camera" / "real_05_medicine.jpg"))
    det_res = engine.detector.detect(img)
    
    # Target crop 4: >=5.5 pounds and
    crop = extract_perspective_crop(img, det_res.boxes_4pt[4])[0]
    model = engine.hindi_model
    char_list = model.character_list
    space_idx = char_list.index(" ") if " " in char_list else -1
    print(f"Space token index in Hindi dictionary: {space_idx}")
    
    h, w = crop.shape[:2]
    resized_w = int(round(48 * (w / float(h))))
    resized = cv2.resize(crop, (resized_w, 48), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    
    # 1. Canvas 320 (Sequential)
    canvas_320 = np.zeros((1, 3, 48, 320), dtype=np.float32)
    canvas_320[0, :, :, :resized_w] = norm.transpose((2, 0, 1))
    preds_320 = model.session.run(None, {model.session.get_inputs()[0].name: canvas_320})[0][0]
    
    # 2. Canvas 448 (Batched with wide crop)
    canvas_448 = np.zeros((1, 3, 48, 448), dtype=np.float32)
    canvas_448[0, :, :, :resized_w] = norm.transpose((2, 0, 1))
    preds_448 = model.session.run(None, {model.session.get_inputs()[0].name: canvas_448})[0][0]
    
    print(f"\n--- CANVAS 320 (T={preds_320.shape[0]}, valid_T={int(round(preds_320.shape[0]*resized_w/320))}) ---")
    argmax_320 = np.argmax(preds_320, axis=-1)
    emitted_320 = []
    prev = -1
    for t in range(int(round(preds_320.shape[0] * resized_w / 320))):
        idx = argmax_320[t]
        ch = char_list[idx] if idx < len(char_list) else "?"
        prob = preds_320[t, idx]
        sp_prob = preds_320[t, space_idx]
        blank_prob = preds_320[t, 0]
        if idx != 0 and idx != prev:
            emitted_320.append((t, ch, prob))
        if "5" in ch or "p" in ch or idx == space_idx or sp_prob > 0.05:
            print(f"  t={t:2d}: argmax={idx:3d} ('{ch}': {prob:.4f}) | blank: {blank_prob:.4f} | space({space_idx}): {sp_prob:.4f}")
        prev = idx
    print("Decoded text (320):", "".join(e[1] for e in emitted_320))

    print(f"\n--- CANVAS 448 (T={preds_448.shape[0]}, valid_T={int(round(preds_448.shape[0]*resized_w/448))}) ---")
    argmax_448 = np.argmax(preds_448, axis=-1)
    emitted_448 = []
    prev = -1
    for t in range(int(round(preds_448.shape[0] * resized_w / 448))):
        idx = argmax_448[t]
        ch = char_list[idx] if idx < len(char_list) else "?"
        prob = preds_448[t, idx]
        sp_prob = preds_448[t, space_idx]
        blank_prob = preds_448[t, 0]
        if idx != 0 and idx != prev:
            emitted_448.append((t, ch, prob))
        if "5" in ch or "p" in ch or idx == space_idx or sp_prob > 0.05:
            print(f"  t={t:2d}: argmax={idx:3d} ('{ch}': {prob:.4f}) | blank: {blank_prob:.4f} | space({space_idx}): {sp_prob:.4f}")
        prev = idx
    print("Decoded text (448):", "".join(e[1] for e in emitted_448))

if __name__ == "__main__":
    inspect_tokens()
