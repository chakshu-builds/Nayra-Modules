"""Test effect of canvas pad width on recognition of narrow crops."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from pipeline.ocr_engine import OCREngine
from poc_hindi_ocr import ctc_decode

def main():
    engine = OCREngine(use_batching=False)
    img = cv2.imread(str(Path(__file__).parent / "data" / "real_camera" / "real_05_medicine.jpg"))
    res = engine.process_image(img)

    for r in res.regions:
        if "5.5" in r.text or "months" in r.text:
            x1, y1, x2, y2 = [int(v) for v in r.box_xyxy]
            crop = img[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            ratio = w / float(h)
            resized_w = int(round(48 * ratio))
            print(f"\nTarget crop: [{r.text}], original: {w}x{h}, resized_w={resized_w}")
            
            model = engine.hindi_model
            for pad_w in [resized_w, 256, 320, 384, 448, 512, 576, 640, 704, 768]:
                pad_w_eff = max(pad_w, resized_w)
                if pad_w_eff % 32 != 0:
                    pad_w_eff = int(np.ceil(pad_w_eff / 32) * 32)
                resized = cv2.resize(crop, (resized_w, 48), interpolation=cv2.INTER_LINEAR)
                norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
                canvas = np.zeros((1, 3, 48, pad_w_eff), dtype=np.float32)
                canvas[0, :, :, :resized_w] = norm.transpose((2, 0, 1))
                valid_ratio = min(1.0, resized_w / float(pad_w_eff))
                
                preds = model.session.run(None, {model.session.get_inputs()[0].name: canvas})[0]
                text, conf, details = ctc_decode(preds[0], model.character_list, valid_ratio=valid_ratio)
                print(f"  pad_w={pad_w_eff:<4} (T={preds.shape[1]}) -> [{text}] (conf: {conf:.4f})")

if __name__ == "__main__":
    main()
