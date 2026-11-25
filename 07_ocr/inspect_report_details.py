"""Inspect real_camera_evaluation_report.json to see how char_edits and characters were tracked."""

import json

with open("07_ocr/real_camera_evaluation_report.json", "r", encoding="utf-8") as f:
    rep = json.load(f)

dev_edits = 0
dev_chars = 0
dev_w_edits = 0
dev_words = 0

lat_edits = 0
lat_chars = 0
lat_w_edits = 0
lat_words = 0

all_edits = 0
all_chars = 0
all_w_edits = 0
all_words = 0

for s in rep["per_sample_details"]:
    img = s["image"]
    for l in s["line_evaluations"]:
        sc = l["gt_script"]
        gt = l["gt_text"]
        pred = l.get("best_pred", l.get("matched_pred", ""))
        ed = l["char_edits"]
        ch = len(gt)
        w_ed = l["word_edits"]
        w_cnt = len([w for w in gt.split() if w.strip()])
        
        all_edits += ed
        all_chars += ch
        all_w_edits += w_ed
        all_words += w_cnt

        if sc == "DEVANAGARI":
            dev_edits += ed
            dev_chars += ch
            dev_w_edits += w_ed
            dev_words += w_cnt
        else:
            lat_edits += ed
            lat_chars += ch
            lat_w_edits += w_ed
            lat_words += w_cnt

print(f"DEVANAGARI: edits={dev_edits}, chars={dev_chars}, CER={dev_edits/dev_chars:.4f} | w_edits={dev_w_edits}, words={dev_words}, WER={dev_w_edits/dev_words:.4f}")
print(f"LATIN:      edits={lat_edits}, chars={lat_chars}, CER={lat_edits/lat_chars:.4f} | w_edits={lat_w_edits}, words={lat_words}, WER={lat_w_edits/lat_words:.4f}")
print(f"TOTAL:      edits={all_edits}, chars={all_chars}, CER={all_edits/all_chars:.4f} | w_edits={all_w_edits}, words={all_words}, WER={all_w_edits/all_words:.4f}")
print(f"Sum of edits (dev + lat): {dev_edits + lat_edits} == all_edits: {dev_edits + lat_edits == all_edits}")
print(f"Sum of chars (dev + lat): {dev_chars + lat_chars} == all_chars: {dev_chars + lat_chars == all_chars}")
print(f"Sum of w_edits (dev + lat): {dev_w_edits + lat_w_edits} == all_w_edits: {dev_w_edits + lat_w_edits == all_w_edits}")
print(f"Sum of words (dev + lat): {dev_words + lat_words} == all_words: {dev_words + lat_words == all_words}")
