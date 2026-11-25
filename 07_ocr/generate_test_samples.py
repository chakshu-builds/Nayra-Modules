"""Generate 10 verified Hindi test-line crops for Module 07 POC benchmark."""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

# 1. Load dictionary to check coverage
lines = Path("models/ocr/devanagari_PP-OCRv5_mobile_rec.yml").read_text(encoding="utf-8").splitlines()
in_dict = False
dict_chars = set()
for line in lines:
    if line.strip().startswith("character_dict:"):
        in_dict = True
        continue
    if in_dict:
        if line.startswith("  - "):
            val = line[4:]
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]
            dict_chars.add(val)
        elif not line.startswith(" "):
            break

dict_chars.add(" ")

# 10 test samples
test_cases = [
    {
        "id": "sample_01",
        "category": "Basic greeting & common conjunct (स्त)",
        "ground_truth": "नमस्ते भारत",
    },
    {
        "id": "sample_02",
        "category": "Double consonants & conjuncts (ल्ल, श्व, द्य)",
        "ground_truth": "दिल्ली विश्वविद्यालय",
    },
    {
        "id": "sample_03",
        "category": "Public signage & conjunct (स्पताल, आपातकालीन)",
        "ground_truth": "अस्पताल एवं आपातकालीन सेवा",
    },
    {
        "id": "sample_04",
        "category": "Punctuation (danda ।) & nasalization (ँ, ं)",
        "ground_truth": "कृपया यहाँ कूड़ा न फेंकें।",
    },
    {
        "id": "sample_05",
        "category": "Official signage & pipe separator (|)",
        "ground_truth": "भारत सरकार | गृह मंत्रालय",
    },
    {
        "id": "sample_06",
        "category": "Devanagari numerals (२५, २०२६) & colon (:)",
        "ground_truth": "दिनांक: २५ अगस्त २०२६",
    },
    {
        "id": "sample_07",
        "category": "Devanagari numerals, currency & conjunct (प्रति, व्यक्ति)",
        "ground_truth": "शुल्क: रु ५०० प्रति व्यक्ति",  # Note: checking ₹ vs रु
    },
    {
        "id": "sample_08",
        "category": "Arabic digits (100%), Latin punctuation, complex conjuncts",
        "ground_truth": "सुरक्षा नियम 100% अनिवार्य",
    },
    {
        "id": "sample_09",
        "category": "Nuqta loan consonant (ज़) & conjuncts (प्रशासन, कार्यालय)",
        "ground_truth": "ज़िला प्रशासन कार्यालय",
    },
    {
        "id": "sample_10",
        "category": "Transit signage, Arabic digits (12345), danda (।)",
        "ground_truth": "ट्रेन संख्या 12345 समय पर है।",
    },
]

# Verify characters in dictionary
print("--- Dictionary Verification ---")
for tc in test_cases:
    gt = tc["ground_truth"]
    missing = [c for c in gt if c not in dict_chars]
    status = "OK" if not missing else f"MISSING: {missing}"
    print(f"[{status}] {tc['id']}: '{gt}'")

# Check if '₹' is in dict
print("Is '₹' in dict?", "₹" in dict_chars)
print("Is 'रु' in dict?", "र" in dict_chars and "ु" in dict_chars)

out_dir = Path("07_ocr/data/poc")
out_dir.mkdir(parents=True, exist_ok=True)

# Generate rendered images using Nirmala UI
font_path = "C:/Windows/Fonts/Nirmala.ttc"
font_size = 36
font = ImageFont.truetype(font_path, font_size)

for tc in test_cases:
    gt = tc["ground_truth"]
    
    # Calculate text bounding box
    dummy = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), gt, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Image with padding
    pad_x = 24
    pad_y = 16
    img_w = text_w + pad_x * 2
    img_h = max(54, text_h + pad_y * 2)
    
    im = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), gt, font=font, fill=(15, 15, 15))
    
    img_filename = f"{tc['id']}.png"
    img_path = out_dir / img_filename
    im.save(img_path)
    tc["image_path"] = str(img_path.as_posix())
    tc["width"] = img_w
    tc["height"] = img_h

# Save ground truth metadata
gt_file = out_dir / "ground_truth.json"
with open(gt_file, "w", encoding="utf-8") as f:
    json.dump(test_cases, f, ensure_ascii=False, indent=2)

print(f"\nGenerated {len(test_cases)} test images in {out_dir}")
print(f"Saved ground truth catalog to {gt_file}")
