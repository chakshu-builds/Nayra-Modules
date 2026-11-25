"""Generate 15 verified English test-line crops for Module 07 English PoC benchmark."""

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

# 1. Load dictionary to check coverage
lines = Path("models/ocr/en_PP-OCRv5_mobile_rec.yml").read_text(encoding="utf-8").splitlines()
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

# 15 diverse test cases
test_cases = [
    {
        "id": "sample_en_01",
        "category": "Uppercase public signage",
        "ground_truth": "EMERGENCY EXIT",
    },
    {
        "id": "sample_en_02",
        "category": "Title case brand & assistive system",
        "ground_truth": "NAYRA Assistive Vision",
    },
    {
        "id": "sample_en_03",
        "category": "Lowercase sentence with period",
        "ground_truth": "please keep this area clear.",
    },
    {
        "id": "sample_en_04",
        "category": "Alphanumeric transit gate & terminal",
        "ground_truth": "Gate 4B - Terminal 2",
    },
    {
        "id": "sample_en_05",
        "category": "Train & platform signage with digits",
        "ground_truth": "Train 1240 to Central Station",
    },
    {
        "id": "sample_en_06",
        "category": "Currency symbol, price & parentheses",
        "ground_truth": "Total: $49.99 (Tax Incl.)",
    },
    {
        "id": "sample_en_07",
        "category": "Safety warning with colon and exclamation",
        "ground_truth": "WARNING: High Voltage!",
    },
    {
        "id": "sample_en_08",
        "category": "Hyphenated technical product identifier",
        "ground_truth": "Model: NYR-8000-PRO",
    },
    {
        "id": "sample_en_09",
        "category": "Room directory & professional title",
        "ground_truth": "Room 402: Chief Medical Officer",
    },
    {
        "id": "sample_en_10",
        "category": "Email, URL and pipe delimiter",
        "ground_truth": "contact@nayra.org | www.nayra.ai",
    },
    {
        "id": "sample_en_11",
        "category": "Date, timestamp & hyphens",
        "ground_truth": "Date: 25-Aug-2026 14:30",
    },
    {
        "id": "sample_en_12",
        "category": "Door instructions & forward slash",
        "ground_truth": "Push to Open / Pull to Close",
    },
    {
        "id": "sample_en_13",
        "category": "Phone number, plus sign & parentheses",
        "ground_truth": "Call Support: +1 (800) 555-0199",
    },
    {
        "id": "sample_en_14",
        "category": "Percentage symbol & audit statement",
        "ground_truth": "Safety Compliance 100%",
    },
    {
        "id": "sample_en_15",
        "category": "Street address with comma & numerals",
        "ground_truth": "221B Baker Street, London",
    },
]

print("--- English Dictionary Character Verification ---")
all_ok = True
for tc in test_cases:
    gt = tc["ground_truth"]
    missing = [c for c in gt if c not in dict_chars]
    status = "OK" if not missing else f"MISSING: {missing}"
    if missing:
        all_ok = False
    print(f"[{status}] {tc['id']}: '{gt}'")

if not all_ok:
    print("Warning: some characters are missing from dictionary!")
else:
    print("All characters are verified present in English dictionary.")

out_dir = Path("07_ocr/data/poc_en")
out_dir.mkdir(parents=True, exist_ok=True)

# Generate rendered images using standard clean font (Arial)
font_path = "C:/Windows/Fonts/arial.ttf"
font_size = 36
font = ImageFont.truetype(font_path, font_size)

for tc in test_cases:
    gt = tc["ground_truth"]
    dummy = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), gt, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
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

gt_file = out_dir / "ground_truth.json"
with open(gt_file, "w", encoding="utf-8") as f:
    json.dump(test_cases, f, ensure_ascii=False, indent=2)

print(f"\nGenerated {len(test_cases)} test images in {out_dir}")
print(f"Saved ground truth catalog to {gt_file}")
