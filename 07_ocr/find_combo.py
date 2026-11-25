"""Search for the exact subset of lines that produces 191 chars and 14 edits."""

import json
import itertools

with open("07_ocr/data/real_camera/ground_truth.json", "r", encoding="utf-8") as f:
    gt = json.load(f)

# Let's collect all lines
all_lines = []
for s in gt:
    for l in s["lines"]:
        all_lines.append((s["image"], l["text"], l["script"], len(l["text"].strip())))

# All DEVANAGARI lines
dev_lines = [l for l in all_lines if l[2] == "DEVANAGARI"]
print(f"DEVANAGARI lines count: {len(dev_lines)}, total chars: {sum(l[3] for l in dev_lines)}")

# Latin lines
lat_lines = [l for l in all_lines if l[2] == "LATIN"]
print(f"LATIN lines count: {len(lat_lines)}, total chars: {sum(l[3] for l in lat_lines)}")

# Can any combination of Latin lines sum to 41 chars?
print("\nSubsets of LATIN lines summing to 41 chars:")
for r in range(1, 5):
    for combo in itertools.combinations(lat_lines, r):
        if sum(c[3] for c in combo) == 41:
            print("  Combo:", [c[1] for c in combo], "from", [c[0] for c in combo])
