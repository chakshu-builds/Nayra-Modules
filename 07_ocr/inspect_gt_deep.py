"""Deep inspection of ground_truth.json lines, characters, words, and scripts."""

import json
import unicodedata

def analyze():
    with open("07_ocr/data/real_camera/ground_truth.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    scripts = {}
    total_lines = 0
    total_chars = 0
    total_words = 0

    print("SAMPLE-BY-SAMPLE GROUND TRUTH AUDIT:")
    for s_i, s in enumerate(gt):
        img = s["image"]
        lines = s["lines"]
        s_chars = 0
        s_words = 0
        print(f"\n[{s_i+1}/12] {img} ({len(lines)} lines):")
        for l_i, l in enumerate(lines):
            raw_txt = l["text"]
            txt = unicodedata.normalize("NFC", raw_txt.strip())
            sc = l.get("script", "UNKNOWN")
            words = [w for w in txt.split() if w.strip()]
            num_chars = len(txt)
            num_words = len(words)
            s_chars += num_chars
            s_words += num_words

            if sc not in scripts:
                scripts[sc] = {"lines": 0, "chars": 0, "words": 0}
            scripts[sc]["lines"] += 1
            scripts[sc]["chars"] += num_chars
            scripts[sc]["words"] += num_words

            total_lines += 1
            total_chars += num_chars
            total_words += num_words

            print(f"  L{l_i+1}: script={sc:<10} chars={num_chars:2d} words={num_words:2d} | '{txt}'")

        print(f"  Subtotal: chars={s_chars}, words={s_words}")

    print("\n" + "=" * 60)
    print("TOTALS BY SCRIPT IN GROUND TRUTH:")
    print("=" * 60)
    for sc, c in sorted(scripts.items()):
        print(f"  {sc:<15}: {c['lines']:2d} lines, {c['chars']:3d} chars, {c['words']:3d} words")
    print(f"  {'TOTAL':<15}: {total_lines:2d} lines, {total_chars:3d} chars, {total_words:3d} words")

if __name__ == "__main__":
    analyze()
