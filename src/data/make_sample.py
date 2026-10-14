"""Generate the SYNTHETIC sample dataset in data/sample/.

These examples were written by hand for this project and the images are drawn
programmatically (coloured backgrounds with simple "face" strokes). They exist ONLY to
test the data pipeline, the models' plumbing and the user interface. Metrics computed on
them say nothing about real-world performance and must never be reported as results.

The file also deliberately contains one corrupt image, one missing image and one empty
text so that validation and missing-modality handling can be exercised.

Usage:  python -m src.data.make_sample
"""
from __future__ import annotations

import random

import pandas as pd
from PIL import Image, ImageDraw

from ..utils import resolve_path

SAMPLES = [
    # (text, label) -- Bengali, Hindi (Devanagari), Hinglish (romanised, code-mixed), English
    ("আজকের দিনটা দারুণ কাটলো! 😄", "positive"),
    ("আমি খুব খুশি", "positive"),
    ("দারুণ খেলা হয়েছে আজ", "positive"),
    ("यह फिल्म बहुत अच्छी थी 👍", "positive"),
    ("aaj ka din mast tha yaar 😂", "positive"),
    ("Exam pass ho gaya, finally!!! 🎉", "positive"),
    ("Best chai ever ☕❤️", "positive"),
    ("kya baat hai, superb performance", "positive"),
    ("আজ সোমবার", "neutral"),
    ("ট্রেন সকাল ৯টায় ছাড়বে", "neutral"),
    ("कल बारिश हो सकती है", "neutral"),
    ("पानी की बोतल मेज़ पर है", "neutral"),
    ("meeting 5 baje hai", "neutral"),
    ("office ja raha hoon", "neutral"),
    ("This is the new syllabus", "neutral"),
    ("Results will be announced tomorrow", "neutral"),
    ("একদম বাজে সিনেমা 😡", "negative"),
    ("আমার মন খারাপ", "negative"),
    ("পরীক্ষা খুব খারাপ হয়েছে", "negative"),
    ("यह बिल्कुल अच्छा नहीं था", "negative"),
    ("traffic ne din kharab kar diya 😤", "negative"),
    ("network nahi hai, bekaar 😒", "negative"),
    ("mood off hai aaj", "negative"),
    ("Worst service ever, never again", "negative"),
]

PALETTE = {
    "positive": [(255, 214, 102), (140, 220, 140), (255, 170, 120)],
    "neutral": [(180, 180, 180), (160, 175, 190), (200, 200, 190)],
    "negative": [(90, 40, 50), (50, 60, 110), (120, 30, 30)],
}


def draw_image(label: str, rng: random.Random, size: int = 256) -> Image.Image:
    bg = rng.choice(PALETTE[label])
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    cx, cy, r = size // 2 + rng.randint(-20, 20), size // 2 + rng.randint(-10, 10), 70
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(20, 20, 20), width=5)
    d.ellipse([cx - 30, cy - 25, cx - 18, cy - 13], fill=(20, 20, 20))
    d.ellipse([cx + 18, cy - 25, cx + 30, cy - 13], fill=(20, 20, 20))
    if label == "positive":
        d.arc([cx - 40, cy - 10, cx + 40, cy + 40], 20, 160, fill=(20, 20, 20), width=5)
    elif label == "negative":
        d.arc([cx - 40, cy + 20, cx + 40, cy + 70], 200, 340, fill=(20, 20, 20), width=5)
    else:
        d.line([cx - 35, cy + 30, cx + 35, cy + 30], fill=(20, 20, 20), width=5)
    d.rectangle([0, size - 22, size, size], fill=(255, 255, 255))
    d.text((6, size - 18), "SYNTHETIC SAMPLE - NOT REAL DATA", fill=(200, 0, 0))
    return img


def main(seed: int = 42) -> None:
    out = resolve_path("data/sample")
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for i, (text, label) in enumerate(SAMPLES):
        name = f"synthetic_{i:03d}.png"
        draw_image(label, rng).save(img_dir / name)
        rows.append({"id": f"synthetic_{i:03d}", "text": text, "image": name, "label": label, "synthetic": True})
    # deliberately problematic records -> removed / flagged by validation
    (img_dir / "synthetic_corrupt.png").write_bytes(b"this is not a PNG file")
    rows.append({"id": "synthetic_corrupt", "text": "corrupt image row", "image": "synthetic_corrupt.png",
                 "label": "neutral", "synthetic": True})
    rows.append({"id": "synthetic_missing", "text": "missing image row", "image": "does_not_exist.png",
                 "label": "neutral", "synthetic": True})
    name = "synthetic_notext.png"
    draw_image("positive", rng).save(img_dir / name)
    rows.append({"id": "synthetic_notext", "text": "", "image": name, "label": "positive", "synthetic": True})
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    (out / "README.md").write_text(
        "# SYNTHETIC sample data\n\n"
        "Hand-written sentences and programmatically drawn images, created for this project to test the\n"
        "pipeline, the model plumbing and the Streamlit interface. **Not real social-media data. Never\n"
        "report metrics computed on it.** Regenerate with `python -m src.data.make_sample`.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} synthetic rows to {out / 'labels.csv'}")


if __name__ == "__main__":
    main()
