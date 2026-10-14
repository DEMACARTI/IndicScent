"""Generate an explanation gallery for test-set memes.

For a deterministic selection of test examples (per true class: one the best multimodal
model gets right and one it gets wrong, when available) this renders one figure per meme:
original image, Grad-CAM for the image-only and multimodal models, and occlusion word
attributions for the text-only and multimodal models. A JSON file stores all raw scores,
including Integrated-Gradients residuals.

    python -m src.explainability.generate_examples [--n-per-class 2]

Output: artifacts/reports/explanations/
"""
from __future__ import annotations

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import Image

from ..data.dataset import load_split
from ..inference import SentimentPredictor, best_multimodal
from ..utils import load_config, resolve_path, save_json


def _indic_font() -> str | None:
    """Pick an installed font that can render Bengali/Devanagari for figure labels."""
    for name in ["Noto Sans Bengali", "Kohinoor Bangla", "Bangla Sangam MN", "Vrinda", "Nirmala UI",
                 "Lohit Bengali", "Noto Sans Devanagari", "Arial Unicode MS"]:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return None


def select_examples(cfg: dict, mm_exp: str, n_per_class: int) -> pd.DataFrame:
    test = load_split(cfg, "test")
    preds = pd.read_csv(resolve_path(cfg["paths"]["reports_dir"]) / mm_exp / "predictions.csv")
    df = test.merge(preds[["id", "pred"]], on="id")
    df["correct"] = df["label"] == df["pred"]
    picks = []
    for label in cfg["labels"]:
        sub = df[df["label"] == label]
        for correct in (True, False):
            s = sub[sub["correct"] == correct]
            picks.append(s.sample(n=min(len(s), max(1, n_per_class // 2)), random_state=cfg["seed"]))
    return pd.concat(picks).drop_duplicates("id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-per-class", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pred = SentimentPredictor(args.config)
    avail = pred.available()
    mm = best_multimodal(avail)
    if mm is None or not (resolve_path(cfg["paths"]["reports_dir"]) / mm / "predictions.csv").exists():
        print("Need a trained + evaluated multimodal model first.")
        return
    out = resolve_path(cfg["paths"]["reports_dir"]) / "explanations"
    out.mkdir(parents=True, exist_ok=True)
    font = _indic_font()
    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    examples = select_examples(cfg, mm, args.n_per_class)
    records = []
    for _, row in examples.iterrows():
        img_path = resolve_path(row["image_source"] or row["image_path"])   # original resolution
        res = {}
        for exp in ["text", "image", mm]:
            if exp not in avail:
                continue
            res[exp] = pred.predict(row["text"], img_path if exp != "text" else None, exp, explain=True)
        fig = plt.figure(figsize=(16, 7.5))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.set_title(f"True: {row['label']}", fontsize=11)
        ax.axis("off")
        for col, exp in [(1, "image"), (2, mm)]:
            if exp in res and res[exp].gradcam is not None:
                ax = fig.add_subplot(gs[0, col])
                ax.imshow(res[exp].gradcam)
                r = res[exp]
                ax.set_title(f"Grad-CAM · {exp}\npred {r.label} (p={r.confidence:.2f})", fontsize=10)
                ax.axis("off")
        for col, exp in [(0, "text"), (1, mm)]:
            if exp in res and res[exp].text_occlusion is not None:
                ax = fig.add_subplot(gs[1, col])
                ws = res[exp].text_occlusion["words"][:25]
                ax.barh(range(len(ws)), [w["score"] for w in ws],
                        color=["#2a9d5c" if w["score"] >= 0 else "#d64545" for w in ws])
                ax.set_yticks(range(len(ws)), [w["word"] for w in ws], fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color="k", lw=0.6)
                r = res[exp]
                ax.set_title(f"Occlusion · {exp}: pred {r.label} (p={r.confidence:.2f})", fontsize=10)
                ax.set_xlabel("Δ P(pred) when word removed")
        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        caption = "\n".join(textwrap.wrap(row["text"] or "(no text)", 60)[:8])
        ax.text(0, 1, f"id: {row['id']}\n\n{caption}", va="top", fontsize=9)
        fig.suptitle("Explanations show model sensitivity, not causal reasons. Grad-CAM is 7×7 resolution.",
                     fontsize=9, color="#555")
        fig.tight_layout()
        fig.savefig(out / f"{row['id']}.png", dpi=110)
        plt.close(fig)
        records.append({"id": row["id"], "true": row["label"], "text": row["text"],
                        **{exp: {"pred": r.label, "probabilities": r.probabilities,
                                 "occlusion": r.text_occlusion["words"] if r.text_occlusion else None,
                                 "integrated_gradients": None if r.text_attribution is None else {
                                     k: r.text_attribution[k] for k in
                                     ("words", "convergence_delta", "reliable", "prob_input", "prob_baseline")},
                                 "gradcam_7x7": None if r.gradcam_raw is None else r.gradcam_raw.round(3).tolist()}
                           for exp, r in res.items()}})
        print(f"explained {row['id']}")
    save_json({"multimodal_model": mm, "font": font, "examples": records}, out / "explanations.json")
    ig = [(e, r[e]["integrated_gradients"]["reliable"]) for r in records for e in r
          if isinstance(r[e], dict) and r[e].get("integrated_gradients")]
    if ig:
        print(f"IG completeness check passed for {sum(v for _, v in ig)}/{len(ig)} explanations")
    print(f"Wrote {len(records)} explanation figures to {out}")


if __name__ == "__main__":
    main()
