"""Build the 6-page project report PDF (docs/IndicSent-M_Project_Report.pdf).

Sections: 1 Proposed solution · 2 Technical stack · 3 Feasibility and viability ·
4 Improvements and benefits · 5 Research and references.

All numbers are read from the evaluation outputs (artifacts/reports*/), the dataset statistics
(data/processed*/stats.json) and the checkpoints, so the report cannot drift from the experiments.

    python scripts/make_report_pdf.py
Requires: reportlab, matplotlib (both in requirements.txt)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "IndicSent-M_Project_Report.pdf"
FIG = ROOT / "docs" / "figures"

INK = colors.HexColor("#1B1F3B")
SAFFRON = colors.HexColor("#E07A1F")
MUTED = colors.HexColor("#5F647A")
TINT = colors.HexColor("#EEF0F7")
GRID = colors.HexColor("#D5D8E3")

# DejaVu ships with matplotlib and covers Δ, →, ≥, ⇄ etc. (the built-in PDF fonts do not)
_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_ttf / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(_ttf / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(_ttf / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(_ttf / "DejaVuSans-BoldOblique.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

S = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=12.6, textColor=colors.HexColor("#222533"),
                           spaceAfter=4, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.4, leading=9.4, textColor=MUTED, spaceAfter=3),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=14.5, leading=18, textColor=INK, spaceBefore=2,
                         spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10, leading=13, textColor=SAFFRON, spaceBefore=5,
                         spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.1, leading=10.2),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.1, leading=10.2, textColor=colors.white),
    "ref": ParagraphStyle("ref", fontName="DV", fontSize=8.2, leading=10.6, leftIndent=12, firstLineIndent=-12,
                          spaceAfter=2.5),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.2, leading=12.6, leftIndent=11, bulletIndent=2,
                             spaceAfter=2.5),
}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True, bold_rows=()):
    data = [[P(c, "cellb" if (header and i == 0) else "cell") if isinstance(c, str) else c for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT", repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), INK))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F6F7FB")))
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#FAEBDD")))
    t.setStyle(TableStyle(st))
    return t


def callout(text):
    t = Table([[P(text)]], colWidths=[17.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


# ------------------------------------------------------------------------------ figures
def architecture_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    ink, saf, teal, tint = "#1B1F3B", "#E07A1F", "#1A936F", "#EEF0F7"

    def box(x, y, w, h, txt, fc, tc="white", fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#5F647A",
                                     lw=1.4))

    box(0.05, 2.05, 1.25, 0.85, "Meme caption\n(Bengali / mixed)", saf, bold=True, fs=8.5)
    box(0.05, 0.4, 1.25, 0.85, "Meme image\n(JPG / PNG)", saf, bold=True, fs=8.5)
    box(1.6, 2.05, 1.6, 0.85, "Normalise +\nscript ID", "#2E3463", fs=8.5)
    box(1.6, 0.4, 1.6, 0.85, "Validate +\nresize 224²", "#2E3463", fs=8.5)
    box(3.5, 2.05, 1.9, 0.85, "XLM-R base\n(top 4 layers tuned)", ink, fs=8.5)
    box(3.5, 0.4, 1.9, 0.85, "ResNet-50\n(layer4 tuned)", ink, fs=8.5)
    box(5.75, 0.75, 1.75, 1.8, "Fusion\nconcat (256+256)\nor cross-attention\ntext ⇄ 49 regions", teal, fs=8.5)
    box(7.85, 1.15, 1.0, 1.0, "MLP +\nsoftmax", tint, tc=ink, fs=8.5)
    box(9.05, 1.15, 0.9, 1.0, "pos\nneu\nneg", tint, tc=ink, fs=8.5, bold=True)
    for y in (2.47, 0.82):
        arrow(1.3, y, 1.6, y)
        arrow(3.2, y, 3.5, y)
    arrow(5.4, 2.47, 5.75, 2.1)
    arrow(5.4, 0.82, 5.75, 1.2)
    arrow(7.5, 1.65, 7.85, 1.65)
    arrow(8.85, 1.65, 9.05, 1.65)
    ax.text(5.0, 3.15, "Explainability: word occlusion / Integrated Gradients on the caption · Grad-CAM on ResNet layer4",
            ha="center", va="center", fontsize=8.5, color="#5F647A", style="italic")
    fig.tight_layout(pad=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def comparison_figure(T: dict, Sd: dict | None, path: Path) -> None:
    exps = ["text", "image", "multimodal_concat", "multimodal_attention"]
    names = ["Text-only", "Image-only", "MM concat", "MM cross-attn"]
    fig, ax = plt.subplots(figsize=(10, 3.0))
    x = range(len(exps))
    w = 0.38 if Sd else 0.6
    v1 = [T[e]["macro_f1"] for e in exps]
    err = [[T[e]["macro_f1"] - T[e]["macro_f1_ci95"][0] for e in exps],
           [T[e]["macro_f1_ci95"][1] - T[e]["macro_f1"] for e in exps]]
    b1 = ax.bar([i - (w / 2 if Sd else 0) for i in x], v1, w, yerr=err, capsize=3, color="#1B1F3B",
                label="Random split (seen pages)")
    bars = [b1]
    if Sd:
        v2 = [Sd[e]["macro_f1"] for e in exps]
        err2 = [[Sd[e]["macro_f1"] - Sd[e]["macro_f1_ci95"][0] for e in exps],
                [Sd[e]["macro_f1_ci95"][1] - Sd[e]["macro_f1"] for e in exps]]
        bars.append(ax.bar([i + w / 2 for i in x], v2, w, yerr=err2, capsize=3, color="#E07A1F",
                           label="Unseen source pages"))
    for bs in bars:
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, 0.02, f"{b.get_height():.3f}", ha="center", va="bottom",
                    color="white", fontsize=8.5, fontweight="bold")
    base = T["baseline_majority"]["macro_f1"]
    ax.axhline(base, color="#8A8F99", ls="--", lw=1, label=f"Majority-class baseline ({base:.3f})")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def dir_size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2 ** 20


# ------------------------------------------------------------------------------ document
def main() -> int:
    stats = load(ROOT / "data" / "processed" / "stats.json")
    sd_stats = load(ROOT / "data" / "processed_source_disjoint" / "stats.json")
    mc = load(ROOT / "artifacts" / "reports" / "comparison" / "comparison.json")
    sc = load(ROOT / "artifacts" / "reports_source_disjoint" / "comparison" / "comparison.json")
    if not (stats and mc):
        print("Run the pipeline first (data/processed/stats.json and the comparison report are required).")
        return 1
    T = {r["experiment"]: r for r in mc["table"]}
    Sd = {r["experiment"]: r for r in sc["table"]} if sc else None
    M = {e: load(ROOT / "artifacts" / "reports" / e / "metrics.json") for e in
         ("text", "image", "multimodal_concat", "multimodal_attention")}
    tests, sdt = mc["paired_bootstrap"], (sc or {}).get("paired_bootstrap", {})
    ab = mc.get("modality_ablation", {})
    ckpt = {e: dir_size_mb(ROOT / "artifacts" / "checkpoints" / e) for e in M}
    expl = load(ROOT / "artifacts" / "reports" / "explanations" / "explanations.json")

    architecture_figure(FIG / "architecture.png")
    comparison_figure(T, Sd, FIG / "results_seen_vs_unseen.png")

    n_tr, n_va, n_te = (stats["split_sizes"][k] for k in ("train", "val", "test"))
    ld = stats["label_distribution_total"]
    N = stats["usable_records"]
    pct = lambda k: f"{ld[k] / N:.1%}"
    att_txt, cat_txt = tests.get("multimodal_attention_vs_text"), tests.get("multimodal_concat_vs_text")
    f3 = lambda v: f"{v:.3f}"

    story = []

    # ============================================================== PAGE 1 — Proposed solution
    story += [P("1. Proposed Solution", "h1"),
              P("1.1 Problem", "h2"),
              P("Indian social-media content is multilingual, code-mixed and visual. A meme's sentiment often lives in "
                "the <i>interaction</i> between its caption and its picture (a cheerful line over a grim image can be "
                "sarcasm), yet most sentiment tools read only text and are built for English. We need a system that "
                "reads an Indian-language meme's caption <b>and</b> image, classifies it as <b>positive, neutral or "
                "negative</b>, and shows <i>why</i>."),
              P("1.2 Solution overview", "h2"),
              P("<b>IndicSent-M</b> is an end-to-end multimodal pipeline. The caption is encoded by a multilingual "
                "Transformer (XLM-RoBERTa base); the image by a pretrained CNN (ResNet-50). The two representations are "
                "projected to a shared 256-d space and <b>fused</b> — either by concatenation (baseline) or by "
                "bidirectional <b>cross-attention</b>, in which every caption token attends over the 49 regions of the "
                "image feature grid and vice versa — before a small classifier outputs the three class probabilities. "
                "Text-only and image-only models are trained with the identical recipe so the value of fusion can be "
                "measured, not assumed."),
              Image(str(FIG / "architecture.png"), width=17.4 * cm, height=17.4 * cm * 3.3 / 10),
              P("Figure 1. IndicSent-M architecture. Both inputs are mandatory for the multimodal models: the code "
                "refuses a missing modality, and unit tests verify that changing only the image or only the text "
                "changes the output.", "small"),
              P("1.3 Solution components", "h2")]
    story += bullets([
        f"<b>Data pipeline.</b> Loads the MemoSen Bengali meme dataset ({stats['raw_records']:,} annotations), validates "
        f"every image, normalises text conservatively (keeps emojis, negations, casing and the ZWJ/ZWNJ joiners Indian "
        f"scripts need), identifies scripts (Bengali / Latin / code-mixed), removes duplicates and builds a "
        f"<b>leakage-safe</b> split: {n_tr:,} train / {n_va:,} validation / {n_te:,} test memes.",
        "<b>Three model families.</b> Text-only (XLM-R), image-only (ResNet-50) and multimodal (two fusion variants), "
        "partially fine-tuned so they train on an 8 GB laptop.",
        "<b>Explainability.</b> Word-level attributions (leave-one-word-out occlusion; Integrated Gradients with an "
        "automatic completeness check) and Grad-CAM heatmaps on the image.",
        "<b>Application.</b> A Streamlit web app (analysis, model comparison, dataset and methodology pages) and a "
        "FastAPI REST service; Docker file for deployment; automated tests.",
    ])
    story.append(PageBreak())

    # ============================================================== PAGE 2 — Solution design
    story += [P("1.4 How a prediction is made", "h2")]
    story += bullets([
        "<b>Input checks</b> — the caption must be non-empty (text/multimodal modes); the image must decode, be ≤ 10 MB, "
        "≤ 40 MP and ≥ 32 px per side. Unsupported scripts, emoji-only text and truncation produce warnings.",
        "<b>Encoding</b> — caption → XLM-R tokens (max 96; the 95th-percentile caption is "
        f"{stats['token_length']['p95']:.0f} tokens) → mean-pooled 768-d vector and token sequence; image → 224×224 → "
        "ResNet-50 → 2048-d vector and 7×7 region grid.",
        "<b>Fusion and classification</b> — concat or cross-attention → MLP → softmax over positive / neutral / negative.",
        "<b>Output</b> — label, confidence (explicitly labelled as the predicted-class probability, not a guarantee), all "
        "three probabilities, highlighted words and a Grad-CAM overlay.",
    ])
    story += [P("1.5 Key design decisions", "h2"),
              table([["Decision", "Choice", "Reason"],
                     ["Text encoder", "XLM-R base (not IndicBERT)", "Pre-training data covers Bengali and romanised "
                      f"Bengali/Hindi; {stats['code_mixed_fraction']:.0%} of captions mix Bengali and Latin script"],
                     ["Image encoder", "ResNet-50, ImageNet weights", "Strong, small, Grad-CAM-compatible; runs on CPU"],
                     ["Fine-tuning", "Top 4 XLM-R layers + ResNet layer4", "Fits 8 GB unified memory; limits over-"
                      "fitting on ~3 k memes; frozen BatchNorm kept in eval mode"],
                     ["Fusion", "Concatenation and cross-attention", "Simple baseline vs. token-to-region interaction"],
                     ["Class imbalance", "√-inverse-frequency loss weights", f"Labels are {pct('negative')} negative, "
                      f"{pct('positive')} positive, {pct('neutral')} neutral"],
                     ["Model selection", "Best validation macro-F1, early stopping", "Test set never used for "
                      "choices"],
                     ["Text cleaning", "Conservative normalisation", "Emojis, negation, casing and joiners carry "
                      "sentiment or script meaning"],
                     ["Checkpoints", "Trained weights only (frozen rebuilt)", "~60–200 MB instead of 1.1 GB+; loader "
                      "refuses partially random models"]],
                    [3.0, 5.0, 9.4]),
              P("1.6 Evaluation design", "h2")]
    story += bullets([
        "<b>Fair comparison</b> — all models use the same split, preprocessing, optimiser, schedule, epochs budget and "
        "selection rule; only the architecture changes.",
        "<b>Primary metric: macro-F1</b> — with this imbalance, always predicting “negative” already gives "
        f"{T['baseline_majority']['accuracy']:.1%} accuracy but only {T['baseline_majority']['macro_f1']:.3f} macro-F1.",
        "<b>Uncertainty</b> — 95 % bootstrap confidence intervals and paired bootstrap tests between models.",
        "<b>Modality ablation</b> — re-test the fused models with a neutral image or without the caption to prove both "
        "inputs are used.",
        "<b>Robustness</b> — a second split that holds out whole <i>source pages</i>, because we found the page a meme "
        "comes from strongly predicts its label (§4.2).",
    ])
    story.append(callout("<b>Leakage control.</b> Near-duplicate memes (same image hash or the same long caption) are "
                         "grouped with union-find and each group is kept inside a single split; "
                         f"{stats['groups_spanning_splits']} groups span splits. A unit test checks that two runs "
                         "produce the identical split."))
    story += [P("1.7 Implementation status", "h2"),
              table([["Component", "Status"],
                     ["Data pipeline, leakage-safe and source-disjoint splits", "Implemented and tested"],
                     ["Text-only, image-only, concat and cross-attention models", "Implemented, trained, evaluated"],
                     ["Statistics (bootstrap CIs, paired tests), modality ablation", "Implemented"],
                     ["Occlusion, Integrated Gradients (verified), Grad-CAM", "Implemented"],
                     ["Streamlit app, FastAPI service, automated tests", "Implemented and tested"],
                     ["Docker image", "Dockerfile written; not built in this environment"],
                     ["OCR, Hinglish data, calibration, multiple seeds", "Future work"]],
                    [10.4, 7.0])]
    story.append(PageBreak())

    # ============================================================== PAGE 3 — Technical stack
    story += [P("2. Technical Stack", "h1"),
              P("Every component is open-source and runs on a laptop. Versions are those the reported results were "
                "produced with (pinned in <font face='DV-B'>requirements.txt</font>).", "body"),
              table([["Layer", "Technology (version)", "Role in IndicSent-M"],
                     ["Language", "Python 3.13 (3.10+ supported)", "All code"],
                     ["Deep learning", "PyTorch 2.7.1, torchvision 0.22.1", "Models, training loop, MPS/CUDA/CPU"],
                     ["NLP models", "Hugging Face Transformers 4.57.6, SentencePiece", "XLM-RoBERTa base encoder and "
                      "tokenizer (FacebookAI/xlm-roberta-base)"],
                     ["Computer vision", "torchvision ResNet-50 (IMAGENET1K_V2)", "Image encoder, Grad-CAM target "
                      "layer"],
                     ["Fusion", "PyTorch nn.MultiheadAttention, MLP", "Concatenation and cross-attention fusion"],
                     ["Data processing", "pandas, NumPy, Pillow, openpyxl", "Loading, validation, hashing, caching, "
                      "splits"],
                     ["Evaluation", "scikit-learn, SciPy", "Metrics, confusion matrices, stratified and "
                      "source-disjoint splits"],
                     ["Explainable AI", "Own implementations", "Occlusion, Integrated Gradients (completeness "
                      "check), Grad-CAM"],
                     ["Visualisation", "Matplotlib, Seaborn", "Dataset figures, result plots, explanation gallery"],
                     ["Frontend", "Streamlit 1.65", "Five-page web app"],
                     ["Backend", "FastAPI + Uvicorn", "REST endpoint /predict, /health"],
                     ["Experiment tracking", "CSV / JSON logs", "Per-epoch history, metadata with config and "
                      "environment"],
                     ["Testing", "pytest (44 tests), Streamlit AppTest", "Data, models, inference, explainers, API, "
                      "app"],
                     ["Deployment", "Docker (python:3.11-slim, CPU wheels)", "Containerised app/API; checkpoints "
                      "mounted"],
                     ["Documents", "python-pptx, ReportLab", "Slides and this report generated from results"]],
                    [3.0, 5.6, 8.8]),
              P("2.1 Hardware and runtime", "h2"),
              P("All results were produced on an <b>Apple M1 with 8 GB unified memory</b> using PyTorch's MPS backend; "
                "the same code runs on NVIDIA GPUs (CUDA) and on CPU without changes (automatic device selection). "
                "Pretrained weights are downloaded once (≈1.1 GB XLM-R, ≈100 MB ResNet-50) and the system then works "
                "fully offline."),
              P("2.2 Code organisation", "h2")]
    story += bullets([
        "<font face='DV-B'>src/data</font> — download, preparation, text processing, image validation, PyTorch dataset.",
        "<font face='DV-B'>src/models</font> — text, image and multimodal models; factory and checkpoint loader.",
        "<font face='DV-B'>src/training</font>, <font face='DV-B'>src/evaluation</font>, "
        "<font face='DV-B'>src/explainability</font> — shared trainer, metrics and statistics, explainers.",
        "<font face='DV-B'>app.py</font>, <font face='DV-B'>src/api.py</font>, "
        "<font face='DV-B'>scripts/</font> — UI, REST API, one-command pipeline, environment check, document builders.",
        "<font face='DV-B'>configs/</font> — one YAML file holds every hyper-parameter; experiment variants inherit from it.",
    ])
    story += [P("2.3 Why this stack", "h2")]
    story += bullets([
        "<b>Standard and teachable</b> — PyTorch and Hugging Face are the de-facto research stack; every model can be "
        "swapped by changing one config line.",
        "<b>Multilingual by default</b> — XLM-R's 250 k SentencePiece vocabulary covers Indic scripts and romanised "
        "text without custom tokenisation.",
        "<b>Lightweight serving</b> — Streamlit gives an interactive demo in pure Python; FastAPI exposes the same "
        "predictor to other systems; Docker packages both.",
        "<b>No heavy extras</b> — explainability methods are implemented directly (≈200 lines) instead of adding large "
        "XAI libraries, keeping dependencies small and versions compatible.",
    ])
    story.append(PageBreak())

    # ============================================================== PAGE 4 — Feasibility & viability
    def lat(e):
        return f"{M[e]['latency']['median_ms']:.0f} ms"

    names = {"text": "Text-only", "image": "Image-only", "multimodal_concat": "Multimodal (concat)",
             "multimodal_attention": "Multimodal (cross-attn)"}
    story += [P("3. Feasibility and Viability", "h1"),
              P("3.1 Technical feasibility — demonstrated, not estimated", "h2"),
              P("All four models were trained and evaluated end-to-end on a consumer laptop. Measured costs:"),
              table([["Model", "Trainable / total params", "Training time", "Epochs (best)", "Checkpoint",
                      "Latency / meme*"]] +
                    [[names[e], f"{M[e]['parameters']['trainable'] / 1e6:.1f} M / {M[e]['parameters']['total'] / 1e6:.0f} M",
                      f"{M[e]['training_seconds'] / 60:.1f} min", f"{M[e]['epochs_run']} ({M[e]['checkpoint_epoch']})",
                      f"{ckpt[e]:.0f} MB", lat(e)] for e in M],
                    [4.0, 3.6, 2.4, 2.3, 2.2, 2.9]),
              P("*Median at batch size 1 on the M1 GPU (MPS), including image loading, preprocessing and tokenisation.",
                "small"),
              P("3.2 Data feasibility", "h2"),
              table([["Annotations", "Usable memes", "Train / val / test", "Positive / neutral / negative",
                      "Code-mixed captions"],
                     [f"{stats['raw_records']:,}", f"{N:,}", f"{n_tr:,} / {n_va:,} / {n_te:,}",
                      f"{pct('positive')} / {pct('neutral')} / {pct('negative')}", f"{stats['code_mixed_fraction']:.1%}"]],
                    [2.8, 2.8, 3.6, 4.6, 3.6]),
              Spacer(1, 4),
              P(f"MemoSen (LREC 2022) is publicly downloadable and provides images, transcribed captions and 3-class labels; "
                f"{N:,} of {stats['raw_records']:,} annotations are usable after validation. It carries no explicit licence, "
                "so it is used for non-commercial research with citation and is not redistributed. Hinglish data "
                "(Memotion 3) can be added through a generic CSV adapter once its registration is completed."),
              P("3.3 Economic viability", "h2")]
    story += bullets([
        "<b>Zero licence cost</b> — every library and both pretrained models are open-source.",
        "<b>No special hardware</b> — training all four models took about "
        f"{sum(M[e]['training_seconds'] for e in M) / 60:.0f} minutes of compute on an 8 GB laptop; a cloud GPU would "
        "cost well under one hour of rental.",
        "<b>Cheap to serve</b> — CPU-compatible inference, small checkpoints, a single Docker container.",
    ])
    story += [P("3.4 Operational viability", "h2")]
    story += bullets([
        "Starts even without trained models (shows setup steps; never serves random-weight predictions).",
        "Handles corrupt, oversized or missing images, empty captions and unsupported scripts with clear messages.",
        "Reproducible: fixed seed, pinned versions, one-command pipeline, config and environment saved with every model.",
    ])
    story += [P("3.5 Risks and mitigations", "h2"),
              table([["Risk", "Mitigation in this project"],
                     ["Small, imbalanced dataset (neutral ≈ 7 %)", "Class-weighted loss, macro-F1 as primary metric, "
                      "early stopping"],
                     ["Shortcut learning (source page predicts label)", "Detected; source-disjoint evaluation reported "
                      "alongside the random split"],
                     ["Over-confident predictions", "Confidence labelled as a probability, not a guarantee; calibration "
                      "listed as future work"],
                     ["Misleading explanations", "IG completeness checked and hidden when it fails; limitations shown "
                      "in the app"],
                     ["Out-of-domain languages", "Script detection warns; documentation states the training language"]],
                    [6.4, 11.0]),
              Spacer(1, 6),
              callout("<b>Verdict.</b> The solution is <b>technically feasible</b> (built, trained and tested on an 8 GB "
                      "laptop), <b>economically viable</b> (open-source, no paid services, CPU-servable) and "
                      "<b>operationally robust</b> for research and demonstration use. Production use would first need "
                      "more diverse training data, probability calibration and human review of predictions.")]
    story.append(PageBreak())

    # ============================================================== PAGE 5 — Improvements & benefits
    story += [P("4. Improvements and Benefits", "h1"),
              P("4.1 Measured improvement from multimodal fusion", "h2"),
              Image(str(FIG / "results_seen_vs_unseen.png"), width=17.4 * cm, height=17.4 * cm * 3.0 / 10),
              P(f"Figure 2. Test macro-F1 with 95 % bootstrap intervals. Random split: {T['text']['n_test']} test memes; "
                + (f"unseen source pages: {Sd['text']['n_test']} test memes (different test sets)." if Sd else ""),
                "small"),
              table([["Model (random split)", "Accuracy", "Macro-F1 [95 % CI]", "F1 neutral", "Δ macro-F1 vs text"]] +
                    [[names[e], f3(T[e]["accuracy"]),
                      f"{f3(T[e]['macro_f1'])} [{T[e]['macro_f1_ci95'][0]:.2f}, {T[e]['macro_f1_ci95'][1]:.2f}]",
                      f3(T[e]["f1_neutral"]),
                      "–" if e == "text" else f"{T[e]['macro_f1'] - T['text']['macro_f1']:+.3f}"] for e in M],
                    [4.6, 2.2, 4.2, 2.4, 4.0], bold_rows=(4,))]
    imp = [
        f"<b>Fusion beats both single modalities.</b> Cross-attention fusion reaches {f3(T['multimodal_attention']['macro_f1'])} "
        f"macro-F1 vs {f3(T['text']['macro_f1'])} (text-only) and {f3(T['image']['macro_f1'])} (image-only); accuracy "
        f"rises from {T['text']['accuracy']:.1%} to {T['multimodal_attention']['accuracy']:.1%}. The gain over text "
        f"is {att_txt['observed_diff']:+.3f} (95 % CI {att_txt['ci95'][0]:+.3f} to {att_txt['ci95'][1]:+.3f}) — real but "
        "modest; over image-only it is large and clearly significant.",
    ]
    if ab:
        a = ab["multimodal_attention"]
        imp.append(f"<b>Both inputs matter.</b> Neutralising the image or removing the caption flips "
                   f"{a['image_neutralised']['prediction_changed_rate']:.0%} / {a['text_removed']['prediction_changed_rate']:.0%} "
                   "of the cross-attention model's predictions.")
    story += bullets(imp)
    imp = []
    story += [P("4.2 Robustness check and the improvement it motivated", "h2")]
    if Sd:
        k = "multimodal_concat_vs_text"
        imp.append(f"<b>Honest generalisation estimate.</b> The source page alone predicts the label "
                   f"({T['baseline_source_prior']['accuracy']:.0%} accuracy without reading the meme). On unseen pages, "
                   f"scores drop to {f3(min(Sd[e]['macro_f1'] for e in M))}–{f3(max(Sd[e]['macro_f1'] for e in M))}; "
                   f"concatenation fusion stays best and significantly ahead of text-only "
                   f"({sdt[k]['observed_diff']:+.3f}, CI {sdt[k]['ci95'][0]:+.3f} to {sdt[k]['ci95'][1]:+.3f}).")
    story += bullets(imp)
    story += [P("4.3 Benefits", "h2")]
    story += bullets([
        "<b>Researchers</b> — a reproducible, tested baseline for Indian-language multimodal sentiment, with "
        "statistical tests and a documented dataset shortcut that future work must control for.",
        "<b>Platforms and analysts</b> — sentiment that accounts for the image, not just the text, in code-mixed posts; "
        "explanations make decisions reviewable by humans.",
        "<b>Students and educators</b> — an end-to-end reference implementation (data → models → XAI → app → Docker).",
        "<b>Low-resource languages</b> — shows a laptop-scale route for languages with little labelled data.",
    ])
    story += [P("4.4 Planned improvements", "h2"),
              P("Hinglish and other Indian-language memes (Memotion 3) · OCR so users need not type captions · CLIP-style "
                "vision-language encoders · source-balanced or page-adversarial training against the shortcut · more "
                "neutral examples / focal loss · several training seeds and probability calibration · user study of the "
                "explanations.")]
    story.append(PageBreak())

    # ============================================================== PAGE 6 — Research & references
    story += [P("5. Research and References", "h1"),
              P("5.1 Research questions and findings", "h2")]
    rq = [
        ["RQ", "Finding (from the experiments)"],
        ["Does text + image beat a single modality?",
         f"Yes. Best random-split model: cross-attention fusion ({f3(T['multimodal_attention']['macro_f1'])}); both fusion "
         f"models significantly beat image-only; vs text-only the gain is {att_txt['observed_diff']:+.3f} (attention) and "
         f"{cat_txt['observed_diff']:+.3f} (concat, not significant)."],
        ["Concatenation or cross-attention?",
         "Indistinguishable on the random split"
         + (f"; on unseen pages concatenation is significantly better "
            f"({-sdt['multimodal_attention_vs_multimodal_concat']['observed_diff']:+.3f})." if sdt else ".")],
        ["Do scores reflect sentiment understanding?",
         f"Partly. A content-free source-page prior reaches {T['baseline_source_prior']['accuracy']:.1%} accuracy; "
         "every model loses 0.14–0.25 macro-F1 on unseen pages (92 % of positive training memes come from one page)."
         if Sd else "Source-page prior is a strong shortcut."],
        ["Can predictions be explained reliably?",
         "Occlusion is exact; Integrated Gradients failed its completeness check for all text-only gallery examples"
         + (f" ({sum(1 for r in expl['examples'] for k, v in r.items() if isinstance(v, dict) and v.get('integrated_gradients') and v['integrated_gradients']['reliable'])}"
            f"/{sum(1 for r in expl['examples'] for k, v in r.items() if isinstance(v, dict) and v.get('integrated_gradients'))} "
            "passed overall), so it is shown only when verified." if expl else ".")],
    ]
    story.append(table(rq, [4.6, 12.8]))
    story += [P("5.2 Research contribution", "h2"),
              P("(i) A leakage-safe, fully reproducible comparison of text, image and two fusion models for Bengali meme "
                "sentiment with confidence intervals and paired significance tests; (ii) identification and "
                "quantification of a source-page shortcut in MemoSen and a source-disjoint protocol to measure it; "
                "(iii) an explainability layer that verifies Integrated Gradients' completeness before showing it; "
                "(iv) an open, tested application. Full details: <font face='DV-B'>docs/research_results.md</font>."),
              P("5.3 References", "h2")]
    refs = [
        "E. Hossain, O. Sharif, M. M. Hoque. MemoSen: A Multimodal Dataset for Sentiment Analysis of Memes. "
        "<i>Proc. LREC 2022</i>, pp. 1542–1554. aclanthology.org/2022.lrec-1.165",
        "A. Conneau, K. Khandelwal, N. Goyal, et al. Unsupervised Cross-lingual Representation Learning at Scale. "
        "<i>ACL 2020</i>, pp. 8440–8451. aclanthology.org/2020.acl-main.747",
        "K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. <i>CVPR 2016</i>, pp. 770–778.",
        "A. Vaswani, N. Shazeer, N. Parmar, et al. Attention Is All You Need. <i>NeurIPS 2017</i>.",
        "D. Kakwani, A. Kunchukuttan, S. Golla, et al. IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and "
        "Pre-trained Multilingual Language Models for Indian Languages. <i>Findings of EMNLP 2020</i>, pp. 4948–4961.",
        "S. Mishra, S. Suryavardan, P. Patwa, et al. Memotion 3: Dataset on Sentiment and Emotion Analysis of Codemixed "
        "Hindi-English Memes. arXiv:2303.09892, 2023.",
        "C. Sharma, D. Bhageria, W. Scott, et al. SemEval-2020 Task 8: Memotion Analysis – the Visuo-Lingual Metaphor! "
        "<i>Proc. SemEval-2020</i>, pp. 759–773.",
        "M. Sundararajan, A. Taly, Q. Yan. Axiomatic Attribution for Deep Networks. <i>ICML 2017</i>.",
        "R. R. Selvaraju, M. Cogswell, A. Das, et al. Grad-CAM: Visual Explanations from Deep Networks via "
        "Gradient-based Localization. <i>ICCV 2017</i>.",
        "J. Li, W. Monroe, D. Jurafsky. Understanding Neural Networks through Representation Erasure. "
        "arXiv:1612.08220, 2016.",
        "T. Berg-Kirkpatrick, D. Burkett, D. Klein. An Empirical Investigation of Statistical Significance in NLP. "
        "<i>EMNLP-CoNLL 2012</i>, pp. 995–1005.",
        "I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. <i>ICLR 2019</i>.",
        "D. Kiela, H. Firooz, A. Mohan, et al. The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes. "
        "<i>NeurIPS 2020</i>.",
    ]
    story += [Paragraph(f"[{i}] {r}", S["ref"]) for i, r in enumerate(refs, 1)]
    story.insert(story.index(next(x for x in story if isinstance(x, Paragraph) and x.getPlainText() == "5.3 References")),
                 P("<b>Method followed:</b> literature and dataset survey → leakage-safe data preparation → identical "
                   "training of four models → evaluation with uncertainty → ablation → robustness split → explanation "
                   "analysis. Every number in this report is read from the saved evaluation files."))
    story.append(Spacer(1, 4))
    story.append(P("Entries [1], [2], [5]–[7], [10], [11] were checked against the ACL Anthology / arXiv; the others are "
                   "standard references cited from the published record (see docs/references.md).", "small"))

    # ------------------------------------------------------------------ page decoration
    def first_page(c, doc):
        c.saveState()
        c.setFillColor(INK)
        c.rect(0, A4[1] - 3.1 * cm, A4[0], 3.1 * cm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("DV-B", 17)
        c.drawString(1.8 * cm, A4[1] - 1.45 * cm, "IndicSent-M")
        c.setFont("DV", 10.5)
        c.drawString(1.8 * cm, A4[1] - 2.1 * cm,
                     "Multimodal Sentiment Analysis for Indian Languages Using Text and Images")
        c.setFillColor(SAFFRON)
        c.setFont("DV", 8.5)
        c.drawString(1.8 * cm, A4[1] - 2.65 * cm,
                     "Project report · proposed solution · technical stack · feasibility · benefits · research")
        footer(c, doc)
        c.restoreState()

    def footer(c, doc):
        c.setFont("DV", 7.5)
        c.setFillColor(MUTED)
        c.drawString(1.8 * cm, 1.0 * cm, "IndicSent-M · results generated from the project's evaluation outputs")
        c.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {doc.page} of 6")

    def later_pages(c, doc):
        c.saveState()
        footer(c, doc)
        c.restoreState()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.6 * cm, title="IndicSent-M Project Report",
                            author="IndicSent-M project", subject="Multimodal sentiment analysis for Indian languages")

    def build():
        # first page has a 3.1 cm title band: start the story below it
        story.insert(0, Spacer(1, 2.0 * cm))
        doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)

    build()
    from pypdf import PdfReader
    n = len(PdfReader(str(OUT)).pages)
    print(f"Wrote {OUT} ({n} pages)")
    return 0 if n == 6 else 2


if __name__ == "__main__":
    sys.exit(main())
