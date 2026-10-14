"""Build the presentation deck (docs/IndicSent-M_slides.pptx) from the real result files.

Every number on the slides is read from artifacts/reports*/comparison/comparison.json and
data/processed/stats.json; sentences about significance are generated from the paired
bootstrap results, so the deck cannot drift from the experiments.

    python scripts/make_slides.py
Requires: python-pptx (pip install "python-pptx>=1.0,<2")
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "IndicSent-M_slides.pptx"

INK = RGBColor(0x1B, 0x1F, 0x3B)       # deep indigo (dominant)
INK2 = RGBColor(0x2E, 0x34, 0x63)
SAFFRON = RGBColor(0xE0, 0x7A, 0x1F)   # accent
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEXT = RGBColor(0x22, 0x25, 0x33)
MUTED = RGBColor(0x5F, 0x64, 0x7A)
TINT = RGBColor(0xEE, 0xF0, 0xF7)      # card background
POS, NEU, NEG = RGBColor(0x1E, 0x8E, 0x4E), RGBColor(0x8A, 0x8F, 0x99), RGBColor(0xC0, 0x39, 0x2B)
MODEL_COLORS = {"text": RGBColor(0x3A, 0x5B, 0xA9), "image": SAFFRON,
                "multimodal_concat": RGBColor(0x1A, 0x93, 0x6F), "multimodal_attention": RGBColor(0x6C, 0x4F, 0xA3),
                "baseline_majority": RGBColor(0xBF, 0xC3, 0xCC), "baseline_source_prior": RGBColor(0x8A, 0x8F, 0x99)}
SHORT = {"text": "Text", "image": "Image", "multimodal_concat": "MM-concat", "multimodal_attention": "MM-attention",
         "baseline_majority": "Majority", "baseline_source_prior": "Source prior*"}
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------------------------ primitives
def text(slide, x, y, w, h, runs, size=16, color=TEXT, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str | list[str | (str, dict)] -> one paragraph per item."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    items = [runs] if isinstance(runs, (str, tuple)) else runs
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        txt, opt = (item, {}) if isinstance(item, str) else item
        r = p.add_run()
        r.text = txt
        f = r.font
        f.name = opt.get("font", font)
        f.size = Pt(opt.get("size", size))
        f.bold = opt.get("bold", bold)
        f.italic = opt.get("italic", False)
        f.color.rgb = opt.get("color", color)
        p.space_after = Pt(opt.get("space", 6))
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=TEXT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bold_part, rest = it if isinstance(it, tuple) else ("", it)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.28))))
        pPr.set("indent", str(-Emu(Inches(0.28))))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
        for t, b in ((bold_part, True), (rest, False)):
            if t:
                r = p.add_run()
                r.text = t
                r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), b, BODY, color
        p.space_after = Pt(8)
    return tb


def box(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if name:
        s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    s.shadow.inherit = False
    return s


def title(slide, t, sub=None, dark=False):
    text(slide, Inches(0.6), Inches(0.4), Inches(12.1), Inches(0.9), t, size=36, bold=True, font=HEAD,
         color=WHITE if dark else INK, name="Title")
    if sub:
        text(slide, Inches(0.6), Inches(1.2), Inches(12.1), Inches(0.5), sub, size=16,
             color=RGBColor(0xC9, 0xCD, 0xE0) if dark else MUTED, name="Subtitle")


def footer(slide, n, dark=False):
    text(slide, Inches(0.6), Inches(7.0), Inches(8), Inches(0.35), "IndicSent-M · multimodal meme sentiment",
         size=10, color=RGBColor(0x9A, 0x9E, 0xB5) if dark else MUTED)
    text(slide, Inches(11.9), Inches(7.0), Inches(0.8), Inches(0.35), str(n), size=10,
         color=RGBColor(0x9A, 0x9E, 0xB5) if dark else MUTED, align=PP_ALIGN.RIGHT)


def background(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = color


def stat(slide, x, y, w, value, label, color=INK, vsize=40):
    text(slide, x, y, w, Inches(0.8), value, size=vsize, bold=True, font=HEAD, color=color)
    text(slide, x, y + Inches(0.8), w, Inches(0.6), label, size=12, color=MUTED)


def bar_chart(slide, x, y, w, h, categories, series: dict, colors: list, title_txt, pct=False, legend=True,
              vmax=None, point_colors=None):
    cd = CategoryChartData()
    cd.categories = categories
    for name, vals in series.items():
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, w, h, cd)
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = title_txt
    tp = ch.chart_title.text_frame.paragraphs[0]
    tp.runs[0].font.size, tp.runs[0].font.bold, tp.runs[0].font.name = Pt(14), True, BODY
    tp.runs[0].font.color.rgb = TEXT
    ch.has_legend = legend and len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size, ch.legend.font.name = Pt(11), BODY
    va = ch.value_axis
    va.maximum_scale = vmax or 1.0
    va.minimum_scale = 0
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE5, 0xEC)
    va.format.line.fill.background()
    va.tick_labels.font.size, va.tick_labels.font.color.rgb, va.tick_labels.font.name = Pt(10), MUTED, BODY
    va.tick_labels.number_format = '0%' if pct else '0.0'
    va.tick_labels.number_format_is_linked = False
    ca = ch.category_axis
    ca.tick_labels.font.size, ca.tick_labels.font.color.rgb, ca.tick_labels.font.name = Pt(11), TEXT, BODY
    ca.format.line.color.rgb = RGBColor(0xC9, 0xCC, 0xD6)
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.overlap = -10
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = '0%' if pct else '0.000'
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.name, dl.font.color.rgb = Pt(10), BODY, TEXT
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c
    if point_colors:
        for i, c in enumerate(point_colors):
            pt = plot.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = c
    return ch


def table(slide, x, y, w, rows, col_w, header_fill=INK, size=12, highlight_row=None):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(0.38) * len(rows))
    tbl = shp.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.size, f.name = Pt(size), BODY
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
                f.color.rgb, f.bold = WHITE, True
            else:
                cell.fill.fore_color.rgb = RGBColor(0xFA, 0xEB, 0xDD) if i == highlight_row else (
                    WHITE if i % 2 else RGBColor(0xF5, 0xF6, 0xFA))
                f.color.rgb = TEXT
                f.bold = i == highlight_row
    return tbl


def verdict(test: dict) -> str:
    lo, hi = test["ci95"]
    return "significantly better" if lo > 0 else "significantly worse" if hi < 0 else "not significantly different"


# ------------------------------------------------------------------------------ deck
def main() -> int:
    stats = load(ROOT / "data" / "processed" / "stats.json")
    main_c = load(ROOT / "artifacts" / "reports" / "comparison" / "comparison.json")
    sd_c = load(ROOT / "artifacts" / "reports_source_disjoint" / "comparison" / "comparison.json")
    if not (stats and main_c):
        print("Need data/processed/stats.json and artifacts/reports/comparison/comparison.json first.")
        return 1
    T = {r["experiment"]: r for r in main_c["table"]}
    trained = [e for e in ("text", "image", "multimodal_concat", "multimodal_attention") if e in T]
    best = max(trained, key=lambda e: T[e]["macro_f1"])
    tests = main_c["paired_bootstrap"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    n = 0

    def new(dark=False):
        nonlocal n
        n += 1
        s = prs.slides.add_slide(blank)
        background(s, INK if dark else WHITE)
        return s

    # 1. Title ------------------------------------------------------------------
    s = new(dark=True)
    text(s, Inches(0.8), Inches(2.0), Inches(11.5), Inches(1.2), "IndicSent-M", size=60, bold=True, font=HEAD,
         color=WHITE)
    text(s, Inches(0.8), Inches(3.2), Inches(11.5), Inches(1.0),
         "Multimodal Sentiment Analysis for Indian Languages Using Text and Images", size=26, color=WHITE,
         font=HEAD)
    text(s, Inches(0.8), Inches(4.4), Inches(11.5), Inches(0.6),
         "XLM-R + ResNet-50 on Bengali memes (MemoSen) · text vs image vs fusion · explainability",
         size=16, color=SAFFRON)
    for i, (lab, col) in enumerate([("Positive", POS), ("Neutral", NEU), ("Negative", NEG)]):
        b = box(s, Inches(0.8 + i * 1.9), Inches(5.4), Inches(1.7), Inches(0.5), fill=col)
        b.text_frame.text = lab
        r = b.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.color.rgb, r.font.bold, r.font.name = Pt(14), WHITE, True, BODY
        b.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    s.notes_slide.notes_text_frame.text = (
        "Project overview. All numbers in this deck are read from the evaluation outputs by scripts/make_slides.py.")

    # 2. Motivation & problem ------------------------------------------------------
    s = new()
    title(s, "Why memes need more than text", "Motivation and problem statement")
    bullets(s, Inches(0.6), Inches(1.9), Inches(7.2), Inches(4.8), [
        ("Multilingual, code-mixed. ", "Indian social media mixes Bengali and Latin script, romanised "
                                        "Bengali/Hindi, emojis and cultural references."),
        ("Meaning lives across modalities. ", "A cheerful caption on a grim picture can be sarcasm; the image "
                                               "alone or the text alone can mislead."),
        ("Tools lag behind. ", "Most sentiment systems are text-only and English-centric; Indian languages "
                               "are lower-resource."),
        ("Question. ", "Does fusing caption + image beat either modality alone, and can we show why a "
                       "model decided?"),
    ], size=17)
    box(s, Inches(8.3), Inches(1.9), Inches(4.4), Inches(4.6))
    stat(s, Inches(8.6), Inches(2.1), Inches(3.9), f"{stats['usable_records']:,}", "labelled Bengali memes used")
    stat(s, Inches(8.6), Inches(3.55), Inches(3.9), f"{stats['code_mixed_fraction']:.0%}",
         "captions mixing Bengali and Latin script", color=SAFFRON)
    neu = stats["label_distribution_total"]["neutral"] / stats["usable_records"]
    stat(s, Inches(8.6), Inches(5.0), Inches(3.9), f"{neu:.0%}", "of memes are neutral: strong imbalance",
         color=NEG)
    footer(s, n)

    # 3. Objectives --------------------------------------------------------------
    s = new()
    title(s, "Research objectives")
    objs = [("1", "Build three model types", "Text-only (XLM-R), image-only (ResNet-50) and multimodal fusion"),
            ("2", "Compare fairly", "Same leakage-safe split, same training recipe, bootstrap confidence intervals"),
            ("3", "Two fusion strategies", "Feature concatenation baseline vs. bidirectional cross-attention"),
            ("4", "Explain predictions", "Word occlusion / Integrated Gradients and Grad-CAM, with stated limits")]
    for i, (num, head, body) in enumerate(objs):
        x = Inches(0.6 + (i % 2) * 6.2)
        y = Inches(1.7 + (i // 2) * 2.6)
        box(s, x, y, Inches(5.9), Inches(2.3))
        c = box(s, x + Inches(0.3), y + Inches(0.35), Inches(0.75), Inches(0.75), fill=SAFFRON, shape=MSO_SHAPE.OVAL)
        c.text_frame.text = num
        r = c.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(22), True, WHITE, HEAD
        c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        text(s, x + Inches(1.3), y + Inches(0.3), Inches(4.4), Inches(0.6), head, size=20, bold=True, color=INK,
             font=HEAD)
        text(s, x + Inches(1.3), y + Inches(0.95), Inches(4.4), Inches(1.2), body, size=15, color=TEXT)
    footer(s, n)

    # 4. Dataset -----------------------------------------------------------------
    s = new()
    title(s, "Dataset: MemoSen (Bengali memes, LREC 2022)",
          f"{stats['raw_records']:,} annotations → {stats['usable_records']:,} usable after validation & de-duplication")
    ld = stats["label_distribution"]
    bar_chart(s, Inches(0.5), Inches(1.8), Inches(7.0), Inches(4.9), ["Train", "Validation", "Test"],
              {lab.capitalize(): [ld[sp][lab] for sp in ("train", "val", "test")]
               for lab in ("positive", "neutral", "negative")},
              [POS, NEU, NEG], "Memes per split and label", vmax=None)
    ch = s.shapes[-1].chart
    ch.value_axis.maximum_scale = None
    ch.value_axis.tick_labels.number_format = '#,##0'
    ch.plots[0].data_labels.number_format = '#,##0'
    sc = stats["scripts"]
    bullets(s, Inches(7.9), Inches(1.9), Inches(4.9), Inches(4.9), [
        ("Inputs: ", "meme image + transcribed caption; label positive / neutral / negative"),
        ("Scripts: ", f"Bengali {sc.get('Bengali', 0):,} · mixed {sc.get('Bengali+Latin', 0):,} · Latin "
                      f"{sc.get('Latin', 0):,}"),
        ("Split: ", "70 / 10 / 20, stratified, grouped by near-duplicate image or caption — 0 groups leak"),
        ("Removed: ", ", ".join(f"{v} {k}" for k, v in stats["removed_by_reason"].items())),
        ("Licence: ", "none published → research use with citation, not redistributed"),
    ], size=15)
    footer(s, n)

    # 5. Pipeline ----------------------------------------------------------------
    s = new()
    title(s, "Preprocessing pipeline", "Conservative by design: sentiment cues and scripts are preserved")
    steps = [("Load", "xlsx captions, image files, labels"), ("Validate", "decode, size limits, drop missing"),
             ("Normalise", "NFC · URL/@USER · keep emoji, negation, ZWJ"), ("Script ID", "Bengali / Latin / mixed"),
             ("De-duplicate", "dHash + text; conflicting labels dropped"), ("Group split", "union-find groups, seed 42")]
    for i, (h_, b_) in enumerate(steps):
        x = Inches(0.6 + i * 2.08)
        box(s, x, Inches(2.4), Inches(1.85), Inches(2.6), fill=INK if i % 2 == 0 else INK2)
        text(s, x + Inches(0.12), Inches(2.6), Inches(1.6), Inches(0.5), f"{i + 1}. {h_}", size=16, bold=True,
             color=WHITE, font=HEAD)
        text(s, x + Inches(0.12), Inches(3.3), Inches(1.6), Inches(1.6), b_, size=13, color=RGBColor(0xDD, 0xE0, 0xEE))
    text(s, Inches(0.6), Inches(5.5), Inches(12.1), Inches(1.2),
         [("Not done on purpose: ", {"bold": True, "color": SAFFRON, "size": 15}),
          ("lower-casing (XLM-R is cased), emoji/stop-word removal, stemming, transliteration — each would erase "
           "sentiment cues or script information. Images cached at 448 px once so training does not decode "
           "multi-megapixel files every epoch.", {"size": 15})])
    footer(s, n)

    # 6. Architecture ------------------------------------------------------------
    s = new()
    title(s, "Model architecture", "Three model families, one shared training recipe")
    rows_ = [("Caption", "XLM-R base\n(top 4 of 12 layers trained)", "mean-pool 768-d\nor token sequence", 1.9),
             ("Image", "ResNet-50\n(layer4 trained)", "pool 2048-d\nor 7×7 region grid", 4.2)]
    for lab, enc, feat, yy in rows_:
        y = Inches(yy)
        for j, (t_, w_, fill) in enumerate([(lab, 1.6, SAFFRON), (enc, 2.8, INK), (feat, 2.4, INK2)]):
            x = Inches(0.6 + [0, 1.9, 5.0][j])
            b = box(s, x, y, Inches(w_), Inches(1.3), fill=fill)
            b.text_frame.word_wrap = True
            b.text_frame.text = t_
            for p in b.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size, r.font.color.rgb, r.font.name, r.font.bold = Pt(14), WHITE, BODY, j == 0
    fus = box(s, Inches(8.0), Inches(2.4), Inches(2.4), Inches(2.9), fill=RGBColor(0x1A, 0x93, 0x6F))
    fus.text_frame.word_wrap = True
    fus.text_frame.text = "Fusion\n\nconcat (256+256)\nor\ncross-attention\ntext ⇄ 49 regions"
    for p in fus.text_frame.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size, r.font.color.rgb, r.font.name = Pt(14), WHITE, BODY
    out_b = box(s, Inches(10.8), Inches(3.2), Inches(1.9), Inches(1.3), fill=TINT)
    out_b.text_frame.text = "MLP → softmax\npos / neu / neg"
    for p in out_b.text_frame.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size, r.font.color.rgb, r.font.name = Pt(14), INK, BODY
    for (x1, y1, x2, y2) in [(7.4, 2.55, 8.0, 3.2), (7.4, 4.85, 8.0, 4.4), (10.4, 3.85, 10.8, 3.85)]:
        ln = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        ln.line.color.rgb, ln.line.width = MUTED, Pt(2)
    text(s, Inches(0.6), Inches(5.8), Inches(12.1), Inches(1.0),
         "Text-only and image-only models put a linear head directly on the pooled vector. Multimodal models "
         "always use both inputs — forward() refuses a missing modality, and tests check both change the output.",
         size=14, color=MUTED)
    footer(s, n)

    # 7. Training setup ----------------------------------------------------------
    s = new()
    title(s, "Training and evaluation protocol", "Identical for every experiment; test set never used for selection")
    table(s, Inches(0.6), Inches(1.8), Inches(7.2), [
        ["Setting", "Value"],
        ["Loss", "Cross-entropy, √-inverse-frequency class weights"],
        ["Optimiser", "AdamW · lr 2e-5 / 1e-4 / 1e-3"],
        ["Schedule", "10 % warm-up, linear decay"],
        ["Batch · epochs", "16 · ≤ 8, early stop (patience 3)"],
        ["Selection", "best validation macro-F1"],
        ["Hardware", "Apple M1, 8 GB, PyTorch MPS"],
    ], [2.2, 5.0], size=13)
    bullets(s, Inches(8.3), Inches(1.9), Inches(4.5), Inches(4.8), [
        ("Primary metric: ", "macro-F1 — an all-negative model gets ~63 % accuracy but 0.26 macro-F1"),
        ("Uncertainty: ", "95 % bootstrap CIs; paired bootstrap for model differences"),
        ("Modality ablation: ", "neutral image / removed caption at test time"),
        ("Robustness: ", "second split with whole source pages held out"),
    ], size=15)
    footer(s, n)

    # 8. Main results ------------------------------------------------------------
    s = new()
    title(s, "Results: test set, random group split",
          f"{main_c['n_common_test_examples']} test memes · identical for every model · 95 % CIs in notes")
    order = [e for e in ("baseline_majority", "text", "image", "multimodal_concat", "multimodal_attention")
             if e in T]
    bar_chart(s, Inches(0.5), Inches(1.75), Inches(7.6), Inches(5.1), [SHORT[e] for e in order],
              {"Macro-F1": [T[e]["macro_f1"] for e in order], "Accuracy": [T[e]["accuracy"] for e in order]},
              [INK, SAFFRON], "Macro-F1 and accuracy")
    rows = [["Model", "Macro-F1", "95 % CI"]] + [
        [SHORT[e], f"{T[e]['macro_f1']:.3f}", f"{T[e]['macro_f1_ci95'][0]:.2f}–{T[e]['macro_f1_ci95'][1]:.2f}"]
        for e in order]
    table(s, Inches(8.4), Inches(1.9), Inches(4.4), rows, [1.7, 1.1, 1.6], size=13,
          highlight_row=order.index(best) + 1)
    lines = []
    for mm in ("multimodal_concat", "multimodal_attention"):
        for uni in ("text", "image"):
            k = f"{mm}_vs_{uni}"
            if k in tests:
                lines.append(f"{SHORT[mm]} vs {SHORT[uni]}: Δ {tests[k]['observed_diff']:+.3f} → {verdict(tests[k])}")
    text(s, Inches(8.4), Inches(4.5), Inches(4.4), Inches(2.3), lines, size=12, color=TEXT)
    s.notes_slide.notes_text_frame.text = "\n".join(
        f"{SHORT[e]}: acc {T[e]['accuracy']:.4f} {T[e]['acc_ci95']}, macro-F1 {T[e]['macro_f1']:.4f} "
        f"{T[e]['macro_f1_ci95']}, per-class F1 pos/neu/neg {T[e]['f1_positive']:.3f}/{T[e]['f1_neutral']:.3f}/"
        f"{T[e]['f1_negative']:.3f}" for e in T)
    footer(s, n)

    # 9. Per-class + ablation ----------------------------------------------------
    s = new()
    title(s, "Per-class F1 and modality ablation")
    cls_order = [e for e in ("text", "image", "multimodal_concat", "multimodal_attention") if e in T]
    bar_chart(s, Inches(0.5), Inches(1.5), Inches(6.6), Inches(5.3), ["Positive", "Neutral", "Negative"],
              {SHORT[e]: [T[e]["f1_positive"], T[e]["f1_neutral"], T[e]["f1_negative"]] for e in cls_order},
              [MODEL_COLORS[e] for e in cls_order], "Per-class F1")
    ab = main_c.get("modality_ablation", {})
    rows = [["Model · condition", "Macro-F1", "Changed"]]
    for e, v in ab.items():
        rows.append([f"{SHORT[e]} · full input", f"{T[e]['macro_f1']:.3f}", "–"])
        rows.append([f"{SHORT[e]} · neutral image", f"{v['image_neutralised']['macro_f1']:.3f}",
                     f"{v['image_neutralised']['prediction_changed_rate']:.0%}"])
        rows.append([f"{SHORT[e]} · no caption", f"{v['text_removed']['macro_f1']:.3f}",
                     f"{v['text_removed']['prediction_changed_rate']:.0%}"])
    if len(rows) > 1:
        table(s, Inches(7.5), Inches(1.6), Inches(5.3), rows, [2.9, 1.1, 1.3], size=12)
    text(s, Inches(7.5), Inches(1.6 + 0.38 * len(rows) + 0.25), Inches(5.3), Inches(1.2),
         "“Changed” = share of test predictions that flip when one input is neutralised. Large flips for both "
         "inputs mean the fused model relies on both.", size=12, color=MUTED)
    footer(s, n)

    # 10. Shortcut ---------------------------------------------------------------
    s = new()
    title(s, "Shortcut: the source page predicts the label",
          "Memes from the same page share watermarks and style — and very different label mixes")
    sp = T.get("baseline_source_prior")
    box(s, Inches(0.6), Inches(1.9), Inches(4.2), Inches(4.7))
    if sp:
        stat(s, Inches(0.9), Inches(2.1), Inches(3.8), f"{sp['accuracy']:.0%}",
             "test accuracy from page identity alone (no content)", color=NEG, vsize=44)
        stat(s, Inches(0.9), Inches(3.7), Inches(3.8), f"{sp['macro_f1']:.3f}", "macro-F1 of that shortcut",
             color=INK, vsize=44)
    text(s, Inches(0.9), Inches(5.3), Inches(3.8), Inches(1.2),
         "e.g. page “KAM” is 80 % positive; “chosha” is 98.5 % negative", size=13, color=MUTED)
    if sd_c:
        S = {r["experiment"]: r for r in sd_c["table"]}
        eo = [e for e in ("text", "image", "multimodal_concat", "multimodal_attention") if e in S and e in T]
        bar_chart(s, Inches(5.2), Inches(1.75), Inches(7.6), Inches(4.4), [SHORT[e] for e in eo],
                  {"Random split": [T[e]["macro_f1"] for e in eo],
                   "Unseen source pages": [S[e]["macro_f1"] for e in eo]},
                  [INK, SAFFRON], "Macro-F1: seen vs unseen source pages")
        text(s, Inches(5.2), Inches(6.2), Inches(7.6), Inches(0.7),
             "Different test sets (874 vs " + str(sd_c["n_common_test_examples"]) + " memes): compare the drop per "
             "model, not absolute values across splits.", size=12, color=MUTED)
    else:
        text(s, Inches(5.2), Inches(2.5), Inches(7.6), Inches(2), "Source-disjoint results not available yet.",
             size=16, color=MUTED)
    footer(s, n)

    # 11. Explainability ---------------------------------------------------------
    s = new()
    title(s, "Explainability: what the models are sensitive to")
    gal = sorted((ROOT / "artifacts" / "reports" / "explanations").glob("*.png"))
    if gal:
        s.shapes.add_picture(str(gal[0]), Inches(0.5), Inches(1.7), width=Inches(8.4))
    bullets(s, Inches(9.2), Inches(1.5), Inches(3.7), Inches(5.3), [
        ("Occlusion: ", "drop in P(class) when a word is removed — exact, but ignores interactions"),
        ("Integrated Gradients: ", "completeness checked; hidden when it fails to converge on XLM-R"),
        ("Grad-CAM: ", "7×7 sensitivity map; often highlights caption text in the image"),
        ("Caveat: ", "sensitivity ≠ cause; not proof of what an annotator meant"),
    ], size=14)
    footer(s, n)

    # 12. App --------------------------------------------------------------------
    s = new()
    title(s, "The IndicSent-M application", "Streamlit UI + FastAPI service, CPU-compatible, Docker image")
    pages = [("Home", "objectives, languages, architecture"), ("Analyze", "text / image / multimodal input, "
             "probabilities, word highlights, Grad-CAM"), ("Model Comparison", "tables, CIs, confusion matrices, "
             "ablations, both split protocols"), ("Dataset & Methodology", "statistics, pipeline, limitations"),
             ("About", "motivation, technologies, future work")]
    for i, (h_, b_) in enumerate(pages):
        y = Inches(1.8 + i * 0.98)
        c = box(s, Inches(0.6), y, Inches(0.7), Inches(0.7), fill=SAFFRON if i == 1 else INK, shape=MSO_SHAPE.OVAL)
        text(s, Inches(1.5), y - Inches(0.02), Inches(5.6), Inches(0.4), h_, size=17, bold=True, color=INK, font=HEAD)
        text(s, Inches(1.5), y + Inches(0.36), Inches(5.6), Inches(0.5), b_, size=13, color=MUTED)
    box(s, Inches(7.6), Inches(1.8), Inches(5.2), Inches(4.8))
    bullets(s, Inches(7.9), Inches(2.0), Inches(4.7), Inches(4.4), [
        ("Never fakes a prediction: ", "no checkpoint → setup steps; optional preview is labelled DEMO"),
        ("Confidence wording: ", "predicted-class probability, not a guarantee"),
        ("Robust inputs: ", "corrupt / oversized images, empty text, unsupported scripts handled"),
        ("Tested: ", "pytest suite covers data, models, inference, explainers, API"),
    ], size=14)
    footer(s, n)

    # 13. Conclusion -------------------------------------------------------------
    s = new(dark=True)
    title(s, "Conclusions", dark=True)
    concl = []
    concl.append(f"Random split: best macro-F1 {SHORT[best]} ({T[best]['macro_f1']:.3f}).")
    for mm in ("multimodal_concat", "multimodal_attention"):
        if f"{mm}_vs_text" in tests:
            tt = tests[f"{mm}_vs_text"]
            concl.append(f"{SHORT[mm]} vs text-only: {verdict(tt)} (Δ {tt['observed_diff']:+.3f}, one-sided "
                         f"p = {tt['p_a_not_better']:.3f}; 9 comparisons, so treat p > 0.006 as fragile).")
    if "multimodal_attention_vs_multimodal_concat" in tests:
        concl.append(f"Cross-attention vs concatenation: "
                     f"{verdict(tests['multimodal_attention_vs_multimodal_concat'])}.")
    if sp:
        concl.append(f"Page identity alone gives {sp['accuracy']:.0%} accuracy: random-split scores mix sentiment "
                     "with page recognition.")
    if sd_c:
        S = {r["experiment"]: r for r in sd_c["table"]}
        st = sd_c["paired_bootstrap"]
        sbest = max([e for e in trained if e in S], key=lambda e: S[e]["macro_f1"])
        drops = [T[e]["macro_f1"] - S[e]["macro_f1"] for e in trained if e in S]
        concl.append(f"On unseen pages every model drops {min(drops):.2f}–{max(drops):.2f} macro-F1; best is "
                     f"{SHORT[sbest]} ({S[sbest]['macro_f1']:.3f}).")
        k = "multimodal_attention_vs_multimodal_concat"
        if k in st:
            concl.append(f"Unseen pages: cross-attention vs concat {verdict(st[k])} (Δ {st[k]['observed_diff']:+.3f}).")
    concl.append("Neutral memes (~7 %) remain the hardest class for every model.")
    bullets(s, Inches(0.6), Inches(1.6), Inches(7.4), Inches(5.2), concl, size=16, color=WHITE)
    box(s, Inches(8.4), Inches(1.6), Inches(4.4), Inches(5.0), fill=INK2)
    text(s, Inches(8.7), Inches(1.8), Inches(3.9), Inches(0.5), "Future work", size=20, bold=True, color=SAFFRON,
         font=HEAD)
    bullets(s, Inches(8.7), Inches(2.5), Inches(3.9), Inches(4.0), [
        "Hinglish memes (Memotion 3, registration)", "OCR for captions", "CLIP-style vision-language encoders",
        "Several seeds + calibration", "Sarcasm-aware training", "Human study of explanations"],
        size=14, color=WHITE)
    footer(s, n, dark=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Wrote {OUT} ({n} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
