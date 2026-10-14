"""IndicSent-M — Indian Multimodal Sentiment Analyzer (Streamlit app).

    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import streamlit as st
from PIL import Image

from src.data.image_utils import DEFAULT_MAX_BYTES, InvalidImageError, load_image
from src.inference import InputError, ModelNotAvailable, SentimentPredictor, best_multimodal
from src.utils import PROJECT_ROOT, load_config, resolve_path

st.set_page_config(page_title="IndicSent-M", page_icon="🗨️", layout="wide")

CFG = load_config()
REPORTS = resolve_path(CFG["paths"]["reports_dir"])
LABEL_STYLE = {"positive": ("#1e8e4e", "😊"), "neutral": ("#6b7280", "😐"), "negative": ("#c0392b", "😠")}
EXP_NAMES = {"text": "Text-only · XLM-R", "image": "Image-only · ResNet-50",
             "multimodal_concat": "Multimodal · concatenation fusion",
             "multimodal_attention": "Multimodal · cross-attention fusion"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {padding: 1.2rem 1.5rem; border-radius: 14px; color: white;
       background: linear-gradient(120deg, #3b2f8f 0%, #1f6fb2 55%, #1aa38a 100%); margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0; font-size: 2rem;} .hero p {margin: .3rem 0 0; opacity: .92;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 1rem 1.1rem; height: 100%;}
.pill {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.8rem;
       background: rgba(31,111,178,.12); color:#1f6fb2; margin:.1rem;}
.result {border-radius: 14px; padding: 1rem 1.3rem; color: white; margin-bottom: .6rem;}
.result h2 {color:white; margin:0;}
.demo {border: 3px dashed #d35400; border-radius: 12px; padding: .8rem 1rem; background: rgba(211,84,0,.08);}
.small {font-size: .85rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model runtime…")
def get_predictor() -> SentimentPredictor:
    return SentimentPredictor()


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def dataset_is_synthetic() -> bool:
    s = read_json(PROJECT_ROOT / "data" / "processed" / "stats.json")
    return bool(s and s.get("synthetic"))


# ------------------------------------------------------------------------------------ pages
def page_home() -> None:
    hero("IndicSent-M — Indian Multimodal Sentiment Analyzer",
         "Sentiment of Indian-language memes from their text <b>and</b> image: Positive · Neutral · Negative")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h4>🎯 Objective</h4>Test whether combining the caption and the '
                    'picture of a meme classifies its sentiment better than either alone, and show '
                    '<i>why</i> a model decided what it did.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h4>🧩 Modalities</h4>'
                    '<span class="pill">Text</span><span class="pill">Image</span>'
                    '<span class="pill">Text + Image</span><br><br>Each mode is a separately trained '
                    'model evaluated on the same test memes.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><h4>🌐 Languages</h4>Trained on <b>Bengali</b> memes (MemoSen), '
                    'incl. Bengali–English code-mixing and romanised Bengali. XLM-R also covers Hindi, '
                    'Tamil, Telugu, Marathi, Urdu… but those were <b>not</b> in the training data.</div>',
                    unsafe_allow_html=True)
    st.markdown("### Architecture")
    st.code(
        "caption ─► normalise ─► XLM-R base (top 4 layers fine-tuned) ─► mean-pool 768-d ─► proj 256 ┐\n"
        "                                                                                    ├─► fusion ─► MLP ─► softmax(3)\n"
        "image ───► 224×224 ───► ResNet-50 (layer4 fine-tuned) ──────► 2048-d / 7×7 grid ─► proj 256 ┘\n"
        "fusion = concatenation (baseline)  |  bidirectional cross-attention (text tokens ⇄ 49 image regions)",
        language=None)
    st.markdown("### How to use")
    st.markdown("1. Open **Analyze** in the sidebar.\n2. Pick a mode — text-only, image-only or multimodal.\n"
                "3. Enter the meme text and/or upload the image (JPG/PNG, ≤ 10 MB), then press **Analyze**.\n"
                "4. Read the prediction, class probabilities, highlighted words and the Grad-CAM heatmap.\n"
                "5. **Model Comparison** shows how the four models performed on the held-out test set.")
    avail = get_predictor().available()
    if not avail:
        st.warning("No trained model checkpoints were found yet — see the **Analyze** page for setup steps.")
    elif dataset_is_synthetic():
        st.error("The current processed data is the SYNTHETIC sample — results are not meaningful.")


def _setup_message() -> None:
    st.error("No trained checkpoint is available for this mode, so no real prediction can be made.")
    st.markdown("Train the models first (≈1 h on an Apple M1 / faster on a GPU):")
    st.code("python -m src.data.download --dataset memosen\npython -m src.data.prepare\n"
            "python -m src.training.train_text\npython -m src.training.train_image\n"
            "python -m src.training.train_multimodal --fusion concat\n"
            "python -m src.training.train_multimodal --fusion attention\n\n"
            "# or all at once:\npython scripts/run_pipeline.py", language="bash")


def _show_demo(mode: str) -> None:
    st.markdown('<div class="demo"><h3>⚠️ DEMONSTRATION — NO MODEL WAS RUN</h3>'
                'This placeholder exists only to preview the interface layout. The values below are fixed '
                '(uniform) and carry <b>no information</b> about your input.</div>', unsafe_allow_html=True)
    st.bar_chart(pd.DataFrame({"probability (placeholder)": [1 / 3] * 3},
                              index=["positive", "neutral", "negative"]))


def _render_result(r, show_ig: bool) -> None:
    color, emoji = LABEL_STYLE.get(r.label, ("#333", ""))
    st.markdown(f'<div class="result" style="background:{color}"><h2>{emoji} {r.label.capitalize()}</h2>'
                f'Confidence {r.confidence:.1%} · {EXP_NAMES[r.experiment]}</div>', unsafe_allow_html=True)
    st.caption("Confidence is the model's predicted probability for this class — not a guarantee that "
               "the prediction is correct. Models can be confidently wrong, e.g. on sarcasm.")
    probs = pd.DataFrame({"probability": r.probabilities}).reindex(["positive", "neutral", "negative"])
    st.bar_chart(probs, horizontal=True, height=180)
    for w in r.warnings:
        st.warning(w)
    if r.script_info:
        si = r.script_info
        st.caption(f"Detected script(s): {', '.join(si['scripts']) or 'none'} · "
                   f"{'code-mixed' if si['code_mixed'] else 'single script'} · hint: {si['language_hint']}")

    if r.text_occlusion or r.gradcam is not None:
        st.markdown("#### Why this prediction?")
    cols = st.columns(2) if (r.text_occlusion and r.gradcam is not None) else [st.container()]
    i = 0
    if r.text_occlusion:
        from src.explainability.text_explainer import render_html
        with cols[i]:
            st.markdown("**Influential words** — occlusion (leave-one-word-out)")
            st.markdown(render_html(r.text_occlusion["words"], r.label), unsafe_allow_html=True)
            st.caption("Score = drop in P(predicted class) when the word is removed.")
            if show_ig and r.text_attribution:
                ig = r.text_attribution
                st.markdown("**Integrated Gradients**")
                if ig["reliable"]:
                    st.markdown(render_html(ig["words"], r.label), unsafe_allow_html=True)
                    st.caption(f"Completeness residual {ig['convergence_delta']:+.4f} ({ig['steps']} steps) — OK.")
                else:
                    st.info(f"IG did not converge for this input (completeness residual "
                            f"{ig['convergence_delta']:+.4f} vs. ΔP {ig['prob_input'] - ig['prob_baseline']:+.4f}); "
                            "its attributions are hidden because they would be misleading. Use occlusion above.")
        i += 1
    if r.gradcam is not None:
        with cols[min(i, len(cols) - 1)]:
            st.markdown("**Image regions** — Grad-CAM (ResNet-50 layer4)")
            st.image(r.gradcam, width="stretch")
            st.caption("Red = regions the predicted-class score is most sensitive to (7×7 grid, upsampled). "
                       "A sensitivity map, not a causal explanation; it often lights up caption text.")
    with st.expander("Limitations of these explanations"):
        st.markdown(
            "- Both explainers describe the **model's sensitivity**, not the true reason a person would give.\n"
            "- Occlusion removes one word at a time, so it misses interactions (e.g. negation + adjective).\n"
            "- Integrated Gradients depends on the chosen baseline and can fail to converge on transformer "
            "models; we check its completeness property and hide it when it fails.\n"
            "- Grad-CAM is coarse (7×7) and can highlight text rendered in the image rather than visual content.\n"
            "- In multimodal mode the two explanations are computed separately with the other input held fixed.")


def page_analyze() -> None:
    hero("Sentiment Analysis", "Analyze a meme from its text, its image, or both")
    pred = get_predictor()
    avail = pred.available()
    if dataset_is_synthetic() and avail:
        st.error("Checkpoints were trained on SYNTHETIC sample data — predictions are not meaningful.")

    mode = st.radio("Analysis mode", ["Multimodal (text + image)", "Text-only", "Image-only"], horizontal=True)
    mode_key = {"Multimodal (text + image)": "multimodal", "Text-only": "text", "Image-only": "image"}[mode]
    if mode_key == "multimodal":
        mm_opts = [e for e in ("multimodal_concat", "multimodal_attention") if e in avail] or ["multimodal_concat"]
        default = best_multimodal(avail) or mm_opts[0]
        exp = st.selectbox("Fusion model", mm_opts, index=mm_opts.index(default), format_func=EXP_NAMES.get,
                           help="Default = the multimodal model with the best validation macro-F1.")
    else:
        exp = mode_key

    if "example" not in st.session_state:
        st.session_state.example = None
    test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    if test_csv.exists():
        if st.button("🎲 Load a random meme from the held-out test set",
                     help="Real MemoSen memes; some contain crude or offensive humour."):
            row = pd.read_csv(test_csv, keep_default_na=False).sample(1).iloc[0]
            st.session_state.example = row.to_dict()
            st.session_state.text_input = row["text"]
    ex = st.session_state.example

    c1, c2 = st.columns([1.1, 1])
    text, image_src = None, None
    with c1:
        if mode_key in ("text", "multimodal"):
            text = st.text_area("Meme text / caption", key="text_input", height=130,
                                placeholder="e.g. আজকের দিনটা দারুণ কাটলো! 😄   or   aaj ka din mast tha 😂")
        if ex:
            st.caption(f"Test example `{ex['id']}` · annotated label: **{ex['label']}**")
    with c2:
        if mode_key in ("image", "multimodal"):
            up = st.file_uploader("Meme image (JPG, JPEG, PNG · max 10 MB)", type=["jpg", "jpeg", "png"])
            if up is not None:
                data = up.getvalue()
                try:
                    preview = load_image(data, max_bytes=DEFAULT_MAX_BYTES)
                    image_src = data
                    st.image(preview, caption="Uploaded image", width="stretch")
                except InvalidImageError as e:
                    st.error(f"Image rejected: {e}")
            elif ex:
                path = resolve_path(ex.get("image_source") or ex["image_path"])
                image_src = str(path)
                st.image(Image.open(path), caption="Test-set image", width="stretch")

    explain = st.toggle("Compute explanations (slower)", value=True)
    show_ig = st.toggle("Also show Integrated Gradients (experimental)", value=False, disabled=not explain)
    demo = False
    if exp not in avail:
        _setup_message()
        demo = st.checkbox("Preview the interface in clearly-labelled DEMO mode (no model)")
    if st.button("Analyze", type="primary", width="stretch"):
        if demo:
            _show_demo(mode_key)
            return
        if exp not in avail:
            st.stop()
        try:
            with st.spinner("Running the model…"):
                r = pred.predict(text, image_src, exp, explain=explain)
        except InputError as e:
            st.warning(str(e))
            return
        except ModelNotAvailable as e:
            st.error(str(e))
            return
        except Exception as e:  # never crash the UI
            st.error(f"Unexpected error during inference: {type(e).__name__}: {e}")
            return
        _render_result(r, show_ig)
        st.caption(f"Inference {r.latency_ms:.0f} ms on {r.device} (excluding explanations) · checkpoint epoch "
                   f"{r.checkpoint['epoch']}, val macro-F1 {r.checkpoint['val_macro_f1']:.3f}")


PROTOCOLS = {
    "Random group split (main)": (REPORTS, PROJECT_ROOT / CFG["paths"]["checkpoints_dir"]),
    "Source-disjoint split (robustness)": (PROJECT_ROOT / "artifacts" / "reports_source_disjoint",
                                           PROJECT_ROOT / "artifacts" / "checkpoints_source_disjoint"),
}


def page_comparison() -> None:
    hero("Model Comparison", "Held-out test-set results generated from the saved evaluation outputs")
    avail_protocols = [k for k, (r, _) in PROTOCOLS.items() if (r / "comparison" / "comparison.json").exists()]
    if len(avail_protocols) > 1:
        proto = st.radio("Evaluation protocol", avail_protocols, horizontal=True,
                         help="Main: memes split at random (grouped by near-duplicates). Robustness: whole "
                              "source pages held out, so models cannot exploit page-specific style.")
    else:
        proto = avail_protocols[0] if avail_protocols else list(PROTOCOLS)[0]
    reports, ckpts = PROTOCOLS[proto]
    comp = read_json(reports / "comparison" / "comparison.json")
    if not comp:
        st.info("No comparison yet. Run `python -m src.evaluation.evaluate` then "
                "`python -m src.evaluation.compare_models`.")
        return
    if dataset_is_synthetic():
        st.error("These numbers come from the SYNTHETIC sample data and are meaningless.")
    t = pd.DataFrame(comp["table"])
    show = t[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
              "f1_positive", "f1_neutral", "f1_negative", "trainable_params_M", "training_min",
              "latency_median_ms"]].set_index("model")
    st.markdown(f"**Test set:** {comp['n_common_test_examples']} memes (identical for every model).")
    st.dataframe(show.style.format("{:.3f}").highlight_max(subset=["accuracy", "macro_f1", "weighted_f1"],
                                                           color="#cdeccd"), width="stretch")
    st.caption("Macro-F1 (unweighted mean of per-class F1) is the primary metric because classes are imbalanced. "
               "Latency = median per-meme time at batch size 1 incl. preprocessing. "
               "\\*The source-page prior predicts the most common label of the page a meme came from — it uses "
               "metadata, not content, and shows how much of the task a model could solve by recognising the page.")
    c1, c2 = st.columns(2)
    for col, img in [(c1, "accuracy_macro_f1.png"), (c2, "per_class_f1.png")]:
        p = reports / "comparison" / img
        if p.exists():
            col.image(str(p), width="stretch")
    p = reports / "comparison" / "confusion_matrices.png"
    if p.exists():
        st.markdown("#### Confusion matrices")
        st.image(str(p), width="stretch")

    st.markdown("#### Findings (computed from the results above)")
    for line in findings(t, comp):
        st.markdown(f"- {line}")
    if comp.get("paired_bootstrap"):
        st.markdown("#### Paired bootstrap tests (macro-F1 difference)")
        st.dataframe(pd.DataFrame(comp["paired_bootstrap"]).T, width="stretch")
    if comp.get("modality_ablation"):
        st.markdown("#### Does the multimodal model use both inputs? (modality ablation)")
        rows = [{"model": EXP_NAMES[e], "condition": c, **v} for e, ab in comp["modality_ablation"].items()
                for c, v in ab.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch")
    hist = {e: ckpts / e / "history.csv" for e in EXP_NAMES}
    hist = {e: pd.read_csv(p) for e, p in hist.items() if p.exists()}
    if hist:
        st.markdown("#### Validation macro-F1 per epoch")
        st.line_chart(pd.DataFrame({EXP_NAMES[e]: h.set_index("epoch")["val_macro_f1"] for e, h in hist.items()}))
    gal = reports / "explanations"
    figs = sorted(gal.glob("*.png")) if gal.exists() else []
    if figs:
        with st.expander(f"Explanation gallery ({len(figs)} test memes)"):
            for f in figs:
                st.image(str(f), width="stretch")


def findings(t: pd.DataFrame, comp: dict) -> list[str]:
    """Plain-language statements derived only from the numbers in ``t`` and ``comp``."""
    names = {**EXP_NAMES, "baseline_majority": "majority-class baseline",
             "baseline_source_prior": "source-page prior (shortcut reference)"}
    out = []
    t = t[~t["experiment"].str.startswith("baseline")] if (~t["experiment"].str.startswith("baseline")).any() else t
    best = t.loc[t["macro_f1"].idxmax()]
    out.append(f"Highest test macro-F1 among trained models: **{best['model']}** ({best['macro_f1']:.3f}, 95% CI "
               f"{best['macro_f1_ci95'][0]:.3f}–{best['macro_f1_ci95'][1]:.3f}).")
    for k, v in comp.get("paired_bootstrap", {}).items():
        a, b = k.split("_vs_")
        lo, hi = v["ci95"]
        verdict = ("significantly better" if lo > 0 else "significantly worse" if hi < 0
                   else "not significantly different")
        out.append(f"{names.get(a, a)} vs {names.get(b, b)}: Δ macro-F1 = {v['observed_diff']:+.3f} "
                   f"(95% CI {lo:+.3f} to {hi:+.3f}) → {verdict} at the 95% level.")
    if "f1_neutral" in t:
        worst = t["f1_neutral"].max()
        out.append(f"Neutral is the hardest class (best neutral F1 = {worst:.3f}); it is only ~7% of MemoSen.")
    return out


def page_dataset() -> None:
    hero("Dataset & Methodology", "What the models were trained on, and how")
    s = read_json(PROJECT_ROOT / "data" / "processed" / "stats.json")
    if not s:
        st.info("Run `python -m src.data.prepare` to generate dataset statistics.")
    else:
        if s.get("synthetic"):
            st.error("Processed data is the SYNTHETIC sample.")
        st.markdown(f"**Dataset:** `{s['dataset']}` · {s['usable_records']} usable memes of {s['raw_records']} "
                    f"annotations · split: {s['split_source']}")
        c = st.columns(4)
        c[0].metric("Train", s["split_sizes"]["train"])
        c[1].metric("Validation", s["split_sizes"]["val"])
        c[2].metric("Test", s["split_sizes"]["test"])
        c[3].metric("Code-mixed captions", f"{s['code_mixed_fraction']:.0%}")
        st.dataframe(pd.DataFrame(s["label_distribution"]), width="stretch")
        figs = REPORTS / "dataset"
        cols = st.columns(2)
        for i, f in enumerate(["label_distribution.png", "script_distribution.png", "text_length.png",
                               "image_sizes.png"]):
            if (figs / f).exists():
                cols[i % 2].image(str(figs / f), width="stretch")
        with st.expander("Removed records & raw statistics"):
            st.json(s)
    st.markdown("""
#### MemoSen
Bengali meme sentiment dataset (Hossain, Sharif & Hoque, *LREC 2022*): meme images with their
embedded caption text transcribed and a positive / neutral / negative label. Captions are mostly
Bengali script, with English and romanised Bengali mixed in. No explicit licence is published with
the data; it is used here for non-commercial research with citation, and is not redistributed.

#### Preprocessing
1. Load captions, image file names and labels; map labels to `positive / neutral / negative`.
2. Validate every image (decodable, size limits); drop records whose image is missing or corrupt.
3. Normalise text conservatively: Unicode NFC, URLs → `URL`, mentions → `@USER`, character floods
   capped at 3 — **emojis, punctuation, negations, casing and ZWJ/ZWNJ are kept**.
4. Script-based language identification (Bengali, Devanagari, Latin, …) and code-mixing flag.
5. Remove exact duplicates (same image hash + text); drop duplicates with conflicting labels.
6. Group near-duplicates (same image hash or same long caption) and split **by group**, stratified by
   label, 70/10/20 with seed 42, so no meme template or caption leaks across splits.

#### Training (identical for all experiments)
AdamW (lr 2e-5 text encoder, 1e-4 ResNet layer4, 1e-3 new layers, weight decay 0.01), linear
warm-up 10 % + decay, batch 16, ≤ 8 epochs, early stopping (patience 3) and checkpoint selection
on **validation macro-F1**, gradient clipping 1.0, dropout 0.3, cross-entropy with square-root
inverse-frequency class weights. Frozen: XLM-R embeddings + layers 1–8, ResNet stem + layers 1–3.

#### Evaluation
Accuracy, macro & weighted precision / recall / F1, per-class report, confusion matrix, 95 %
bootstrap confidence intervals, paired bootstrap tests between models, per-meme latency, and a
modality ablation (neutral image / removed text) for the multimodal models.

#### Limitations & responsible use
- Trained on ~3 k Bengali memes from a few sources: it will not generalise reliably to other
  languages, platforms or time periods. Hindi/Tamil/etc. inputs run but are out-of-distribution.
- Labels are subjective; sarcasm and cultural context are frequently misread.
- Some memes in the dataset contain crude, sexual or offensive humour.
- Do not use for moderation, profiling or any decision about individuals without human review.
""")


def page_about() -> None:
    hero("About IndicSent-M", "Motivation, objectives and technology")
    st.markdown("""
#### Motivation
Indian social media is multilingual, code-mixed and visual. A meme's sentiment often lives in the
interaction between its caption and its picture — a cheerful caption over a grim image can be sarcasm.
Text-only, English-centric sentiment tools miss this.

#### Research objectives
1. Build text-only, image-only and multimodal sentiment classifiers for Indian-language memes.
2. Compare them fairly on identical data splits with uncertainty estimates.
3. Compare concatenation with cross-attention fusion.
4. Provide word- and region-level explanations, with their limitations stated.

#### Technologies
Python · PyTorch · Hugging Face Transformers (XLM-RoBERTa base) · torchvision (ResNet-50) ·
scikit-learn · pandas · Matplotlib/Seaborn · Streamlit · FastAPI · Docker · pytest

#### Implemented vs. future work
**Implemented:** data pipeline with leakage-safe splits, 4 trained models, evaluation + bootstrap
statistics, modality ablation, occlusion / Integrated Gradients / Grad-CAM, this app, REST API, tests, Docker.

**Future work:** Hinglish data (Memotion 3, needs registration), OCR for memes without transcribed text,
CLIP-style vision–language encoders, multiple training seeds, calibration (temperature scaling),
sarcasm-aware training, human evaluation of explanations.
""")


PAGES = {"🏠 Home": page_home, "🔍 Analyze": page_analyze, "📊 Model Comparison": page_comparison,
         "📚 Dataset & Methodology": page_dataset, "ℹ️ About": page_about}

with st.sidebar:
    st.markdown("## IndicSent-M")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    try:
        avail = get_predictor().available()
        st.markdown("**Trained models**")
        for e, name in EXP_NAMES.items():
            if e in avail:
                st.markdown(f"✅ {name}<br><span class='small'>val macro-F1 {avail[e]['val_macro_f1']:.3f}</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"⬜ {name} <span class='small'>(not trained)</span>", unsafe_allow_html=True)
        st.caption(f"Device: {get_predictor().device}")
    except Exception as e:
        st.error(f"Model runtime failed to initialise: {e}")

PAGES[choice]()
