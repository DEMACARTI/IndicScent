# IndicSent-M: Research report

**Multimodal Sentiment Analysis for Indian Languages Using Text and Images**

Every number in this report was copied from files produced by the evaluation scripts:
`artifacts/reports/<experiment>/metrics.json`, `artifacts/reports/comparison/comparison.{json,md}`,
the `artifacts/reports_source_disjoint/` equivalents, and the checkpoints' `training_summary.json`.
Rerunning `python -m src.evaluation.compare_models` regenerates the tables.

## Abstract

We study whether combining a meme's caption with its image improves three-way sentiment
classification (positive / neutral / negative) for an Indian language. On MemoSen, 4,362 Bengali
memes of which 21 % mix Bengali and Latin script, we compare four models on an identical,
leakage-safe split: a partially fine-tuned XLM-R base text model, a ResNet-50 image model, and
two fusion models (concatenation and bidirectional cross-attention). On the 874-meme test set,
**cross-attention fusion reaches macro-F1 0.589 (95 % CI 0.547–0.629) vs. 0.545 for text-only
and 0.503 for image-only.** The gain over text-only is +0.044 (paired bootstrap 95 % CI
+0.004 to +0.086): small, and only marginally significant. Concatenation fusion (0.578) is not
significantly better than text-only. Both fusion models are significantly better than
image-only. A modality ablation shows that the fusion models depend on both inputs.

We also find a strong **dataset shortcut**: the page a meme was collected from predicts its
label, so a content-free "source-page prior" reaches 78.3 % accuracy, more than any model.
When whole source pages are held out (§6), **every model loses 19–25 macro-F1 points**
(best: concatenation fusion, 0.387). The models still beat a majority-class baseline on
macro-F1, but **not on accuracy**. On unseen pages, concatenation fusion is significantly better
than text-only (+0.061) and than cross-attention fusion (+0.049), which reverses the
random-split ordering of the two fusion methods. Neutral memes (6.6 % of the data) remain hard
for every model.

**Bottom line.** Fusing caption and image helps on this data. However, the random-split numbers,
ours included, substantially overstate how well these models understand meme sentiment.

## 1. Setup (summary)

Full details: `docs/methodology.md` and `docs/dataset_documentation.md`.

- **Data:** MemoSen (Hossain et al., LREC 2022), public release. After validation and
  de-duplication, 4,362 memes remain, split by groups (near-duplicates kept together) and
  stratified into 3,054 train / 434 validation / 874 test (seed 42). Labels: 62.5 % negative,
  30.8 % positive, 6.6 % neutral.
- **Models:** XLM-R base (embeddings and layers 1–8 frozen, 28.4 M trainable parameters);
  ResNet-50 (`layer4` and head trainable, 15.0 M); fusion models (44.2 M / 44.8 M trainable)
  combining both encoders through concatenation (256 + 256) or text⇄image cross-attention.
- **Training:** identical recipe for all models. AdamW, square-root class-weighted
  cross-entropy, at most 8 epochs, early stopping, checkpoint chosen on **validation**
  macro-F1. Single seed (42). Apple M1 8 GB, PyTorch MPS.
- **Metrics:** macro-F1 (primary), accuracy, macro/weighted P/R/F1, per-class scores and
  confusion matrices. 95 % bootstrap CIs (1,000 resamples) and paired bootstrap tests.

## 2. Main results (random group split, 874 test memes)

| Model | Accuracy (95 % CI) | Macro-P | Macro-R | **Macro-F1 (95 % CI)** | Weighted-F1 |
|---|---|---|---|---|---|
| Text-only (XLM-R) | 0.690 [0.659, 0.722] | 0.542 | 0.554 | 0.545 [0.504, 0.589] | 0.694 |
| Image-only (ResNet-50) | 0.681 [0.650, 0.714] | 0.522 | 0.497 | 0.503 [0.465, 0.543] | 0.673 |
| Multimodal: concat | **0.743** [0.713, 0.770] | **0.595** | 0.567 | 0.578 [0.534, 0.621] | 0.735 |
| Multimodal: cross-attention | **0.743** [0.713, 0.771] | 0.592 | **0.589** | **0.589** [0.547, 0.629] | **0.743** |
| *Reference: majority class* | 0.626 | 0.209 | 0.333 | 0.257 | 0.482 |
| *Reference: source-page prior†* | *0.783* [0.756, 0.810] | 0.526 | 0.530 | 0.524 [0.503, 0.543] | 0.752 |

† The source-page prior predicts each test meme's label as the most frequent training label of
the page it came from. It uses metadata that a deployed model never sees, so it is a diagnostic,
not a competitor (§5).

**Per-class F1**

| Model | Positive | Neutral | Negative |
|---|---|---|---|
| Text-only | 0.637 | 0.226 | 0.772 |
| Image-only | 0.562 | 0.167 | 0.781 |
| Multimodal: concat | 0.672 | 0.242 | 0.819 |
| Multimodal: cross-attention | **0.689** | **0.256** | **0.822** |

Figures: `artifacts/reports/comparison/accuracy_macro_f1.png`, `per_class_f1.png`,
`confusion_matrices.png`.

### 2.1 Are the differences real? (paired bootstrap on macro-F1, 1,000 resamples)

| A vs B | Δ macro-F1 | 95 % CI | P(A not better) |
|---|---|---|---|
| Cross-attention vs text-only | +0.044 | [+0.004, +0.086] | 0.015 |
| Concat vs text-only | +0.033 | [−0.018, +0.084] | 0.097 |
| Cross-attention vs image-only | +0.086 | [+0.043, +0.132] | < 0.001 |
| Concat vs image-only | +0.074 | [+0.034, +0.119] | 0.002 |
| Cross-attention vs concat | +0.011 | [−0.035, +0.058] | 0.318 |
| Text-only vs source prior | +0.021 | [−0.021, +0.067] | 0.159 |
| Image-only vs source prior | −0.020 | [−0.061, +0.021] | 0.852 |
| Concat vs source prior | +0.054 | [+0.014, +0.095] | 0.005 |
| Cross-attention vs source prior | +0.065 | [+0.027, +0.104] | < 0.001 |

**Multiple comparisons.** Nine comparisons are listed. With a Bonferroni-corrected threshold
(0.05 / 9 ≈ 0.0056), the cross-attention vs text-only gain (one-sided p ≈ 0.015) would **not**
be significant. The gains of both fusion models over image-only, and over the source prior,
remain significant.

### 2.2 Do the fusion models use both modalities? (test-time ablation)

| Model | Condition | Accuracy | Macro-F1 | Predictions changed |
|---|---|---|---|---|
| Concat | full input | 0.743 | 0.578 | – |
| Concat | image → ImageNet-mean image | 0.674 | 0.511 | 23.1 % |
| Concat | caption removed | 0.728 | 0.470 | 14.0 % |
| Cross-attention | full input | 0.743 | 0.589 | – |
| Cross-attention | image → ImageNet-mean image | 0.731 | 0.467 | 20.9 % |
| Cross-attention | caption removed | 0.675 | 0.376 | 27.2 % |

Removing either input costs both fusion models 7–21 macro-F1 points and flips 14–27 % of their
predictions. Neither model ignores a modality. The cross-attention model relies more on the
caption than the concat model does.

### 2.3 Cost

| Model | Trainable params | Epochs run (best) | Training time | Latency median / p95 (bs 1, MPS) | Eval throughput |
|---|---|---|---|---|---|
| Text-only | 28.4 M | 8 (7) | 9.6 min | 18.8 / 195.0 ms | 87 memes/s |
| Image-only | 15.0 M | 8 (7) | 10.8 min | 19.8 / 20.7 ms | 67 memes/s |
| Concat | 44.2 M | 5 (2) | 13.7 min | 38.8 / 44.3 ms | 43 memes/s |
| Cross-attention | 44.8 M | 5 (2) | 13.7 min | 47.9 / 188.2 ms | 42 memes/s |

Latency covers loading and preprocessing the image, tokenisation and the forward pass. The
large p95 for text-carrying models is probably MPS kernel compilation for previously unseen
sequence lengths; this was not investigated further. All four models were trained one after
another on the same machine.

## 3. Discussion

1. **Fusion helps, modestly.** Both fusion models beat both unimodal models on every aggregate
   metric. Against the stronger unimodal model (text), the improvement is 3–4 macro-F1 points
   and statistically fragile. Against image-only it is large and robust. This is consistent with
   the MemoSen authors' report of a small multimodal gain (they state ≈ 1.2 %). Our numbers are
   **not directly comparable** to theirs: we use a different split, metric and models.
2. **Cross-attention vs concatenation:** cross-attention was ahead on macro-F1, macro-recall
   and every per-class F1, but the difference (+0.011) is well inside the noise (CI
   −0.035 to +0.058). We cannot claim attention fusion is better.
3. **The image-only model is surprisingly competitive** (0.503 macro-F1, 68 % accuracy), even
   though an ImageNet ResNet cannot read Bengali. Part of this is the source shortcut (§5–§6).
4. **Overfitting is fast.** Both fusion models peaked at epoch 2. For concat, training loss kept
   falling (0.24 at epoch 5) while validation loss rose (0.79 → 1.57). With ~3 k examples, the ResNet `layer4`
   (15 M parameters) memorises quickly. Rising validation loss alongside stable macro-F1 also
   points to growing over-confidence, so the reported confidences are not calibrated.
5. **Neutral is the bottleneck.** With 203 training examples, neutral F1 is 0.17–0.26 for every
   model, and most neutral memes are predicted negative (confusion matrices). Macro-F1 is held
   down mainly by this class.

## 4. Explainability observations

The gallery is in `artifacts/reports/explanations/` (one figure per selected test meme, plus
`explanations.json`).

The gallery covers 6 test memes, chosen deterministically (seed 42) as one correctly and one
wrongly classified meme per true class according to the cross-attention model. The figures show
explanations for the text-only, image-only and cross-attention models.

- **Integrated Gradients completeness.** The check passed for **5 of 6** cross-attention
  explanations, but for **0 of 6** text-only explanations (residuals up to −0.39 in probability
  units, 32 steps). The text-only model's attributions were therefore hidden in the app, and
  occlusion was used instead. This matches the non-smoothness diagnosed in `methodology.md` §6.
- **Watermark words.** In `Halal_Memes (42)` the transcribed caption contains the page
  watermark "Halal memes", and occlusion shows "memes" is the text model's strongest word for
  its prediction (negative; the gold label is positive). Watermark text is transcribed in only
  a few captions (12 of 347 Halal Memes captions; ≤ 5 for any other page), so this is a vivid
  example of the shortcut, not its main route.
- **Grad-CAM** (from inspecting two figures, `KAM (16)` and `Halal_Memes (42)`): the heatmaps
  concentrate on faces near the panel boundary of these two-panel memes. In `KAM (16)` the
  image-only and multimodal models highlight different regions. Two examples do not support
  general claims about what the models attend to.
- **Rendering caveat.** Matplotlib does not perform complex text shaping, so Bengali vowel signs
  in the static gallery figures appear detached from their consonants. The Streamlit app renders
  words in the browser, which shapes Bengali correctly.
- These explanations describe model sensitivity, not causes. They were not validated with
  human judges.

## 5. Dataset shortcut: source page ⇒ label

MemoSen file names encode their source page (12 pages). Label distributions differ sharply
between pages, e.g. `KAM` 79.9 % positive (n = 1,050), `chosha` 98.5 % negative (n = 131),
`ovodro_img` 83.9 % negative (n = 802). Pages also have recognisable visual styles and
watermarks. As a result:

- the source-page prior gets **78.3 % accuracy**, the best of all systems, and 0.524 macro-F1,
  with no content at all;
- text-only and image-only models are **not** significantly better than this prior on
  macro-F1, while the fusion models are (+0.054 and +0.065).

So random-split numbers on MemoSen, including ours, partly measure page recognition. Any
reported score on this dataset should be read with that in mind.

## 6. Robustness: source-disjoint evaluation

**Protocol.** `configs/source_disjoint.yaml` keeps every MemoSen source page entirely inside one
split. The rule for choosing held-out pages was fixed before any model was evaluated on this
split: pages are ranked by the Jensen–Shannon divergence between their label distribution and
the global one, and the most typical pages go to test, then validation. This keeps the label mix
of the held-out data close to the training data.

- Test pages: bangla-gag, bangla-memes, tangaila, nurani-memes (938 memes).
- Validation pages: bangla-troll, Halal Memes, Bangla Thug Life (817 memes).
- Training pages: the other five (2,602 memes).
- Five memes re-posted by two pages were removed from the later split to avoid leakage.

Model recipe, hyper-parameters and seed are identical to §2.

| Model | Accuracy (95 % CI) | Macro-F1 (95 % CI) | F1 pos / neu / neg | Recall pos / neu / neg | Δ macro-F1 vs random split |
|---|---|---|---|---|---|
| Text-only | 0.550 [0.518, 0.578] | 0.327 [0.298, 0.355] | 0.180 / 0.081 / 0.719 | 0.130 / 0.103 / 0.795 | −0.218 |
| Image-only | 0.557 [0.526, 0.588] | 0.364 [0.331, 0.398] | 0.281 / 0.105 / 0.705 | 0.232 / 0.121 / 0.755 | −0.140 |
| Multimodal: concat | 0.523 [0.490, 0.554] | **0.387** [0.356, 0.418] | 0.301 / 0.169 / 0.692 | 0.242 / 0.379 / 0.672 | −0.191 |
| Multimodal: cross-attention | 0.561 [0.529, 0.593] | 0.338 [0.309, 0.364] | 0.249 / 0.044 / 0.721 | 0.193 / 0.052 / 0.787 | −0.251 |
| *Majority class* (= source prior: every test page is unseen) | *0.634* [0.603, 0.664] | 0.259 | 0 / 0 / 0.776 | – | – |

(The random-split and source-disjoint test sets differ, so the last column compares scores on
different memes. It shows the size of the drop, not a paired difference.)

**Paired bootstrap (macro-F1, 938 test memes)**

| A vs B | Δ | 95 % CI |
|---|---|---|
| Concat vs text-only | +0.061 | [+0.022, +0.099] |
| Concat vs image-only | +0.023 | [−0.008, +0.055] |
| Cross-attention vs text-only | +0.012 | [−0.025, +0.045] |
| Cross-attention vs image-only | −0.026 | [−0.059, +0.005] |
| Cross-attention vs concat | −0.049 | [−0.079, −0.021] |
| Each model vs majority / source prior | +0.068 to +0.128 | all CIs > 0 |

**Modality ablation on unseen pages:**
- Concat with the image replaced by the ImageNet-mean image collapses to 0.099 macro-F1, with
  71.9 % of predictions changing. Removing the caption changes 14.8 % of predictions
  (macro-F1 0.380).
- Cross-attention: 24.6 % of predictions change for the neutral image (macro-F1 0.261) and
  14.6 % for the removed caption (macro-F1 0.312).
- The concat collapse shows that a flat grey image is itself out-of-distribution for that model:
  "neutralising" a modality this way is not a neutral intervention.

**Interpretation.**
1. **Most of the random-split performance does not transfer to new pages.** All models lose
   0.14–0.25 macro-F1, and none beats always predicting "negative" on accuracy.
2. **The positive class collapses** (recall 0.13–0.24). In this training split, 839 of the 911
   positive memes (92.1 %) come from a single page (KAM). The models learned "positive ≈ looks
   and reads like KAM", which a page-held-out test exposes directly.
3. **Fusion still helps, but the ranking changes.** Concatenation fusion is the best model on
   unseen pages and significantly beats text-only and cross-attention. Cross-attention, the best
   model on the random split, is no better than the unimodal models here. Its larger capacity
   (token-level and region-level interactions) appears to fit page-specific cues more readily.
   Its best checkpoint came at epoch 7 here versus epoch 2 on the random split, and its
   validation pages (Bangla Thug Life, Halal Memes, bangla-troll) are themselves very skewed,
   which makes early stopping noisy.
4. **Text-only drops the most** relative to its own random-split score (−0.218), even though
   XLM-R reads the caption. A plausible reading is that page-specific vocabulary and topics were
   doing much of the work.
5. Even on unseen pages, every model is significantly better than the content-free baselines on
   macro-F1. So some genuine sentiment signal does transfer, just far less than the random
   split suggests.

## 7. Conclusions

1. **Does multimodal fusion help?** Yes. On both protocols, the best model is a fusion model,
   and both fusion models beat image-only on the random split. The size of the gain over
   text-only is protocol-dependent: +0.044 (cross-attention, fragile) on the random split and
   +0.061 (concat, robust) on unseen pages.
2. **Concatenation vs cross-attention.** They are indistinguishable on the random split, and
   concatenation is significantly better on unseen pages. The simple baseline is the safer choice
   on this amount of data.
3. **What the numbers mean.** Random-split scores on MemoSen mix sentiment understanding with
   page recognition. Reporting a source-disjoint evaluation alongside them is necessary.
4. **Explanations.** Occlusion is the reliable word-level tool here. IG on the fine-tuned XLM-R
   text model failed its completeness check in every gallery example. Explanations helped
   uncover the watermark-word shortcut.

## 8. Limitations

- **Single seed per model.** No variance across seeds is measured. The bootstrap CIs capture
  test-set sampling noise only, not training randomness.
- **Hyper-parameters** were fixed in advance and not tuned. The partial fine-tuning
  (frozen embeddings and lower layers) was a memory constraint and may cost accuracy.
- **One dataset, one language.** The models were trained on Bengali memes only. Hindi, Tamil
  and code-mixed Hinglish inputs run but are out of distribution.
- **No OCR.** Captions come from the dataset's transcriptions.
- **Confidence is not calibrated** (see §3.4).
- **Source shortcut** (§5) inflates random-split scores. The source-disjoint split (§6) has only
  four test pages and one train/val/test assignment, so its estimates carry page-level variance
  that the bootstrap over memes does not capture. A leave-one-page-out study would be stronger.
- **Training time vs wall clock.** Reported training times are compute time measured with a
  monotonic clock. The machine slept for about 40 minutes during the source-disjoint concat run,
  which affected wall-clock time but not the measurements.
- **Explanations** are sensitivity analyses and were not evaluated with users.
- The comparison with the MemoSen paper's own numbers is not like-for-like.

## 9. Future work

Hinglish memes (Memotion 3, after registration); OCR for real-world inputs; CLIP-style
vision–language encoders; several training seeds; temperature-scaling calibration;
source-balanced sampling, page-adversarial training or leave-one-page-out evaluation to
address the shortcut; more diverse positive memes (92 % currently come from one page); focal loss or
data augmentation for the neutral class; a human study of explanation usefulness.

## 10. Reproduction

```bash
python scripts/run_pipeline.py                                           # main experiments
python -m src.data.prepare --config configs/source_disjoint.yaml         # robustness split
./scripts/train_all.sh --config configs/source_disjoint.yaml             # (or the four train_* commands with --config)
python -m src.evaluation.evaluate --config configs/source_disjoint.yaml
python -m src.evaluation.compare_models --config configs/source_disjoint.yaml
```
