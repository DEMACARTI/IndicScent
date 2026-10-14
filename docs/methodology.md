# Methodology

This document describes what the code in `src/` actually does. Hyper-parameters are taken from
`configs/config.yaml`; every checkpoint also stores the exact resolved config it was trained with
(`artifacts/checkpoints/<experiment>/metadata.json`).

## 1. Task

The input is a meme: its caption text *t* and/or its image *v*. The output is a distribution over
𝒴 = {positive, neutral, negative} (class indices 0, 1, 2). We compare four classifiers trained
and evaluated on **identical** splits:

| Experiment | Input | Encoder(s) | Fusion |
|---|---|---|---|
| A `text` | t | XLM-R base | – |
| B `image` | v | ResNet-50 | – |
| C1 `multimodal_concat` | t, v | XLM-R base + ResNet-50 | concatenation (baseline) |
| C2 `multimodal_attention` | t, v | XLM-R base + ResNet-50 | bidirectional cross-attention |

## 2. Encoders

### Text: XLM-RoBERTa base (`FacebookAI/xlm-roberta-base`)
- 12 layers, hidden size 768, 250k SentencePiece vocabulary, 278 M parameters.
- **Why XLM-R rather than IndicBERT.** XLM-R's CC-100 pre-training data includes Bengali and
  *romanised* Bengali/Hindi/Tamil/Telugu/Urdu, and 21 % of MemoSen captions mix Bengali script
  with Latin script. IndicBERT v1 (ALBERT-based) was trained on native-script news text, so it
  matches this data less well. XLM-R is also cased, which preserves "BEST" vs "best".
- Tokenisation: `max_length = 96`, truncation, dynamic padding per batch. The p95 caption
  length is 55 tokens.
- **Pooling:** masked mean over the last hidden layer → h_t ∈ ℝ⁷⁶⁸.
- **Partial fine-tuning:** the embedding matrix (192 M of the 278 M parameters) and layers 1–8
  are frozen; layers 9–12 are trained (28.4 M trainable parameters). Reasons: (i) the optimiser
  state for the full model would not fit comfortably in 8 GB of unified memory, and (ii) with
  ~3 k training examples, full fine-tuning is prone to over-fitting.

### Image: ResNet-50 (torchvision `IMAGENET1K_V2` weights)
- Input: RGB resized to 224 × 224, ImageNet mean/std normalisation.
- Training augmentation: `RandomResizedCrop(scale 0.75–1, ratio 0.8–1.25)`, ±5° rotation,
  colour jitter. **No horizontal flip**, because memes contain rendered text and mirrored text
  is out-of-distribution.
- Output: pooled vector h_v ∈ ℝ²⁰⁴⁸ and the 7 × 7 × 2048 `layer4` feature grid (49 regions)
  used by attention fusion and Grad-CAM.
- **Partial fine-tuning:** stem and `layer1–3` are frozen; `layer4` is trained (15.0 M
  parameters). BatchNorm layers in frozen stages are forced into `eval()` mode, so their running
  statistics remain ImageNet's.

## 3. Classifiers

- **A (text-only):** p = softmax(W · Dropout₀.₃(h_t) + b).
- **B (image-only):** p = softmax(W · Dropout₀.₃(h_v) + b).
- **C1 (concatenation fusion):**
  z_t = GELU(LN(W_t h_t)), z_v = GELU(LN(W_v h_v)), with z ∈ ℝ²⁵⁶ (each followed by dropout).
  f = [z_t ; z_v] ∈ ℝ⁵¹².
  p = softmax(W₂ · Dropout(GELU(W₁ · Dropout(f)))), with W₁: 512→256 and W₂: 256→3.
- **C2 (cross-attention fusion):**
  1. Project every text token (T × 768) and every image region (49 × 2048) to d = 256 with
     Linear + LayerNorm + GELU, giving T and V.
  2. Text→image: A_tv = MHA(Q = T, K = V, V = V), 4 heads. Each word attends over the 49
     regions. Then T' = LN(T + A_tv).
  3. Image→text: A_vt = MHA(Q = V, K = T, V = T, key-padding mask = padded tokens). Then
     V' = LN(V + A_vt).
  4. f = [mean_masked(T) ; mean(V) ; mean_masked(T') ; mean(V')] ∈ ℝ¹⁰²⁴, followed by the same
     MLP head as C1.
  5. Attention weights are exposed (`model.fusion.last_attention`) for inspection.
- **Both modalities are mandatory** in C1/C2. `forward()` raises an error if either input is
  missing. Unit tests verify that changing only the image, or only the text, changes the output.

## 4. Training (`src/training/trainer.py`) — identical for all experiments

| Setting | Value |
|---|---|
| Loss | Cross-entropy with class weights w_c = sqrt(N / (K · N_c)), normalised to mean 1 (train split) |
| Optimiser | AdamW; lr 2e-5 (XLM-R layers), 1e-4 (ResNet layer4), 1e-3 (new layers); weight decay 0.01, none on biases/norms |
| Schedule | Linear warm-up over 10 % of steps, then linear decay to 0 |
| Batch size | 16 (train), 32 (eval) |
| Epochs | at most 8; early stopping after 3 epochs without improvement in validation macro-F1 |
| Model selection | Checkpoint with the best **validation macro-F1** (the test set is never used for selection) |
| Regularisation | dropout 0.3, gradient-norm clipping 1.0 |
| Seed | 42 (Python, NumPy, PyTorch; DataLoader shuffling generator) |
| Hardware | Apple M1, 8 GB, PyTorch MPS backend |

**Why square-root class weights.** With 62.5 % negative and 6.6 % neutral, fully balanced weights
(×3.4 for neutral relative to the mean) tend to make the model over-predict the rare class. The
square root halves the correction on a log scale. This was a design choice made before training,
not tuned on the test set.

**Checkpoints** store only trained parameters and buffers (≈ 110–190 MB rather than 1.1 GB+).
Frozen weights are rebuilt from the pinned pretrained identifiers. `load_checkpoint` refuses to
load if any missing key is not on the recorded frozen list, so random weights can never be used
silently.

## 5. Evaluation (`src/evaluation/`)

- Metrics on the held-out test split (874 memes): accuracy; macro and weighted
  precision/recall/F1; per-class precision/recall/F1/support; confusion matrix.
  **Macro-F1 is the primary metric** because of class imbalance: a model that always predicts
  "negative" gets 62.6 % accuracy but a macro-F1 of 0.257.
- **Uncertainty:** 95 % percentile bootstrap confidence intervals (1,000 resamples of test
  memes) for accuracy and macro-F1.
- **Model differences:** a paired bootstrap over the same resamples gives the distribution of
  macro-F1(A) − macro-F1(B), its 95 % CI, and the one-sided proportion P(Δ ≤ 0). A difference is
  called significant only if the CI excludes 0.
- **Modality ablation** (multimodal models only): re-run the test set with (a) the image replaced
  by the ImageNet-mean image (an all-zero normalised tensor) and (b) the caption removed. We
  report the macro-F1 and the share of predictions that change. If (a) changed nothing, the
  model would be ignoring the image.
- **Cost:** wall-clock training time; per-meme inference latency at batch size 1 (median and p95
  over 100 test memes after 5 warm-up runs, including image loading, preprocessing and
  tokenisation, on the evaluation device); and eval-pass throughput.

## 6. Explainability (`src/explainability/`)

| Method | Applies to | What it measures | Caveats |
|---|---|---|---|
| **Occlusion** (leave-one-word-out; Li et al. 2016) | text, multimodal | P(ŷ \| caption) − P(ŷ \| caption without word w); the image is held fixed in multimodal mode | Exact for single deletions; ignores interactions such as negation + adjective; deleting a word also shifts the positions of later tokens |
| **Integrated Gradients** (Sundararajan et al. 2017) | text, multimodal | Path integral of ∂P(ŷ)/∂embedding from a `<pad>` baseline (special tokens kept), midpoint Riemann sum with 32 steps | Baseline-dependent. **Completeness check:** Σ attributions should equal P(x) − P(baseline). The residual is reported, and the result is marked unreliable (hidden in the app) if \|residual\| > max(0.02, 0.1·\|ΔP\|) |
| **Grad-CAM** (Selvaraju et al. 2017) | image, multimodal | ReLU(Σ_k α_k A_k), where α_k is the spatial mean of ∂logit_ŷ/∂A_k at ResNet `layer4` (7 × 7), upsampled and overlaid | Coarse resolution; often highlights caption text rendered in the image; shows sensitivity, not causation |

**Why IG gets a reliability flag.** While testing, I found that the IG residual on pretrained
XLM-R did **not** shrink as the number of steps grew. In float64 with a randomly initialised head,
128 steps happened to come close, but 512 steps was further off. I checked the cause:

- Autograd gradients matched central finite differences to 6 significant digits, so the gradients
  themselves are correct.
- Hidden states vary smoothly along the interpolation path through layer 8.
- Layers 9–12 change sharply within a narrow α interval: per-token hidden-state jumps of up to 8
  (L2 norm) per 0.0005 step in α.

A fixed-step Riemann sum cannot integrate such a function reliably. A unit test confirms that the
implementation does satisfy completeness on a smooth small network. On the real model, the app
therefore defaults to occlusion and shows IG only when its completeness check passes.

## 7. Inference (`src/inference.py`)

`SentimentPredictor.predict(text, image, experiment, explain)`:

1. Validates inputs per mode: text-only and multimodal require non-empty text; image-only and
   multimodal require an image. Images must be decodable, ≤ 10 MB, ≤ 40 MP, and have a minimum
   side ≥ 32 px.
2. Normalises text and runs script detection. It warns if the script is unsupported, if the text
   contains no letters, or if the text was truncated.
3. Loads the checkpoint lazily (at most two models cached in memory).
4. Returns the label, the confidence (the predicted-class probability), all three probabilities,
   the latency and the warnings, plus optional explanations.

If no checkpoint exists, it raises `ModelNotAvailable` with setup instructions. It never falls
back to untrained weights.

## 8. Reproducibility controls

- Fixed seed; deterministic, group-aware split. A test asserts that two runs give identical
  splits.
- Each checkpoint's `metadata.json` records the full config, dataset name, model identifiers,
  best epoch, validation score, parameter counts and environment (Python/torch/transformers
  versions, platform).
- `history.csv` and `training_summary.json` hold per-epoch losses and metrics.
- The evaluation reports record the checkpoint path and epoch that produced them.
- **Not controlled:** bit-exact GPU/MPS determinism. MPS kernels are not guaranteed to be
  deterministic, so retraining can differ slightly in the last digits.
