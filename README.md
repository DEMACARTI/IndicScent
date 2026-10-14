# IndicSent-M: Multimodal Sentiment Analysis for Indian Languages Using Text and Images

IndicSent-M classifies Indian-language memes as **positive / neutral / negative** from their
caption, their image, or both. It then explains its predictions with word attributions and
Grad-CAM heatmaps.

It trains and compares four models on the same leakage-safe split of the **MemoSen** Bengali
meme dataset (Hossain et al., LREC 2022):

| Experiment | Model |
|---|---|
| A | Text-only: XLM-RoBERTa base (top 4 layers fine-tuned) |
| B | Image-only: ResNet-50 (layer4 fine-tuned) |
| C1 | Multimodal: XLM-R + ResNet-50, concatenation fusion |
| C2 | Multimodal: XLM-R + ResNet-50, bidirectional cross-attention fusion |

➡ **Results:** [`docs/research_results.md`](docs/research_results.md). The tables there are
generated from `artifacts/reports/`.
➡ Dataset: [`docs/dataset_documentation.md`](docs/dataset_documentation.md) ·
Method: [`docs/methodology.md`](docs/methodology.md) ·
Slides: [`docs/IndicSent-M_slides.pptx`](docs/IndicSent-M_slides.pptx) / [PDF](docs/IndicSent-M_slides.pdf) (regenerate: `python scripts/make_slides.py`) ·
6-page report: [`docs/IndicSent-M_Project_Report.pdf`](docs/IndicSent-M_Project_Report.pdf) (regenerate: `python scripts/make_report_pdf.py`) ·
Demo guide: [`docs/demo_guide.md`](docs/demo_guide.md) ·
Reproducibility: [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md) ·
References: [`docs/references.md`](docs/references.md)

---

## Trained models (Git LFS)

All four trained models are in this repository, so you can run the app or call the models
**without downloading the dataset or retraining**. The weight files (`*.pt`) are stored with
[Git LFS](https://git-lfs.com) because they are larger than GitHub's 100 MB per-file limit. The
tokenizers, metadata and reports are ordinary git files.

| Folder | Contents |
|---|---|
| `artifacts/checkpoints/` | Main models (random group split), used by default |
| `artifacts/checkpoints_source_disjoint/` | The same four models trained on the source-disjoint split (robustness check) |
| `artifacts/reports/`, `artifacts/reports_source_disjoint/` | Metrics, confusion matrices, comparison plots, explanation gallery (read by the app) |

Each checkpoint folder holds `model.pt` (LFS), `metadata.json` (training config, epoch,
validation score), `frozen_keys.json`, `history.csv`, `training_summary.json` and, for text
models, a `tokenizer/` folder. The app and API need the whole folder, not just `model.pt`.

| Experiment name | Input | Weights | Test macro-F1 | Test accuracy |
|---|---|---|---|---|
| `text` | caption | 108 MB | 0.545 | 0.690 |
| `image` | image | 57 MB | 0.503 | 0.681 |
| `multimodal_concat` | caption + image | 169 MB | 0.578 | 0.743 |
| `multimodal_attention` | caption + image | 171 MB | 0.589 | 0.743 |

(Main split, 874 test memes. Full results with confidence intervals are in `docs/research_results.md`.)

### Get the models

**1. Install Git LFS once per machine**

| OS | Command |
|---|---|
| macOS | `brew install git-lfs` |
| Ubuntu / Debian | `sudo apt install git-lfs` |
| Windows | Included with [Git for Windows](https://git-scm.com/download/win) |

Then run `git lfs install` once.

**2. Clone.** The models download automatically, about 1.1 GB:
```bash
git clone https://github.com/DEMACARTI/IndicScent.git
cd IndicScent
git lfs ls-files        # "*" after the ID = downloaded, "-" = still a pointer file
```

**Already cloned without LFS?** Then each `model.pt` is a tiny text "pointer" file (about 130 bytes), and
the app reports that a checkpoint could not be loaded. Fix it from inside the repo:
```bash
git lfs install
git lfs pull
```

**Only want some models?** Clone without the weights, then fetch only what you need:
```bash
# Linux/macOS
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/DEMACARTI/IndicScent.git
# Windows PowerShell
$env:GIT_LFS_SKIP_SMUDGE=1; git clone https://github.com/DEMACARTI/IndicScent.git

cd IndicScent
git lfs pull --include="artifacts/checkpoints/multimodal_attention/*"
git lfs pull --include="artifacts/checkpoints/*"            # all four main models
```

**Without git:** download a weight file directly and put it in its checkpoint folder, e.g.
`https://media.githubusercontent.com/media/DEMACARTI/IndicScent/main/artifacts/checkpoints/text/model.pt`.
Note that GitHub's "Download ZIP" button gives pointer files instead of the real weights.

**Pretrained base weights.** To keep the checkpoints small, they store only the fine-tuned layers.
The frozen layers are rebuilt from `FacebookAI/xlm-roberta-base` (about 1.1 GB, cached in
`~/.cache/huggingface`) and torchvision's ResNet-50 (about 100 MB, cached in `~/.cache/torch`), which are
downloaded automatically the **first time** a model loads. This needs internet once. After that you
can work offline with `HF_HUB_OFFLINE=1`.

### Use the models

Set up the environment ([Quick start](#quick-start) step 1), then skip straight to step 6:
```bash
streamlit run app.py
```

**From Python.** `SentimentPredictor` handles text normalisation, tokenisation, image validation and
resizing, device selection (CUDA → MPS → CPU) and model caching:
```python
from src.inference import SentimentPredictor

p = SentimentPredictor()                      # main models; device="cpu" to force CPU
print(p.available())                          # which checkpoints are on disk

r = p.predict(text="আজকের দিনটা দারুণ", image="meme.jpg", experiment="multimodal_attention")
print(r.label, r.confidence, r.probabilities) # label, its probability, all three class probabilities

p.predict(text="আজকের দিনটা দারুণ", experiment="text")      # text-only
p.predict(image="meme.jpg", experiment="image")            # image-only

r = p.predict(text="...", image="meme.jpg", experiment="multimodal_attention", explain=True)
r.text_attribution, r.text_occlusion          # per-word importance
r.gradcam                                     # PIL image with the Grad-CAM heatmap

sd = SentimentPredictor(config_path="configs/source_disjoint.yaml")   # source-disjoint models
```
`image` accepts a file path, raw bytes, a file-like object or a `PIL.Image`. Invalid input raises
`src.inference.InputError`. A missing or unreadable checkpoint raises `src.inference.ModelNotAvailable`.

**From another project.** Put this repo on the Python path, then import as above. Model paths
are resolved relative to the repo, so it works from any working directory:
```python
import sys
sys.path.insert(0, "/path/to/IndicScent")
from src.inference import SentimentPredictor
```
(or `export PYTHONPATH=/path/to/IndicScent`; Windows: `$env:PYTHONPATH="C:\path\to\IndicScent"`).
Your environment needs the packages in `requirements.txt`. For `config_path`, use an absolute
path when calling from outside the repo.

**From any language (REST API):**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health                                   # loaded models + device
curl -F "model=multimodal_attention" -F "text=আজকের দিনটা দারুণ" -F "image=@meme.jpg" \
     http://localhost:8000/predict
```
`/predict` takes form fields `model` (one of the experiment names above), `text`, `image` and
`explain` (true/false). It returns JSON with `label`, `confidence`, `probabilities`, `latency_ms`,
`device`, `warnings`, plus explanation fields when `explain=true`. Errors: `422` for bad input,
`503` for a missing model, `413` for an image over 10 MB. Interactive docs: `http://localhost:8000/docs`.

**Docker:** `docker build -t indicsent-m .`, then mount the checkpoints as in [step 8](#8-docker-cpu).

**Low level:** `src.models.factory.load_trained("artifacts/checkpoints/text", device)` returns
`(torch_model, metadata, tokenizer)` if you want the raw `nn.Module`. You must then reproduce the
preprocessing yourself (see `src/inference.py`).

---

## Quick start

Requirements: Python 3.10–3.13, about 4 GB of free disk (data, pretrained weights and
checkpoints), and 8 GB RAM. A GPU (CUDA) or Apple Silicon (MPS) is used automatically if present.
The CPU works too, but training is slow.

### 1. Create the environment

**Linux / macOS**
```bash
cd IndicScent          # the cloned repo folder
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd IndicScent          # the cloned repo folder
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For an NVIDIA GPU, install the matching CUDA wheel first, e.g.
`pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126`.

All commands below are identical on every OS once the venv is active.

### 2. Check the environment
```bash
python scripts/check_environment.py
```

### 3. Get and prepare the data
```bash
python -m src.data.download --dataset memosen   # ~590 MB from the authors' Google Drive link
python -m src.data.prepare                      # validate, normalise, dedupe, split, stats + figures
```
If the Google Drive download is blocked (quota), download `MemoSen.zip` manually from the link
in <https://github.com/eftekhar-hossain/Multimodal-Sentiment-LREC2022>, save it as
`data/raw/memosen_download/memosen.zip`, and rerun the download command, which then only extracts.

Pipeline smoke test with the **synthetic** sample (results are meaningless):
`python -m src.data.prepare --dataset sample`

### 4. Train
```bash
python -m src.training.train_text
python -m src.training.train_image
python -m src.training.train_multimodal --fusion concat
python -m src.training.train_multimodal --fusion attention
```
Options: `--epochs N`, `--batch-size N`, `--device cpu|mps|cuda`, `--config path.yaml`.
On an Apple M1 (8 GB) the four runs together took the times listed in `docs/research_results.md`.

### 5. Evaluate, compare, explain
```bash
python -m src.evaluation.evaluate             # metrics.json, predictions.csv, confusion matrix per model
python -m src.evaluation.compare_models       # comparison table, bootstrap CIs, paired tests, plots
python -m src.explainability.generate_examples  # Grad-CAM + word-attribution gallery for test memes
```

### Everything at once
```bash
python scripts/run_pipeline.py                       # check → download → prepare → train ×4 → evaluate → compare → explain
python scripts/run_pipeline.py --skip download prepare   # resume
```
(`scripts/train_all.sh` is a bash alternative for the four training runs, with logs written to `artifacts/logs/`.)

### 6. Run the app
```bash
streamlit run app.py          # http://localhost:8501
```
The app starts even without trained checkpoints. It then shows setup instructions; an optional
UI preview is labelled **DEMONSTRATION — NO MODEL WAS RUN**. It never serves predictions from
untrained weights.

Optional REST API:
```bash
uvicorn src.api:app --port 8000
curl -F "model=multimodal_concat" -F "text=আজকের দিনটা দারুণ" -F "image=@meme.jpg" http://localhost:8000/predict
```

### 7. Tests
```bash
pytest -q
```
The tests cover preprocessing, labels, tokenisation, image validation, forward passes and output
shapes of all four architectures, that both modalities influence the multimodal output,
checkpoint round-trips, inference input errors, prediction determinism, and explainer
correctness (IG completeness, occlusion exactness, Grad-CAM shape/range). Tests that need the
pretrained XLM-R/ResNet weights are skipped if those are not in the local cache.

### 8. Docker (CPU)
```bash
docker build -t indicsent-m .
# Linux/macOS
docker run -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts" -v "$(pwd)/data:/app/data" indicsent-m
# Windows PowerShell
docker run -p 8501:8501 -v "${PWD}/artifacts:/app/artifacts" -v "${PWD}/data:/app/data" indicsent-m
```
Checkpoints and data are mounted, not baked into the image. To run the API instead, append
`uvicorn src.api:app --host 0.0.0.0 --port 8000` and publish port 8000.

---

## Repository layout
```
app.py                     Streamlit UI (Home, Analyze, Model Comparison, Dataset & Methodology, About)
configs/config.yaml        all hyper-parameters and paths
src/data/                  download, prepare (validation, dedupe, group split), dataset/collator,
                           text_processing (normalisation, script ID), image_utils, make_sample
src/models/                text_model, image_model, multimodal_model (concat / cross-attention), factory
src/training/              trainer (shared loop) + train_text / train_image / train_multimodal
src/evaluation/            evaluate (metrics, latency, modality ablation), compare_models (bootstrap)
src/explainability/        text_explainer (occlusion, Integrated Gradients), image_explainer (Grad-CAM),
                           generate_examples (gallery)
src/inference.py           SentimentPredictor used by app + API
src/api.py                 FastAPI service
scripts/                   check_environment.py, run_pipeline.py, train_all.sh
tests/                     pytest suite
docs/                      methodology, dataset documentation, results, slides, demo guide, checklist, references
data/sample/               SYNTHETIC examples (pipeline/UI testing only)
artifacts/                 checkpoints/ (trained weights, Git LFS), reports/ (metrics, plots, explanations)
```

## What is *not* included / limitations
- **The MemoSen data is not committed.** The dataset has no explicit licence, so it is not
  redistributed. The trained weights are included (see [Trained models](#trained-models-git-lfs)).
  Download the data with the commands above only if you want to retrain or re-evaluate.
- The models were trained on Bengali memes only. Hindi, Tamil and other inputs are accepted
  (XLM-R covers them) but are out-of-distribution, and the app warns about this.
- Captions must be supplied by the user. There is no OCR.
- Each model was trained with a single seed, so seed-to-seed variance has not been measured.
- See `docs/research_results.md` §Limitations for result-specific caveats.

## Citation
If you use the data, cite MemoSen:
> E. Hossain, O. Sharif, M. M. Hoque. MemoSen: A Multimodal Dataset for Sentiment Analysis of Memes. LREC 2022, pp. 1542–1554.
