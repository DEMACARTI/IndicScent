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

## Quick start

Requirements: Python 3.10–3.13, about 4 GB of free disk (data, pretrained weights and
checkpoints), and 8 GB RAM. A GPU (CUDA) or Apple Silicon (MPS) is used automatically if present.
The CPU works too, but training is slow.

### 1. Create the environment

**Linux / macOS**
```bash
cd IndicSent-M
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd IndicSent-M
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
artifacts/                 checkpoints/ (git-ignored), reports/ (metrics, plots, explanations)
```

## What is *not* included / limitations
- **Model weights and the MemoSen data are not committed.** The dataset has no explicit licence,
  so it is not redistributed. Regenerate both with the commands above.
- The models were trained on Bengali memes only. Hindi, Tamil and other inputs are accepted
  (XLM-R covers them) but are out-of-distribution, and the app warns about this.
- Captions must be supplied by the user. There is no OCR.
- Each model was trained with a single seed, so seed-to-seed variance has not been measured.
- See `docs/research_results.md` §Limitations for result-specific caveats.

## Citation
If you use the data, cite MemoSen:
> E. Hossain, O. Sharif, M. M. Hoque. MemoSen: A Multimodal Dataset for Sentiment Analysis of Memes. LREC 2022, pp. 1542–1554.
