# Demonstration guide (≈ 10 minutes)

## Before the demo
1. `source .venv/bin/activate` (Windows: `.\.venv\Scripts\Activate.ps1`)
2. `python scripts/check_environment.py`. All four checkpoints should show **ok**.
3. `streamlit run app.py` and open http://localhost:8501.
4. Do a first analysis before the audience arrives. The first call loads XLM-R (~5–10 s).

> ⚠️ The "random test meme" button shows real MemoSen memes, and some contain crude or offensive
> humour. Pre-select examples if the audience requires it. The explanation gallery in
> `artifacts/reports/explanations/` also works as an offline backup.

## Script

| Step | Page | What to show | What to say |
|---|---|---|---|
| 1 | Home | Objective, modalities, architecture diagram | The research question: does text + image beat text or image alone on Indian-language memes? |
| 2 | Dataset & Methodology | Split sizes, label imbalance chart, script chart | 62 % negative, 7 % neutral, so we use macro-F1. 21 % of captions mix Bengali and Latin script. Splits are grouped to prevent template leakage. |
| 3 | Analyze → Text-only | Type a Bengali sentence, e.g. `আমার মন খারাপ` ("I feel sad") | Prediction, confidence, and the "confidence ≠ correctness" caption. Word highlights from occlusion. |
| 4 | Analyze → Text-only | Code-mixed input, e.g. `exam e fail korlam 😭` | The script detector reports Latin and explains it cannot tell English from romanised Bengali. |
| 5 | Analyze → Multimodal | 🎲 Load a random test meme → Analyze | Compare the annotated label with the prediction. Grad-CAM heatmap plus word highlights. |
| 6 | Analyze → Image-only | The same meme | How the image-only model differs. Grad-CAM often lights up the caption text in the image. |
| 7 | Analyze | Upload a non-image file renamed `.png`, or leave the text empty in multimodal mode | Graceful error messages. |
| 8 | Model Comparison | Table, CI bar chart, confusion matrices, paired bootstrap, modality ablation | Read the auto-generated findings, which come from the real results. Point out where differences are *not* significant. |
| 9 | Model Comparison → Explanation gallery | 1–2 figures | Limitations of explanations: sensitivity ≠ causation; IG completeness check. |
| 10 | About | Implemented vs. future work | Memotion 3 (Hinglish), OCR, CLIP, multiple seeds. |

## Fallbacks
- **No checkpoints:** the app shows setup steps. The DEMO checkbox previews the layout only and
  is clearly labelled as such.
- **No network:** everything runs offline once the pretrained weights are cached
  (set `HF_HUB_OFFLINE=1`).
- **Slow machine:** switch off "Compute explanations" for faster predictions.
