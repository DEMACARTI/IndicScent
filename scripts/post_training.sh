#!/usr/bin/env bash
# One-off orchestration used for the reported run: waits for the main training, then
# re-prepares (adds the 'source' column; split must be unchanged), evaluates, compares,
# trains + evaluates the source-disjoint robustness experiment, and renders explanations.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
until grep -q "=== multimodal_attention exit" artifacts/logs/training_run.log; do sleep 20; done
for f in text image multimodal_concat multimodal_attention; do cp artifacts/logs/$f.log artifacts/logs/main_$f.log; done
echo "=== re-prepare main $(date)"
$PY -m src.data.prepare > artifacts/logs/prepare_main.log 2>&1 || exit 1
$PY - <<'PY' || exit 1
import pandas as pd
a = pd.read_csv("artifacts/logs/split_before_reprepare.csv")
b = pd.read_csv("data/processed/all.csv")[["id","split","label_id","text","image_path"]]
pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
print("split identical after re-prepare:", len(b), "rows")
PY
echo "=== evaluate main $(date)"
$PY -m src.evaluation.evaluate > artifacts/logs/evaluate_main.log 2>&1 || exit 1
$PY -m src.evaluation.compare_models > artifacts/logs/compare_main.log 2>&1 || exit 1
echo "=== source-disjoint training $(date)"
./scripts/train_all.sh --config configs/source_disjoint.yaml > artifacts/logs/training_run_source_disjoint.log 2>&1
for f in text image multimodal_concat multimodal_attention; do mv artifacts/logs/$f.log artifacts/logs/sd_$f.log 2>/dev/null; done
echo "=== evaluate source-disjoint $(date)"
$PY -m src.evaluation.evaluate --config configs/source_disjoint.yaml > artifacts/logs/evaluate_sd.log 2>&1 || exit 1
$PY -m src.evaluation.compare_models --config configs/source_disjoint.yaml > artifacts/logs/compare_sd.log 2>&1 || exit 1
echo "=== explanations $(date)"
$PY -m src.explainability.generate_examples > artifacts/logs/explanations.log 2>&1 || exit 1
echo "=== all done $(date)"
