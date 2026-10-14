#!/usr/bin/env bash
# Train the four experiments sequentially (Linux/macOS). Logs -> artifacts/logs/
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
mkdir -p artifacts/logs
run() {
  local name=$1; shift
  echo "=== $name start $(date)"
  PYTHONUNBUFFERED=1 "$PY" -m "$@" > "artifacts/logs/$name.log" 2>&1
  echo "=== $name exit $? $(date)"
}
run text                 src.training.train_text "$@"
run image                src.training.train_image "$@"
run multimodal_concat    src.training.train_multimodal --fusion concat "$@"
run multimodal_attention src.training.train_multimodal --fusion attention "$@"
