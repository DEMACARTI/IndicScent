# Reproducibility checklist

| Item | Status | Where |
|---|---|---|
| Exact package versions pinned (torch, torchvision, transformers) | ✅ | `requirements.txt` |
| Environment check script | ✅ | `scripts/check_environment.py` |
| Dataset source, archive size and SHA-256 recorded | ✅ | `data/raw/memosen_provenance.json`, `docs/dataset_documentation.md` §2 |
| Preprocessing fully scripted (no manual steps) | ✅ | `src/data/prepare.py` |
| Records removed during preparation listed with reasons | ✅ | `data/processed/removed.csv` |
| Deterministic, leakage-safe split (seed 42, group-aware) | ✅ | `prepare.split_groups`; test `test_prepare_is_reproducible` |
| Split files saved and identical across all experiments | ✅ | `data/processed/{train,val,test}.csv` |
| Random seeds set (Python, NumPy, PyTorch, DataLoader generator) | ✅ | `src/utils.set_seed`, `dataset.make_loader` |
| All hyper-parameters in one config file | ✅ | `configs/config.yaml` |
| Resolved config + model identifiers + environment saved with each checkpoint | ✅ | `artifacts/checkpoints/<exp>/metadata.json` |
| Pretrained weights pinned (HF id `FacebookAI/xlm-roberta-base`, torchvision `IMAGENET1K_V2`) | ✅ | config + metadata |
| Per-epoch training history | ✅ | `artifacts/checkpoints/<exp>/history.csv`, `training_summary.json` |
| Model selection on validation only (the test set is never used for selection) | ✅ | `trainer.train_experiment` |
| Evaluation reports name the checkpoint path and epoch | ✅ | `artifacts/reports/<exp>/metrics.json` |
| Per-example test predictions saved | ✅ | `artifacts/reports/<exp>/predictions.csv` |
| Uncertainty reported (bootstrap CIs, paired tests) | ✅ | `artifacts/reports/comparison/` |
| Result tables generated from outputs, not typed by hand | ✅ | `compare_models.py` → `comparison.md` |
| Automated tests | ✅ | `pytest -q` |
| One-command pipeline | ✅ | `python scripts/run_pipeline.py` |
| Container | ✅ (not built in this session; see `research_results.md`) | `Dockerfile` |
| Multiple training seeds / variance across seeds | ❌ not done (compute budget) | future work |
| Bit-exact determinism on GPU/MPS | ❌ not guaranteed by PyTorch MPS/CUDA kernels | — |
| Hyper-parameter search | ❌ not performed; a single fixed config was set before training | — |

## Re-running from scratch

```bash
python scripts/check_environment.py
python scripts/run_pipeline.py
pytest -q
```

## Verifying a result

1. Open `artifacts/reports/<exp>/metrics.json` and check `checkpoint` and `checkpoint_epoch`.
2. Open `artifacts/checkpoints/<exp>/metadata.json` and check `config`, `dataset`, `environment`
   and `epoch`.
3. Run `python -m src.evaluation.evaluate --experiments <exp>`. The metrics should match exactly,
   because evaluation is deterministic for a fixed checkpoint (up to last-digit float noise
   across devices).
