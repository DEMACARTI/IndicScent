"""Evaluate trained checkpoints on the held-out test split.

Produces, per experiment, in artifacts/reports/<experiment>/:
  metrics.json            accuracy, macro/weighted P/R/F1, per-class report, confusion matrix,
                          latency, training time, and (multimodal) modality-ablation results
  predictions.csv         per-example probabilities and predictions
  confusion_matrix.png

    python -m src.evaluation.evaluate --experiments text image multimodal_concat multimodal_attention
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from ..data.dataset import load_split, make_loader, move_batch
from ..models.factory import EXPERIMENTS, checkpoint_dir, load_trained
from ..training.trainer import predict_loader
from ..utils import (display_path, get_device, load_config, load_json, now_iso, resolve_path, save_json,
                     sync_device)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    idx = list(range(len(labels)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=idx, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=idx, average="weighted",
                                                    zero_division=0)
    report = classification_report(y_true, y_pred, labels=idx, target_names=labels, output_dict=True,
                                   zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "weighted_precision": float(wp), "weighted_recall": float(wr), "weighted_f1": float(wf),
        "per_class": {l: {k: float(report[l][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for l in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def plot_confusion(cm: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    annot = np.array([[f"{cm[i, j]}\n({norm[i, j]:.0%})" for j in range(len(labels))] for i in range(len(labels))])
    sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def measure_latency(model, cfg, modality, tokenizer, df, device, n: int = 100, warmup: int = 5) -> dict:
    """Per-example latency at batch size 1, *including* image loading + preprocessing and
    tokenisation (what a user of the app experiences), on the evaluation device."""
    warmup = min(warmup, max(len(df) // 2, 0))   # tiny splits: still time at least half
    sub = df.head(n + warmup)
    loader = make_loader(cfg, "test", modality, tokenizer, train=False, df=sub, batch_size=1)
    times = []
    it = iter(loader)
    for i in range(len(sub)):
        sync_device(device)
        t0 = time.perf_counter()
        batch = move_batch(next(it), device)
        logits = model(**batch)
        torch.softmax(logits, -1).cpu()
        sync_device(device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(times)
    return {"device": str(device), "n": int(len(t)), "batch_size": 1,
            "median_ms": float(np.median(t)), "mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95))}


@torch.no_grad()
def modality_ablation(model, cfg, tokenizer, df, device, full_pred: np.ndarray, labels: list[str]) -> dict:
    """Does the multimodal model actually use both inputs? Re-run the test set with
    (a) the image replaced by a neutral ImageNet-mean image and (b) the text removed."""
    out = {}
    y = df["label_id"].to_numpy()
    # (a) neutral image: all-zero tensor after normalisation == the ImageNet mean colour
    loader = make_loader(cfg, "test", "multimodal", tokenizer, train=False, df=df)
    preds = []
    for batch in loader:
        b = move_batch(batch, device)
        b["pixel_values"] = torch.zeros_like(b["pixel_values"])
        preds.append(model(**b).argmax(1).cpu())
    p_img = torch.cat(preds).numpy()
    # (b) empty text
    df_nt = df.copy()
    df_nt["text"] = ""
    loader = make_loader(cfg, "test", "multimodal", tokenizer, train=False, df=df_nt)
    p_txt = predict_loader(model, loader, device, desc="ablate-text")[0].argmax(1).numpy()
    for name, p in {"image_neutralised": p_img, "text_removed": p_txt}.items():
        m = compute_metrics(y, p, labels)
        out[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "prediction_changed_rate": float((p != full_pred).mean())}
    return out


def evaluate_experiment(experiment: str, cfg: dict, split: str = "test", latency_n: int = 100) -> dict:
    device = get_device(cfg["device"])
    ckpt = checkpoint_dir(cfg, experiment)
    model, meta, tokenizer = load_trained(ckpt, device)
    mcfg = meta["config"]           # architecture/preprocessing used at training time
    mcfg["paths"] = cfg["paths"]
    labels = meta["labels"]
    modality = meta["modality"]
    df = load_split(cfg, split)
    loader = make_loader(mcfg, split, modality, tokenizer, train=False, df=df)

    sync_device(device)
    t0 = time.perf_counter()
    logits, y, ids = predict_loader(model, loader, device, desc=f"{experiment}:{split}")
    sync_device(device)
    eval_seconds = time.perf_counter() - t0
    probs = torch.softmax(logits, -1).numpy()
    y, pred = y.numpy(), probs.argmax(1)

    metrics = compute_metrics(y, pred, labels)
    metrics.update({
        "experiment": experiment, "split": split, "modality": modality, "fusion": meta.get("fusion"),
        "checkpoint": display_path(ckpt), "checkpoint_epoch": meta["epoch"],
        "val_macro_f1_at_selection": meta["val_macro_f1"], "dataset": meta["dataset"],
        "parameters": meta["parameters"], "evaluated_at": now_iso(),
        "eval_pass_seconds": eval_seconds,
        "throughput_examples_per_s": len(y) / eval_seconds,
    })
    summ_path = ckpt / "training_summary.json"
    if summ_path.exists():
        summ = load_json(summ_path)
        metrics["training_seconds"] = summ["training_seconds"]
        metrics["epochs_run"] = summ["epochs_run"]
        metrics["training_device"] = summ["device"]
    if latency_n > 0:
        metrics["latency"] = measure_latency(model, mcfg, modality, tokenizer, df, device, n=latency_n)
    if modality == "multimodal":
        metrics["modality_ablation"] = modality_ablation(model, mcfg, tokenizer, df, device, pred, labels)

    rep_dir = resolve_path(cfg["paths"]["reports_dir"]) / experiment
    rep_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, rep_dir / "metrics.json")
    pred_df = pd.DataFrame({"id": ids, "true": [labels[i] for i in y], "pred": [labels[i] for i in pred]})
    for i, l in enumerate(labels):
        pred_df[f"p_{l}"] = probs[:, i]
    pred_df.to_csv(rep_dir / "predictions.csv", index=False)
    plot_confusion(metrics["confusion_matrix"], labels, f"{experiment} ({split})", rep_dir / "confusion_matrix.png")
    print(f"[{experiment}] {split}: acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
          f"-> {rep_dir}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--latency-n", type=int, default=100)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for exp in args.experiments:
        if not (checkpoint_dir(cfg, exp) / "metadata.json").exists():
            print(f"[{exp}] skipped: no checkpoint at {checkpoint_dir(cfg, exp)} (train it first)")
            continue
        evaluate_experiment(exp, cfg, args.split, args.latency_n)


if __name__ == "__main__":
    main()
