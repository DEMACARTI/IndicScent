"""Generic training loop shared by the text, image and multimodal experiments.

* AdamW with separate learning rates for the text encoder, image encoder and new layers
  (no weight decay on biases/LayerNorm).
* Linear warm-up + linear decay schedule.
* Cross-entropy with optional (square-root) inverse-frequency class weights.
* Gradient clipping, early stopping and best-checkpoint selection on validation macro-F1.
* Per-epoch history logged to CSV/JSON; the resolved config is stored with the checkpoint.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..data.dataset import load_split, make_loader, move_batch
from ..models.factory import EXPERIMENTS, build_model, checkpoint_dir, count_parameters, modality_of
from ..utils import (environment_info, get_device, load_config, now_iso, save_checkpoint, save_json,
                     set_seed, sync_device)


def class_weights(labels: np.ndarray, n_classes: int, scheme: str) -> torch.Tensor | None:
    if scheme == "none":
        return None
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)          # sklearn "balanced"
    if scheme == "sqrt_balanced":
        w = np.sqrt(w)                               # softer: avoids over-predicting rare classes
    elif scheme != "balanced":
        raise ValueError(scheme)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_param_groups(model: nn.Module, tc: dict) -> list[dict]:
    lrs = {"text": tc["lr_backbone"], "image": tc["lr_image_backbone"], "head": tc["lr_head"]}
    groups: dict[tuple[str, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("text_encoder.") or (model.modality == "text" and name.startswith("encoder.")):
            kind = "text"
        elif name.startswith("image_encoder.") or (model.modality == "image" and name.startswith("encoder.")):
            kind = "image"
        else:
            kind = "head"
        no_decay = p.ndim < 2  # biases and norm weights
        groups.setdefault((kind, no_decay), []).append(p)
    return [{"params": ps, "lr": lrs[k], "weight_decay": 0.0 if nd else tc["weight_decay"], "name": f"{k}"}
            for (k, nd), ps in groups.items()]


@torch.no_grad()
def predict_loader(model: nn.Module, loader, device: torch.device, desc: str = "eval"):
    model.eval()
    all_logits, all_labels, all_ids = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        b = move_batch(batch, device)
        logits = model(**b)
        all_logits.append(logits.float().cpu())
        all_labels.append(batch["labels"])
        all_ids.extend(batch["ids"])
    return torch.cat(all_logits), torch.cat(all_labels), all_ids


def train_experiment(experiment: str, cfg: dict | None = None, config_path: str | None = None,
                     output_dir: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(cfg) if cfg is not None else load_config(config_path)
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment}")
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    tc = cfg["training"]
    modality = modality_of(experiment)
    out_dir = Path(output_dir) if output_dir else checkpoint_dir(cfg, experiment)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model_name"]) if modality != "image" else None
    train_df, val_df = load_split(cfg, "train"), load_split(cfg, "val")
    train_loader = make_loader(cfg, "train", modality, tokenizer, train=True, df=train_df)
    val_loader = make_loader(cfg, "val", modality, tokenizer, train=False, df=val_df)

    model = build_model(cfg, experiment).to(device)
    params = count_parameters(model)
    weights = class_weights(train_df["label_id"].to_numpy(), len(cfg["labels"]), tc["class_weighting"])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.AdamW(build_param_groups(model, tc))
    total_steps = len(train_loader) * tc["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tc["warmup_ratio"] * total_steps), total_steps)

    print(f"[{experiment}] device={device} train={len(train_df)} val={len(val_df)} "
          f"params trainable={params['trainable']:,}/{params['total']:,}")
    history, best_f1, best_epoch, bad_epochs = [], -1.0, -1, 0
    t_start = time.perf_counter()
    for epoch in range(1, tc["max_epochs"] + 1):
        model.train()
        t0, running, n_seen = time.perf_counter(), 0.0, 0
        for batch in tqdm(train_loader, desc=f"{experiment} epoch {epoch}", leave=False):
            b = move_batch(batch, device)
            logits = model(**b)
            loss = criterion(logits, b["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tc["grad_clip"])
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(b["labels"])
            n_seen += len(b["labels"])
        sync_device(device)
        train_time = time.perf_counter() - t0

        logits, labels, _ = predict_loader(model, val_loader, device, desc="val")
        val_loss = nn.functional.cross_entropy(logits, labels, weight=weights).item()
        preds = logits.argmax(1).numpy()
        val_f1 = f1_score(labels.numpy(), preds, average="macro", zero_division=0)
        val_acc = accuracy_score(labels.numpy(), preds)
        rec = {"epoch": epoch, "train_loss": running / max(n_seen, 1), "val_loss": val_loss,
               "val_accuracy": val_acc, "val_macro_f1": val_f1, "epoch_train_seconds": train_time}
        history.append(rec)
        print(f"[{experiment}] epoch {epoch}: train_loss={rec['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} ({train_time:.0f}s)")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            save_checkpoint(model, out_dir, {
                "experiment": experiment, "modality": modality,
                "fusion": EXPERIMENTS[experiment][1], "epoch": epoch,
                "val_macro_f1": val_f1, "val_accuracy": val_acc,
                "labels": cfg["labels"], "config": cfg, "parameters": params,
                "text_model": cfg["text"]["model_name"] if modality != "image" else None,
                "image_weights": "torchvision ResNet50_Weights.IMAGENET1K_V2" if modality != "text" else None,
                "dataset": cfg["dataset"]["name"], "environment": environment_info(),
            })
            if tokenizer is not None:
                tokenizer.save_pretrained(out_dir / "tokenizer")
        else:
            bad_epochs += 1
            if bad_epochs >= tc["patience"]:
                print(f"[{experiment}] early stopping after epoch {epoch} (best epoch {best_epoch})")
                break
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    total_time = time.perf_counter() - t_start
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {"experiment": experiment, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
               "epochs_run": len(history), "training_seconds": total_time, "device": str(device),
               "finished_at": now_iso(), "class_weights": weights.tolist() if weights is not None else None,
               "parameters": params, "history": history}
    save_json(summary, out_dir / "training_summary.json")
    print(f"[{experiment}] done in {total_time / 60:.1f} min, best val macro-F1={best_f1:.4f} @ epoch {best_epoch}")
    return summary
