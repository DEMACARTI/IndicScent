"""Shared helpers: configuration, seeding, devices, label mapping, checkpoint I/O."""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


# --------------------------------------------------------------------------- config
def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML config and apply nested ``overrides`` (dict of dicts).

    A config may contain ``base: other.yaml`` (relative to its own folder); it is loaded first
    and the file's own keys override it. Used for experiment variants.
    """
    path = Path(path or DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base = cfg.pop("base", None)
    if base:
        merged = load_config(path.parent / base)
        _deep_update(merged, cfg)
        cfg = merged
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def resolve_path(p: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root (works on Windows and Linux)."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- labels
class LabelEncoder:
    """Bidirectional mapping between canonical label strings and class indices."""

    def __init__(self, labels: list[str]):
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate labels: {labels}")
        self.labels = list(labels)
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for i, l in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)

    def encode(self, label: str) -> int:
        key = str(label).strip().lower()
        if key not in self.label2id:
            raise KeyError(f"Unknown label '{label}'. Expected one of {self.labels}")
        return self.label2id[key]

    def decode(self, idx: int) -> str:
        if int(idx) not in self.id2label:
            raise KeyError(f"Unknown class index {idx}")
        return self.id2label[int(idx)]


# --------------------------------------------------------------------------- runtime
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Block until queued kernels finish so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def environment_info() -> dict:
    import transformers
    import torchvision

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        info["git_commit"] = None
    return info


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- checkpoints
# Checkpoints store only parameters that were trained plus all buffers. Frozen
# parameters are rebuilt from the pinned pretrained weights (Hugging Face revision /
# torchvision weight enum recorded in the metadata), which keeps a text checkpoint at
# ~100 MB instead of ~1.1 GB. ``load_checkpoint`` verifies that every key missing from
# the file is one that was frozen during training, so nothing is silently random.
CHECKPOINT_FILE = "model.pt"
METADATA_FILE = "metadata.json"


def save_checkpoint(model: torch.nn.Module, out_dir: str | os.PathLike, metadata: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n for n, _ in model.named_buffers()}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable or k in buffers}
    frozen = sorted(k for k in model.state_dict() if k not in state)
    torch.save(state, out_dir / CHECKPOINT_FILE)
    meta = copy.deepcopy(metadata)
    meta["frozen_keys_count"] = len(frozen)
    meta["saved_keys_count"] = len(state)
    meta["saved_at"] = now_iso()
    save_json(meta, out_dir / METADATA_FILE)
    with open(out_dir / "frozen_keys.json", "w", encoding="utf-8") as fh:
        json.dump(frozen, fh)
    return out_dir / CHECKPOINT_FILE


def load_checkpoint(model: torch.nn.Module, ckpt_dir: str | os.PathLike) -> dict:
    ckpt_dir = Path(ckpt_dir)
    state = torch.load(ckpt_dir / CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    frozen = set(load_json(ckpt_dir / "frozen_keys.json"))
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing_not_frozen = [k for k in result.missing_keys if k not in frozen]
    if unexpected or missing_not_frozen:
        raise RuntimeError(
            f"Checkpoint at {ckpt_dir} does not match the model architecture. "
            f"Unexpected keys: {unexpected[:5]}... Missing (non-frozen) keys: {missing_not_frozen[:5]}..."
        )
    return load_json(ckpt_dir / METADATA_FILE)


def display_path(p: str | os.PathLike) -> str:
    """Project-relative POSIX path when possible (portable in reports), else absolute."""
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)
