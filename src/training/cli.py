"""Shared argument parsing for the train_* entry points."""
from __future__ import annotations

import argparse

from ..utils import load_config


def parse_common(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="path to config.yaml (default: configs/config.yaml)")
    ap.add_argument("--epochs", type=int, default=None, help="override training.max_epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="override training.batch_size")
    ap.add_argument("--device", default=None, help="override device (auto|cpu|mps|cuda)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    return args, cfg
