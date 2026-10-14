"""Experiment A: fine-tune the multilingual Transformer (text only).

    python -m src.training.train_text [--epochs N] [--config configs/config.yaml]
"""
from .cli import parse_common
from .trainer import train_experiment

if __name__ == "__main__":
    _, cfg = parse_common(__doc__)
    train_experiment("text", cfg)
