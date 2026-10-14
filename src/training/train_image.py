"""Experiment B: fine-tune ResNet-50 (image only).

    python -m src.training.train_image [--epochs N] [--config configs/config.yaml]
"""
from .cli import parse_common
from .trainer import train_experiment

if __name__ == "__main__":
    _, cfg = parse_common(__doc__)
    train_experiment("image", cfg)
