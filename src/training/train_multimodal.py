"""Experiment C: text + image model with concatenation (baseline) or cross-attention fusion.

    python -m src.training.train_multimodal --fusion concat
    python -m src.training.train_multimodal --fusion attention
"""
from .cli import parse_common
from .trainer import train_experiment

if __name__ == "__main__":
    args, cfg = parse_common(__doc__, lambda ap: ap.add_argument(
        "--fusion", choices=["concat", "attention"], default="concat"))
    train_experiment(f"multimodal_{args.fusion}", cfg)
