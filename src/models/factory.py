"""Build any of the experiment models from the config, optionally restoring a checkpoint."""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from .image_model import ImageClassifier, ImageEncoder
from .multimodal_model import MultimodalClassifier
from .text_model import TextClassifier, TextEncoder

# experiment name -> (modality, fusion)
EXPERIMENTS = {
    "text": ("text", None),
    "image": ("image", None),
    "multimodal_concat": ("multimodal", "concat"),
    "multimodal_attention": ("multimodal", "attention"),
}


def build_model(cfg: dict, experiment: str, pretrained: bool = True, text_hf_config=None) -> nn.Module:
    """``pretrained=False`` builds the same architecture with random weights (tests only)."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose from {list(EXPERIMENTS)}")
    modality, fusion = EXPERIMENTS[experiment]
    n_cls = len(cfg["labels"])
    dropout = cfg["training"]["dropout"]

    def text_encoder():
        tc = cfg["text"]
        return TextEncoder(tc["model_name"], tc["trainable_layers"], tc["pooling"],
                           hf_config=None if pretrained else text_hf_config)

    def image_encoder():
        return ImageEncoder(cfg["image"]["trainable_blocks"], pretrained=pretrained)

    if modality == "text":
        return TextClassifier(text_encoder(), n_cls, dropout)
    if modality == "image":
        return ImageClassifier(image_encoder(), n_cls, dropout)
    mc = cfg["multimodal"]
    return MultimodalClassifier(text_encoder(), image_encoder(), n_cls, fusion, mc["proj_dim"],
                                mc["attention_heads"], dropout)


def modality_of(experiment: str) -> str:
    return EXPERIMENTS[experiment][0]


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def checkpoint_dir(cfg: dict, experiment: str) -> Path:
    from ..utils import resolve_path
    return resolve_path(cfg["paths"]["checkpoints_dir"]) / experiment


def load_trained(ckpt_dir: str | Path, device=None):
    """Rebuild a trained model from its checkpoint directory.

    The architecture is reconstructed from the config stored *in the checkpoint*, not the
    current config file, so editing config.yaml cannot silently mismatch a saved model.
    Returns ``(model, metadata, tokenizer_or_None)``.
    """
    import torch
    from transformers import AutoTokenizer

    from ..utils import METADATA_FILE, load_checkpoint, load_json

    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / METADATA_FILE).exists():
        raise FileNotFoundError(f"No trained checkpoint in {ckpt_dir}")
    meta = load_json(ckpt_dir / METADATA_FILE)
    model = build_model(meta["config"], meta["experiment"])
    load_checkpoint(model, ckpt_dir)
    model.eval()
    if device is not None:
        model.to(device)
    tok = None
    if meta["modality"] != "image":
        tok_dir = ckpt_dir / "tokenizer"
        tok = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else meta["text_model"])
    return model, meta, tok
