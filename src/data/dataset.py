"""PyTorch dataset/collator shared by all three experiments.

All experiments read the same processed split files, so text-only, image-only and
multimodal models are trained and evaluated on exactly the same examples.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..models.image_model import build_image_transforms
from ..utils import resolve_path

SPLITS = ("train", "val", "test")


def load_split(cfg: dict, split: str) -> pd.DataFrame:
    path = resolve_path(cfg["paths"]["processed_dir"]) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.prepare` first (see README)."
        )
    df = pd.read_csv(path, keep_default_na=False, dtype={"text": str})
    return df


class MemeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, modality: str, image_transform=None):
        if modality not in {"text", "image", "multimodal"}:
            raise ValueError(modality)
        self.df = df.reset_index(drop=True)
        self.modality = modality
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item = {"label": int(row["label_id"]), "id": str(row["id"])}
        if self.modality in {"text", "multimodal"}:
            item["text"] = str(row["text"])
        if self.modality in {"image", "multimodal"}:
            with Image.open(resolve_path(row["image_path"])) as im:
                img = im.convert("RGB")
            item["pixel_values"] = self.image_transform(img)
        return item


class Collator:
    def __init__(self, tokenizer=None, max_length: int = 96):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        out = {"labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
               "ids": [b["id"] for b in batch]}
        if "text" in batch[0]:
            enc = self.tokenizer([b["text"] for b in batch], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            out["input_ids"] = enc["input_ids"]
            out["attention_mask"] = enc["attention_mask"]
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
        return out


def make_loader(cfg: dict, split: str, modality: str, tokenizer=None, train: bool = False,
                df: pd.DataFrame | None = None, batch_size: int | None = None) -> DataLoader:
    df = load_split(cfg, split) if df is None else df
    tfm = build_image_transforms(cfg["image"]["image_size"], train=train)
    ds = MemeDataset(df, modality, tfm)
    tc = cfg["training"]
    bs = batch_size or (tc["batch_size"] if train else tc["eval_batch_size"])
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    return DataLoader(ds, batch_size=bs, shuffle=train, generator=g if train else None,
                      num_workers=tc["num_workers"],
                      collate_fn=Collator(tokenizer, cfg["text"]["max_length"]))


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
