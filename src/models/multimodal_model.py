"""Multimodal (text + image) sentiment model with concatenation or cross-attention fusion.

Both modalities are always encoded and both feed the classifier: there is no code path
in which a multimodal prediction is produced from a single modality.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .image_model import ImageEncoder
from .text_model import TextEncoder


def _projection(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class ConcatFusion(nn.Module):
    """Baseline: project pooled text and pooled image vectors to ``d`` and concatenate."""

    def __init__(self, text_dim: int, image_dim: int, d: int, dropout: float):
        super().__init__()
        self.text_proj = _projection(text_dim, d, dropout)
        self.image_proj = _projection(image_dim, d, dropout)
        self.out_dim = 2 * d

    def forward(self, text_pooled, text_tokens, attention_mask, image_pooled, image_grid):
        return torch.cat([self.text_proj(text_pooled), self.image_proj(image_pooled)], dim=-1)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-modal attention.

    Text tokens attend over the 7x7 ResNet region grid (which image regions are relevant to
    each word) and image regions attend over the text tokens. The fused vector is the
    concatenation of [pooled text, pooled image, pooled text->image, pooled image->text].
    """

    def __init__(self, text_dim: int, image_dim: int, d: int, heads: int, dropout: float):
        super().__init__()
        self.text_proj = _projection(text_dim, d, dropout)
        self.image_proj = _projection(image_dim, d, dropout)
        self.t2i = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.i2t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(d)
        self.norm_i = nn.LayerNorm(d)
        self.out_dim = 4 * d
        self.last_attention: dict[str, torch.Tensor] = {}

    def forward(self, text_pooled, text_tokens, attention_mask, image_pooled, image_grid):
        t = self.text_proj(text_tokens)                     # (B, T, d)
        v = self.image_proj(image_grid)                     # (B, R, d)
        pad = attention_mask == 0                           # True = ignore
        t2i, w_t2i = self.t2i(t, v, v, need_weights=True)   # text queries image regions
        i2t, w_i2t = self.i2t(v, t, t, key_padding_mask=pad, need_weights=True)
        t2i = self.norm_t(t + t2i)
        i2t = self.norm_i(v + i2t)
        self.last_attention = {"text_to_image": w_t2i.detach(), "image_to_text": w_i2t.detach()}
        return torch.cat([
            _masked_mean(t, attention_mask), v.mean(1),
            _masked_mean(t2i, attention_mask), i2t.mean(1),
        ], dim=-1)


class MultimodalClassifier(nn.Module):
    modality = "multimodal"

    def __init__(self, text_encoder: TextEncoder, image_encoder: ImageEncoder, num_classes: int = 3,
                 fusion: str = "concat", proj_dim: int = 256, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion_type = fusion
        if fusion == "concat":
            self.fusion = ConcatFusion(text_encoder.hidden_size, image_encoder.out_dim, proj_dim, dropout)
        elif fusion == "attention":
            self.fusion = CrossAttentionFusion(text_encoder.hidden_size, image_encoder.out_dim, proj_dim,
                                               heads, dropout)
        else:
            raise ValueError(f"Unknown fusion '{fusion}' (expected 'concat' or 'attention')")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(self.fusion.out_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, inputs_embeds=None,
                position_ids=None, **_):
        if pixel_values is None or (input_ids is None and inputs_embeds is None):
            raise ValueError("MultimodalClassifier requires both text and image inputs.")
        text_pooled, text_tokens = self.text_encoder(input_ids, attention_mask, inputs_embeds, position_ids)
        image_pooled, image_grid = self.image_encoder(pixel_values)
        fused = self.fusion(text_pooled, text_tokens, attention_mask, image_pooled, image_grid)
        return self.head(self.dropout(fused))
