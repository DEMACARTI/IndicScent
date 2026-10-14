"""Text-only sentiment model: multilingual Transformer encoder + classification head."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Wraps a Hugging Face encoder (XLM-R by default) and returns pooled + token features.

    Partial fine-tuning: embeddings and the lower ``num_layers - trainable_layers`` encoder
    layers are frozen. This keeps optimiser memory small enough for an 8 GB laptop and
    limits over-fitting on a few thousand memes.
    """

    def __init__(self, model_name: str, trainable_layers: int = 4, pooling: str = "mean",
                 hf_config=None):
        super().__init__()
        if hf_config is None:
            self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        else:  # unit tests: same architecture family, random weights, no download
            self.backbone = AutoModel.from_config(hf_config, add_pooling_layer=False)
        self.model_name = model_name
        self.pooling = pooling
        self.hidden_size = self.backbone.config.hidden_size
        self.trainable_layers = trainable_layers
        self._apply_freezing(trainable_layers)

    def _apply_freezing(self, trainable_layers: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = self.backbone.encoder.layer
        n = len(layers)
        k = max(0, min(trainable_layers, n)) if trainable_layers >= 0 else n
        for layer in layers[n - k:]:
            for p in layer.parameters():
                p.requires_grad = True

    def word_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, position_ids=None):
        out = self.backbone(input_ids=None if inputs_embeds is not None else input_ids,
                            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                            position_ids=position_ids)
        tokens = out.last_hidden_state                                  # (B, T, H)
        if self.pooling == "cls":
            pooled = tokens[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled, tokens


class TextClassifier(nn.Module):
    modality = "text"

    def __init__(self, encoder: TextEncoder, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.hidden_size, num_classes)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, position_ids=None, **_):
        pooled, _ = self.encoder(input_ids, attention_mask, inputs_embeds, position_ids)
        return self.head(self.dropout(pooled))
