"""Word-level attributions for the text branch of the text-only and multimodal models.

Two complementary methods are provided:

1. **Occlusion (leave-one-word-out)** -- score(w) = P(target | text) - P(target | text without w).
   A direct, exact counterfactual measurement on the real model: no approximation error,
   but it ignores interactions between words (removing two words is not the sum of removing
   each). Positive = the word supports the predicted class.

2. **Integrated Gradients** (Sundararajan et al., 2017) over the input word embeddings with a
   baseline where each non-special token is replaced by <pad>. IG's attributions should sum
   to P(target|x) - P(target|baseline) ("completeness"). We approximate the path integral
   with a midpoint Riemann sum and report the residual. On pretrained XLM-R the upper layers
   can change very sharply along the interpolation path, so the residual can be large; when
   it is, the result is flagged ``reliable=False`` and should not be interpreted.

Both measure the model's *sensitivity*; neither proves why an annotator chose a label.
Sub-word scores are summed into whitespace-delimited words via the fast tokenizer's
``word_ids`` and character offsets, so Indic scripts and emojis are reconstructed exactly.
"""
from __future__ import annotations

import html

import torch

IG_RELIABLE_ABS = 0.02     # |residual| below this (probability units) is always acceptable
IG_RELIABLE_REL = 0.10     # otherwise it must be within 10 % of |P(x) - P(baseline)|


def _position_ids(model_type: str, input_ids: torch.Tensor, pad_id: int) -> torch.Tensor | None:
    if model_type in {"xlm-roberta", "roberta", "camembert"}:
        mask = (input_ids != pad_id).long()
        return torch.cumsum(mask, dim=1) * mask + pad_id
    return None


def _text_encoder(model):
    return model.text_encoder if hasattr(model, "text_encoder") else model.encoder


def _encode(tokenizer, text: str, max_length: int, device):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt",
                    return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    word_ids = enc.word_ids(0)
    input_ids = enc["input_ids"].to(device)
    special = torch.tensor(tokenizer.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True),
                           device=device).bool()
    # word id -> (token indices, char span)
    words: dict[int, dict] = {}
    for i, wid in enumerate(word_ids):
        if wid is None or special[i]:
            continue
        s, e = offsets[i]
        w = words.setdefault(wid, {"tokens": [], "start": s, "end": e})
        w["tokens"].append(i)
        w["start"], w["end"] = min(w["start"], s), max(w["end"], e)
    word_spans = [w for _, w in sorted(words.items())]
    for w in word_spans:
        w["word"] = text[w["start"]:w["end"]].strip()
    truncated = len(tokenizer(text)["input_ids"]) > max_length
    return input_ids, special, word_spans, truncated


def _prob_fn(model, target, pixel_values, device):
    extra = {} if pixel_values is None else {"pixel_values": pixel_values.to(device)}

    def prob(**kw):
        n = (kw.get("input_ids") if kw.get("input_ids") is not None else kw["inputs_embeds"]).shape[0]
        px = {k: v.expand(n, *v.shape[1:]) for k, v in extra.items()}
        return torch.softmax(model(**kw, **px).float(), -1)[:, target]
    return prob


@torch.no_grad()
def occlusion(model, tokenizer, text: str, target: int, device, max_length: int = 96,
              pixel_values: torch.Tensor | None = None) -> dict:
    model.eval()
    input_ids, _, spans, truncated = _encode(tokenizer, text, max_length, device)
    prob = _prob_fn(model, target, pixel_values, device)
    full = prob(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).item()
    scores = []
    for w in spans:
        keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)
        keep[w["tokens"]] = False
        ids = input_ids[:, keep]
        p = prob(input_ids=ids, attention_mask=torch.ones_like(ids)).item()
        scores.append({"word": w["word"], "score": full - p, "prob_without": p})
    return {"method": "Occlusion (leave-one-word-out)", "target": target, "words": scores,
            "prob_input": full, "truncated": truncated, "reliable": True}


def integrated_gradients(model, tokenizer, text: str, target: int, device, max_length: int = 96,
                         pixel_values: torch.Tensor | None = None, steps: int = 32, chunk: int = 8) -> dict:
    model.eval()
    input_ids, special, spans, truncated = _encode(tokenizer, text, max_length, device)
    mask = torch.ones_like(input_ids)
    base_ids = torch.where(special, input_ids[0], torch.full_like(input_ids[0], tokenizer.pad_token_id))[None]
    enc_mod = _text_encoder(model)
    emb_layer = enc_mod.word_embeddings()
    pos = _position_ids(enc_mod.backbone.config.model_type, input_ids, tokenizer.pad_token_id)
    with torch.no_grad():
        x, x0 = emb_layer(input_ids), emb_layer(base_ids)             # (1, T, H)
    prob_ = _prob_fn(model, target, pixel_values, device)

    def prob(emb):
        n = emb.shape[0]
        return prob_(inputs_embeds=emb, attention_mask=mask.expand(n, -1),
                     position_ids=None if pos is None else pos.expand(n, -1))

    alphas = (torch.arange(steps, device=device, dtype=x.dtype) + 0.5) / steps   # midpoint rule
    total_grad = torch.zeros_like(x)
    for a in alphas.split(chunk):
        e = (x0 + a.view(-1, 1, 1) * (x - x0)).detach().requires_grad_(True)
        g, = torch.autograd.grad(prob(e).sum(), e)
        total_grad += g.sum(0, keepdim=True)
    attr_tok = ((x - x0) * total_grad / steps).sum(-1)[0]            # (T,)
    with torch.no_grad():
        p_x, p_0 = prob(x).item(), prob(x0).item()
    delta = attr_tok.sum().item() - (p_x - p_0)
    reliable = abs(delta) <= max(IG_RELIABLE_ABS, IG_RELIABLE_REL * abs(p_x - p_0))
    words = [{"word": w["word"], "score": float(attr_tok[w["tokens"]].sum())} for w in spans]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return {
        "method": "Integrated Gradients (input embeddings, <pad> baseline)",
        "target": target, "words": words,
        "tokens": [{"token": t, "score": attr_tok[i].item()} for i, t in enumerate(tokens)],
        "prob_input": p_x, "prob_baseline": p_0, "convergence_delta": delta, "steps": steps,
        "reliable": bool(reliable), "truncated": truncated,
    }


def render_html(words: list[dict], target_label: str) -> str:
    """Words highlighted green (supports the predicted class) or red (opposes it)."""
    if not words:
        return "<i>No words to attribute.</i>"
    m = max(abs(w["score"]) for w in words) or 1.0
    spans = []
    for w in words:
        a = min(abs(w["score"]) / m, 1.0) * 0.85
        color = f"rgba(34,160,90,{a:.2f})" if w["score"] >= 0 else f"rgba(220,50,50,{a:.2f})"
        spans.append(f'<span title="{w["score"]:+.4f}" style="background:{color};padding:2px 4px;'
                     f'margin:1px;border-radius:4px;display:inline-block">{html.escape(w["word"])}</span>')
    legend = (f'<div style="font-size:0.8em;margin-top:6px;opacity:0.8">'
              f'<span style="background:rgba(34,160,90,0.6);padding:0 6px;border-radius:3px">green</span> '
              f'pushes towards <b>{html.escape(target_label)}</b> · '
              f'<span style="background:rgba(220,50,50,0.6);padding:0 6px;border-radius:3px">red</span> '
              f'pushes away · hover a word for its score</div>')
    return f'<div style="line-height:2.1;font-size:1.1em">{" ".join(spans)}</div>{legend}'
