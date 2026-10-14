"""Forward passes, output shapes, freezing, modality usage and checkpoint round-trips."""
import pytest
import torch

from src.models.factory import build_model
from src.models.image_model import ImageEncoder
from src.utils import load_checkpoint, save_checkpoint

B, T = 2, 7


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(5, 1000, (B, T), generator=g),
        "attention_mask": torch.tensor([[1] * T, [1] * (T - 2) + [0, 0]]),
        "pixel_values": torch.randn(B, 3, 224, 224, generator=g),
    }


@pytest.mark.parametrize("exp", ["text", "image", "multimodal_concat", "multimodal_attention"])
def test_forward_shapes_and_probabilities(cfg, tiny_text_config, exp):
    torch.manual_seed(0)
    model = build_model(cfg, exp, pretrained=False, text_hf_config=tiny_text_config).eval()
    with torch.no_grad():
        logits = model(**_inputs())
    assert logits.shape == (B, 3)
    probs = torch.softmax(logits, -1)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-6)
    assert torch.isfinite(logits).all()


def test_text_freezing(cfg, tiny_text_config):
    cfg["text"]["trainable_layers"] = 1
    m = build_model(cfg, "text", pretrained=False, text_hf_config=tiny_text_config)
    bb = m.encoder.backbone
    assert not any(p.requires_grad for p in bb.embeddings.parameters())
    assert not any(p.requires_grad for p in bb.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in bb.encoder.layer[-1].parameters())
    assert all(p.requires_grad for p in m.head.parameters())


def test_image_frozen_batchnorm_stays_in_eval():
    enc = ImageEncoder(trainable_blocks=("layer4",), pretrained=False)
    enc.train()
    assert not enc.layer1.training and not enc.stem.training
    assert enc.layer4.training
    assert not any(p.requires_grad for p in enc.layer3.parameters())
    assert all(p.requires_grad for p in enc.layer4.parameters())


@pytest.mark.parametrize("fusion", ["concat", "attention"])
def test_multimodal_uses_both_modalities(cfg, tiny_text_config, fusion):
    torch.manual_seed(0)
    m = build_model(cfg, f"multimodal_{fusion}", pretrained=False, text_hf_config=tiny_text_config).eval()
    x = _inputs(0)
    with torch.no_grad():
        base = m(**x)
        other_img = m(**{**x, "pixel_values": _inputs(1)["pixel_values"]})
        other_txt = m(**{**x, "input_ids": _inputs(2)["input_ids"]})
    assert not torch.allclose(base, other_img), "image input has no effect on the multimodal output"
    assert not torch.allclose(base, other_txt), "text input has no effect on the multimodal output"


def test_multimodal_requires_both_inputs(cfg, tiny_text_config):
    m = build_model(cfg, "multimodal_concat", pretrained=False, text_hf_config=tiny_text_config).eval()
    x = _inputs()
    with pytest.raises(ValueError):
        m(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
    with pytest.raises(ValueError):
        m(pixel_values=x["pixel_values"])


def test_attention_fusion_exposes_attention_weights(cfg, tiny_text_config):
    m = build_model(cfg, "multimodal_attention", pretrained=False, text_hf_config=tiny_text_config).eval()
    with torch.no_grad():
        m(**_inputs())
    att = m.fusion.last_attention
    assert att["text_to_image"].shape == (B, T, 49)
    assert att["image_to_text"].shape == (B, 49, T)
    # padded text tokens receive no attention from image regions
    assert torch.all(att["image_to_text"][1, :, -2:] == 0)


@pytest.mark.parametrize("exp", ["text", "multimodal_attention"])
def test_checkpoint_roundtrip(cfg, tiny_text_config, tmp_path, exp):
    torch.manual_seed(0)
    m1 = build_model(cfg, exp, pretrained=False, text_hf_config=tiny_text_config).eval()
    save_checkpoint(m1, tmp_path, {"experiment": exp})
    state = torch.load(tmp_path / "model.pt", weights_only=True)
    assert not any("embeddings.word_embeddings" in k for k in state), "frozen embeddings should not be saved"
    # same frozen weights (same seed for backbone), different trainable weights -> restored by load
    torch.manual_seed(0)
    m2 = build_model(cfg, exp, pretrained=False, text_hf_config=tiny_text_config).eval()
    with torch.no_grad():
        for p in m2.parameters():
            if p.requires_grad:
                p.add_(1.0)
    load_checkpoint(m2, tmp_path)
    x = _inputs()
    with torch.no_grad():
        assert torch.allclose(m1(**x), m2(**x), atol=1e-6)


def test_checkpoint_architecture_mismatch_raises(cfg, tiny_text_config, tmp_path):
    m1 = build_model(cfg, "multimodal_concat", pretrained=False, text_hf_config=tiny_text_config)
    save_checkpoint(m1, tmp_path, {})
    m2 = build_model(cfg, "multimodal_attention", pretrained=False, text_hf_config=tiny_text_config)
    with pytest.raises(RuntimeError):
        load_checkpoint(m2, tmp_path)


def test_integrated_gradients_completeness_on_smooth_model(cfg, tiny_text_config, tokenizer):
    """On a small smooth network the Riemann approximation must converge: sum of
    attributions == P(x) - P(baseline). (Pretrained XLM-R can be far less smooth; there the
    residual is reported and flagged instead — see text_explainer docstring.)"""
    from src.explainability.text_explainer import integrated_gradients
    torch.manual_seed(0)
    m = build_model(cfg, "multimodal_concat", pretrained=False, text_hf_config=tiny_text_config).eval()
    r = integrated_gradients(m, tokenizer, "আমি খুব খুশি 😄 aaj", 0, torch.device("cpu"),
                             pixel_values=torch.randn(1, 3, 224, 224), steps=128, chunk=32)
    total = sum(t["score"] for t in r["tokens"])
    diff = r["prob_input"] - r["prob_baseline"]
    assert abs(diff) > 1e-4
    assert total == pytest.approx(diff, rel=0.05, abs=1e-4)
    assert r["reliable"]
