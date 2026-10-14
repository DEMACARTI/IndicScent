"""Inference API: missing checkpoints, input validation, prediction consistency, explainers.

Checkpoints are built with the *real* architectures (pretrained XLM-R / ResNet-50 from the
local cache) and a randomly initialised head, saved to a temp dir and loaded back through the
same code path the app uses. They test plumbing only — not model quality.
"""
import io

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import pretrained_available
from src.inference import InputError, ModelNotAvailable, SentimentPredictor
from src.models.factory import build_model
from src.utils import save_checkpoint

needs_weights = pytest.mark.skipif(not pretrained_available(), reason="pretrained weights not cached locally")


def _png(color=(200, 30, 30), size=(120, 90)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def empty_predictor(cfg, tmp_path):
    cfg["paths"]["checkpoints_dir"] = str(tmp_path / "none")
    return SentimentPredictor(cfg=cfg, device="cpu")


def test_missing_checkpoint_raises_with_instructions(empty_predictor):
    assert empty_predictor.available() == {}
    with pytest.raises(ModelNotAvailable, match="train"):
        empty_predictor.predict("hello", None, "text")


def test_input_validation_happens_before_model_loading(empty_predictor):
    with pytest.raises(InputError):
        empty_predictor.predict("", None, "text")
    with pytest.raises(InputError):
        empty_predictor.predict("   ", _png(), "multimodal_concat")
    with pytest.raises(InputError):
        empty_predictor.predict(None, None, "image")
    with pytest.raises(InputError):
        empty_predictor.predict("ok", None, "multimodal_concat")
    with pytest.raises(InputError, match="could not be read"):
        empty_predictor.predict(None, b"corrupt bytes", "image")
    with pytest.raises(InputError):
        empty_predictor.predict("x", None, "not_a_model")


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    """Real-architecture checkpoints for image + multimodal_attention (random heads)."""
    if not pretrained_available():
        pytest.skip("pretrained weights not cached locally")
    from src.utils import load_config
    cfg = load_config()
    cfg["device"] = "cpu"
    root = tmp_path_factory.mktemp("ckpt")
    cfg["paths"]["checkpoints_dir"] = str(root)
    torch.manual_seed(0)
    for exp in ["image", "multimodal_attention", "text"]:
        m = build_model(cfg, exp)
        save_checkpoint(m, root / exp, {"experiment": exp, "modality": m.modality,
                                        "fusion": "attention" if exp.startswith("multi") else None,
                                        "epoch": 0, "val_macro_f1": 0.0, "labels": cfg["labels"],
                                        "config": cfg, "dataset": "unit-test",
                                        "text_model": cfg["text"]["model_name"]})
    return cfg


@needs_weights
def test_prediction_is_deterministic_and_well_formed(trained_dir):
    p = SentimentPredictor(cfg=trained_dir, device="cpu")
    assert set(p.available()) == {"image", "multimodal_attention", "text"}
    r1 = p.predict("আজকের দিনটা দারুণ 😄", _png(), "multimodal_attention")
    r2 = p.predict("আজকের দিনটা দারুণ 😄", _png(), "multimodal_attention")
    assert r1.probabilities == r2.probabilities
    assert abs(sum(r1.probabilities.values()) - 1) < 1e-5
    assert r1.label == max(r1.probabilities, key=r1.probabilities.get)
    assert r1.confidence == pytest.approx(r1.probabilities[r1.label])
    assert r1.script_info["dominant"] == "Bengali"
    # both modalities influence the multimodal prediction
    r_img = p.predict("আজকের দিনটা দারুণ 😄", _png((20, 200, 20)), "multimodal_attention")
    r_txt = p.predict("একদম বাজে সিনেমা", _png(), "multimodal_attention")
    assert r_img.probabilities != r1.probabilities
    assert r_txt.probabilities != r1.probabilities


@needs_weights
def test_explanations(trained_dir):
    p = SentimentPredictor(cfg=trained_dir, device="cpu")
    r = p.predict("यह फिल्म अच्छी नहीं थी", _png(), "multimodal_attention", explain=True, ig_steps=64)
    words = ["यह", "फिल्म", "अच्छी", "नहीं", "थी"]
    occ = r.text_occlusion
    assert [w["word"] for w in occ["words"]] == words
    for w in occ["words"]:                                   # occlusion is an exact counterfactual
        assert w["score"] == pytest.approx(occ["prob_input"] - w["prob_without"])
    assert occ["prob_input"] == pytest.approx(r.probabilities[r.label], abs=1e-5)
    ig = r.text_attribution
    assert [w["word"] for w in ig["words"]] == words
    total = sum(t["score"] for t in ig["tokens"])
    assert ig["convergence_delta"] == pytest.approx(total - (ig["prob_input"] - ig["prob_baseline"]))
    assert isinstance(ig["reliable"], bool)
    assert r.gradcam_raw.shape == (7, 7)
    assert 0 <= r.gradcam_raw.min() and r.gradcam_raw.max() <= 1 + 1e-6
    assert r.gradcam.size == (120, 90)
    ri = p.predict(None, _png(), "image", explain=True)
    assert ri.text_attribution is None and ri.gradcam is not None
    assert isinstance(r.to_dict()["gradcam_raw"], list)


@needs_weights
def test_corrupt_checkpoint_reported(trained_dir, tmp_path):
    import shutil
    from pathlib import Path
    src = Path(trained_dir["paths"]["checkpoints_dir"]) / "image"
    dst = tmp_path / "image"
    shutil.copytree(src, dst)
    (dst / "model.pt").write_bytes(b"garbage")
    cfg = {**trained_dir, "paths": {**trained_dir["paths"], "checkpoints_dir": str(tmp_path)}}
    with pytest.raises(ModelNotAvailable, match="could not be loaded"):
        SentimentPredictor(cfg=cfg, device="cpu").predict(None, _png(), "image")
