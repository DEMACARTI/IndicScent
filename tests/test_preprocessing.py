"""Text normalisation, script detection, labels, image validation, tokenisation and the
dataset preparation pipeline (run on the SYNTHETIC sample)."""
import io

import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.image_utils import InvalidImageError, load_image
from src.data.text_processing import detect_scripts, normalize_text
from src.models.image_model import build_image_transforms
from src.utils import LabelEncoder


# ----------------------------------------------------------------------------- text
def test_normalize_preserves_sentiment_cues_and_scripts():
    s = "এটা ভালো না 😡!!!!!!  not good   at all"
    out = normalize_text(s)
    assert "😡" in out and "না" in out and "not" in out
    assert "!!!!" not in out and "!!!" in out          # emphasis kept, collapsed to 3
    assert "  " not in out


def test_normalize_keeps_zwj_zwnj_and_drops_bom():
    s = "﻿ক্‍ষ র‌্য"
    out = normalize_text(s)
    assert "‍" in out and "‌" in out
    assert "﻿" not in out


def test_normalize_urls_mentions_and_missing():
    assert normalize_text("see https://x.com/a @rahul_99 now") == "see URL @USER now"
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert normalize_text("   ") == ""


def test_normalize_is_nfc_and_keeps_case():
    decomposed = "é"
    assert normalize_text(decomposed) == "é"
    assert normalize_text("BEST Day") == "BEST Day"


@pytest.mark.parametrize("text,dominant,mixed", [
    ("আমি খুব খুশি", "Bengali", False),
    ("यह बहुत अच्छा है", "Devanagari", False),
    ("aaj ka din mast tha", "Latin", False),
    ("আজকে mood খুব off", "Bengali", True),
    ("😂😂 123", "None", False),
])
def test_detect_scripts(text, dominant, mixed):
    r = detect_scripts(text)
    assert r["dominant"] == dominant
    assert r["code_mixed"] == mixed
    assert r["supported"]


def test_detect_scripts_flags_unsupported():
    assert not detect_scripts("Привет как дела")["supported"]


# ----------------------------------------------------------------------------- labels
def test_label_encoder_roundtrip_and_errors():
    enc = LabelEncoder(["positive", "neutral", "negative"])
    for i, l in enumerate(["positive", "neutral", "negative"]):
        assert enc.encode(l) == i and enc.decode(i) == l
    assert enc.encode("  Positive ") == 0
    with pytest.raises(KeyError):
        enc.encode("very positive")
    with pytest.raises(KeyError):
        enc.decode(3)
    with pytest.raises(ValueError):
        LabelEncoder(["a", "a"])


# ----------------------------------------------------------------------------- images
def _png_bytes(size=(64, 48), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, 128).save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_valid_and_rgba_converted():
    img = load_image(_png_bytes(mode="RGBA"))
    assert img.mode == "RGB" and img.size == (64, 48)


def test_load_image_rejects_bad_inputs(tmp_path):
    with pytest.raises(InvalidImageError):
        load_image(b"not an image")
    with pytest.raises(InvalidImageError):
        load_image(b"")
    with pytest.raises(InvalidImageError):
        load_image(_png_bytes(size=(10, 10)))                      # too small
    with pytest.raises(InvalidImageError):
        load_image(_png_bytes(), max_bytes=10)                     # oversized upload
    with pytest.raises(InvalidImageError):
        load_image(tmp_path / "missing.png")
    truncated = _png_bytes(size=(200, 200))[:120]
    with pytest.raises(InvalidImageError):
        load_image(truncated)


def test_image_transforms_shape_and_normalisation():
    img = Image.new("RGB", (300, 120), (124, 116, 104))            # ~ImageNet mean colour
    x = build_image_transforms(224, train=False)(img)
    assert x.shape == (3, 224, 224)
    assert x.abs().max() < 0.05
    xt = build_image_transforms(224, train=True)(img)
    assert xt.shape == (3, 224, 224)


# ----------------------------------------------------------------------------- tokenisation
def test_tokenizer_handles_indic_and_code_mixed(tokenizer):
    for text in ["আমি খুব খুশি", "यह फिल्म अच्छी थी", "aaj ka din mast tha 😂"]:
        ids = tokenizer(text)["input_ids"]
        assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.sep_token_id
        assert tokenizer.unk_token_id not in ids[1:-1], f"unknown tokens for {text!r}"
    enc = tokenizer(["short", "word " * 300], padding=True, truncation=True, max_length=96, return_tensors="pt")
    assert enc["input_ids"].shape == (2, 96)
    assert enc["attention_mask"][0].sum() < 96


# ----------------------------------------------------------------------------- pipeline
@pytest.fixture
def prepared(cfg, tmp_path):
    from src.data.prepare import prepare
    cfg["dataset"]["name"] = "sample"
    cfg["paths"]["processed_dir"] = str(tmp_path / "processed")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    stats = prepare(cfg)
    read = lambda s: pd.read_csv(tmp_path / "processed" / f"{s}.csv", keep_default_na=False)
    return stats, {s: read(s) for s in ("train", "val", "test")}, tmp_path


def test_prepare_removes_invalid_and_keeps_missing_text(prepared):
    stats, splits, tmp = prepared
    allrows = pd.concat(splits.values())
    assert "synthetic_corrupt" not in set(allrows["id"]) and "synthetic_missing" not in set(allrows["id"])
    removed = pd.read_csv(tmp / "processed" / "removed.csv")
    assert set(removed["id"]) == {"synthetic_corrupt", "synthetic_missing"}
    nt = allrows[allrows["id"] == "synthetic_notext"]
    assert len(nt) == 1 and not nt["has_text"].iloc[0] and nt["text"].iloc[0] == ""
    assert stats["synthetic"] is True


def test_prepare_splits_are_disjoint_and_labels_consistent(prepared, cfg):
    _, splits, _ = prepared
    ids = [set(df["id"]) for df in splits.values()]
    groups = [set(df["group_id"]) for df in splits.values()]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not ids[i] & ids[j]
            assert not groups[i] & groups[j]
    enc = LabelEncoder(cfg["labels"])
    for df in splits.values():
        assert all(enc.encode(l) == i for l, i in zip(df["label"], df["label_id"]))


def test_prepare_is_reproducible(cfg, tmp_path):
    from src.data.prepare import prepare
    cfg["dataset"]["name"] = "sample"
    out = []
    for k in range(2):
        cfg["paths"]["processed_dir"] = str(tmp_path / f"p{k}")
        cfg["paths"]["reports_dir"] = str(tmp_path / f"r{k}")
        prepare(cfg)
        out.append(pd.read_csv(tmp_path / f"p{k}" / "all.csv")[["id", "split"]])
    pd.testing.assert_frame_equal(out[0], out[1])


def test_duplicates_grouped_into_same_split():
    from src.data.prepare import assign_groups, deduplicate
    df = pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "text": ["same long caption text here", "same long caption text here", "other", "x", "x"],
        "image_hash": ["h1", "h2", "h3", "h3", "h3"],
        "label": ["positive", "positive", "neutral", "negative", "positive"],
    })
    dedup, removed, st = deduplicate(df)
    assert st["conflicting_duplicates_removed"] == 2          # d/e: same image+text, different labels
    assert set(dedup["id"]) == {"a", "b", "c"}
    g = assign_groups(dedup.reset_index(drop=True))
    assert g.iloc[0] == g.iloc[1] and g.iloc[2] != g.iloc[0]   # a,b share text -> one group


def test_source_disjoint_split_keeps_sources_together():
    from src.data.prepare import split_by_source
    rows = []
    for src, labs in {"a": "ppnnn", "b": "pnnnn", "c": "nnnnnnnnnn", "d": "pppppp", "e": "pnn", "f": "uunnp"}.items():
        rows += [{"source": src, "label": {"p": "positive", "n": "negative", "u": "neutral"}[c]} for c in labs]
    df = pd.DataFrame(rows)
    split, info = split_by_source(df, 0.1, 0.2, ["positive", "neutral", "negative"])
    assert (df.assign(split=split).groupby("source")["split"].nunique() == 1).all()
    assert (split == "test").mean() >= 0.2 and (split == "val").mean() >= 0.1
    # the most label-skewed sources (all-negative c, all-positive d) are not chosen for test
    assert info["assignment"]["c"] != "test" and info["assignment"]["d"] != "test"


def test_config_inheritance():
    from src.utils import load_config, resolve_path
    c = load_config(resolve_path("configs/source_disjoint.yaml"))
    assert c["dataset"]["split"]["strategy"] == "source_disjoint"
    assert c["text"]["model_name"] == load_config()["text"]["model_name"]
    assert "base" not in c
