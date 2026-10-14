"""Dataset preparation: load -> validate -> normalise -> de-duplicate -> split -> report.

    python -m src.data.prepare                       # dataset from configs/config.yaml
    python -m src.data.prepare --dataset sample      # SYNTHETIC sample (pipeline testing only)

Outputs (data/processed/): train.csv, val.csv, test.csv, all.csv, removed.csv, stats.json
Figures (artifacts/reports/dataset/): label distribution, text length, scripts, image sizes.

Leakage prevention: memes that share an (almost) identical image or an identical non-trivial
text are linked into the same group (union-find) and a whole group always lands in one split.
Exact duplicates (same image and text) are collapsed; if their labels disagree they are
dropped as unreliable annotations.
"""
from __future__ import annotations

import argparse
import copy
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from ..utils import LabelEncoder, display_path, load_config, now_iso, resolve_path, save_json, set_seed
from .image_utils import InvalidImageError, load_image
from .text_processing import detect_scripts, normalize_text

MIN_TEXT_CHARS_FOR_GROUPING = 12   # short captions like "😂" or "lol" are not evidence of a duplicate


# =========================================================================== loaders
# Each loader returns a DataFrame with columns:
#   id, text_raw, image_abs (str path), label_raw, official_split (or None), synthetic (bool)

def load_csv_dataset(csv_path: Path, image_dir: Path, text_col: str, image_col: str, label_col: str,
                     synthetic: bool = False, split_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
    missing = [c for c in (text_col, image_col, label_col) if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} lacks columns {missing}; available: {list(df.columns)}. "
                       "Set dataset.csv.*_column in configs/config.yaml")
    ids = df["id"] if "id" in df.columns else df[image_col].map(lambda s: Path(s).stem)
    return pd.DataFrame({
        "id": ids.astype(str),
        "text_raw": df[text_col],
        "image_abs": [str(image_dir / p) if p else "" for p in df[image_col]],
        "label_raw": df[label_col],
        "official_split": df[split_col] if split_col and split_col in df.columns else None,
        "synthetic": synthetic,
        "source": "synthetic" if synthetic else "",
    })


def load_memosen(root: Path) -> pd.DataFrame:
    """MemoSen release layout (verified on the archive with sha256 483f3940...):
    Dataset/multi-sent.xlsx  (columns: image_name, Captions, Label_Sentiment)
    Dataset/Memes/<image_name>
    The release has no official train/val/test split."""
    xlsx = root / "Dataset" / "multi-sent.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"{xlsx} not found. Run `python -m src.data.download --dataset memosen` first.")
    df = pd.read_excel(xlsx, dtype=str).fillna("")
    img_dir = root / "Dataset" / "Memes"
    names = df["image_name"].str.strip()
    return pd.DataFrame({
        "id": "memosen_" + names.str.replace(r"[^\w\-.]+", "_", regex=True),
        "text_raw": df["Captions"],
        "image_abs": [str(img_dir / n) if n else "" for n in names],
        "label_raw": df["Label_Sentiment"],
        "official_split": None,
        "synthetic": False,
        # Source page, inferred from the file-name prefix ("KAM (12).jpg" -> "KAM"). Used for the
        # source-prior baseline and the source-disjoint robustness split; never a model input.
        "source": names.str.extract(r"^(.+?)\s*\(", expand=False).fillna("unknown"),
    })


def load_raw(cfg: dict) -> tuple[pd.DataFrame, dict]:
    ds = cfg["dataset"]
    name = ds["name"]
    if name == "sample":
        root = resolve_path(cfg["paths"]["sample_dir"])
        if not (root / "labels.csv").exists():
            from .make_sample import main as make_sample
            make_sample()
        df = load_csv_dataset(root / "labels.csv", root / "images", "text", "image", "label", synthetic=True)
        label_map = {"positive": "positive", "neutral": "neutral", "negative": "negative"}
    elif name == "memosen":
        df = load_memosen(resolve_path(ds["memosen"]["root"]))
        label_map = {"positive": "positive", "neutral": "neutral", "negative": "negative"}
    elif name == "csv":
        c = ds["csv"]
        df = load_csv_dataset(resolve_path(c["csv_path"]), resolve_path(c["image_dir"]), c["text_column"],
                              c["image_column"], c["label_column"])
        label_map = {str(k).lower(): v for k, v in c["label_map"].items()}
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    return df, label_map


# =========================================================================== validation
def dhash(img: Image.Image, size: int = 16) -> str:
    """Difference hash (256 bits): robust to resizing / re-encoding, not to crops or edits."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def validate(df: pd.DataFrame, label_map: dict, enc: LabelEncoder, vcfg: dict,
             cache_dir: Path | None = None, cache_max_side: int = 448) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate records; optionally write a resized RGB JPEG copy of each image to ``cache_dir``
    (longest side ``cache_max_side``) so training does not decode multi-megapixel files every epoch."""
    keep, removed = [], []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples(index=False):
        rec = r._asdict()
        label = label_map.get(str(rec["label_raw"]).strip().lower())
        if label is None or label not in enc.label2id:
            removed.append({**rec, "reason": f"unknown label '{rec['label_raw']}'"})
            continue
        if not rec["image_abs"]:
            removed.append({**rec, "reason": "missing image reference"})
            continue
        try:
            img = load_image(rec["image_abs"], max_bytes=None, max_pixels=vcfg["max_image_pixels"],
                             min_side=vcfg["min_image_side"])
        except InvalidImageError as e:
            removed.append({**rec, "reason": f"invalid image: {e}"})
            continue
        cached = ""
        if cache_dir is not None:
            cached_path = cache_dir / f"{rec['id']}.jpg"
            if not cached_path.exists():
                small = img.copy()
                small.thumbnail((cache_max_side, cache_max_side), Image.BICUBIC)
                small.save(cached_path, format="JPEG", quality=95)
            cached = str(cached_path)
        text = normalize_text(rec["text_raw"])
        s = detect_scripts(text)
        keep.append({
            **rec, "text": text, "has_text": bool(text), "label": label, "label_id": enc.encode(label),
            "image_w": img.width, "image_h": img.height, "image_hash": dhash(img), "image_cached": cached,
            "script": s["dominant"], "scripts": "+".join(sorted(s["scripts"])), "code_mixed": s["code_mixed"],
            "n_chars": len(text), "n_words": len(text.split()),
        })
    return pd.DataFrame(keep), pd.DataFrame(removed)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Collapse exact duplicates (same image hash + same text) and drop label-conflicting ones."""
    df = df.copy()
    df["_key"] = df["image_hash"] + "|" + df["text"].str.lower()
    keep_idx, removed = [], []
    n_dup, n_conflict = 0, 0
    for _, g in df.groupby("_key", sort=False):
        if len(g) == 1:
            keep_idx.append(g.index[0])
            continue
        if g["label"].nunique() == 1:
            keep_idx.append(g.index[0])
            n_dup += len(g) - 1
            removed += [{**r, "reason": "exact duplicate"} for r in g.iloc[1:].to_dict("records")]
        else:
            n_conflict += len(g)
            removed += [{**r, "reason": "duplicate with conflicting labels"} for r in g.to_dict("records")]
    out = df.loc[sorted(keep_idx)].drop(columns="_key")
    rem = pd.DataFrame(removed).drop(columns="_key", errors="ignore")
    return out, rem, {"exact_duplicates_removed": n_dup, "conflicting_duplicates_removed": n_conflict}


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Union-find over shared image hash or shared (long enough) text -> group id."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for key in ("image_hash", "text"):
        first: dict[str, int] = {}
        for i, v in enumerate(df[key].tolist()):
            if key == "text" and len(v) < MIN_TEXT_CHARS_FOR_GROUPING:
                continue
            v = v.lower() if key == "text" else v
            if v in first:
                union(first[v], i)
            else:
                first[v] = i
    return pd.Series([find(i) for i in range(len(df))], index=df.index)


def split_groups(df: pd.DataFrame, val_size: float, test_size: float, seed: int) -> pd.Series:
    """Stratified (by majority label of each group) group-level train/val/test split."""
    g = df.groupby("group_id")["label_id"].agg(lambda s: Counter(s).most_common(1)[0][0])
    groups, strat = g.index.to_numpy(), g.to_numpy()
    strat_ok = lambda y, n: min(Counter(y).values()) >= 2 and n >= len(set(y))
    n_test = max(1, int(round(test_size * len(groups))))
    tr_val, te = train_test_split(groups, test_size=n_test, random_state=seed,
                                  stratify=strat if strat_ok(strat, n_test) else None)
    strat_tv = g.loc[tr_val].to_numpy()
    n_val = max(1, int(round(val_size * len(groups))))
    tr, va = train_test_split(tr_val, test_size=n_val, random_state=seed,
                              stratify=strat_tv if strat_ok(strat_tv, n_val) else None)
    m = {**{x: "train" for x in tr}, **{x: "val" for x in va}, **{x: "test" for x in te}}
    return df["group_id"].map(m)


def split_by_source(df: pd.DataFrame, val_size: float, test_size: float, labels: list[str]) -> tuple[pd.Series, dict]:
    """Source-disjoint split: every source page lands entirely in one split.

    Selection rule (fixed before any model was evaluated on this split): rank sources by the
    Jensen-Shannon divergence between their label distribution and the global one, then
    greedily assign the most "typical" sources to test until it holds >= ``test_size`` of the
    data, then to validation until >= ``val_size``. Choosing typical sources keeps the label
    distribution of the held-out sets close to the training one, so the experiment measures
    the effect of unseen *sources* rather than of a different label mix.
    """
    from scipy.spatial.distance import jensenshannon
    glob = df["label"].value_counts(normalize=True).reindex(labels).fillna(0).to_numpy()
    by_src = pd.crosstab(df["source"], df["label"]).reindex(columns=labels).fillna(0)
    js = {src: float(jensenshannon(row / row.sum(), glob)) for src, row in by_src.iterrows()}
    order = sorted(js, key=lambda k: (js[k], k))
    sizes = by_src.sum(1)
    n = len(df)
    assign, test_n, val_n = {}, 0, 0
    for src in order:
        if test_n < test_size * n:
            assign[src], test_n = "test", test_n + sizes[src]
        elif val_n < val_size * n:
            assign[src], val_n = "val", val_n + sizes[src]
        else:
            assign[src] = "train"
    return df["source"].map(assign), {"js_divergence": js, "assignment": assign}


# =========================================================================== reporting
def make_figures(df: pd.DataFrame, labels: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    out.mkdir(parents=True, exist_ok=True)
    order = ["train", "val", "test"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ct = pd.crosstab(df["split"], df["label"]).reindex(index=order, columns=labels).fillna(0)
    ct.plot(kind="bar", ax=ax, color=["#55A868", "#8C8C8C", "#C44E52"])
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    ax.set_title("Label distribution per split")
    ax.set_xlabel("")
    ax.set_ylabel("memes")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out / "label_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="n_words", hue="label", hue_order=labels, bins=40, multiple="stack", ax=ax,
                 palette=["#55A868", "#8C8C8C", "#C44E52"])
    ax.set_title("Text length (whitespace words)")
    fig.tight_layout()
    fig.savefig(out / "text_length.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = df["scripts"].replace("", "None").value_counts().head(10)
    sns.barplot(x=sc.values, y=sc.index, ax=ax, color="#4C72B0")
    ax.set_title("Scripts present in text (≥15% of letters)")
    ax.set_xlabel("memes")
    fig.tight_layout()
    fig.savefig(out / "script_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(df["image_w"], df["image_h"], s=4, alpha=0.4)
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Image sizes")
    fig.tight_layout()
    fig.savefig(out / "image_sizes.png", dpi=150)
    plt.close(fig)


def token_length_stats(texts: list[str], model_name: str) -> dict | None:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return None
    lens = np.array([len(tok(t)["input_ids"]) for t in texts])
    return {"tokenizer": model_name, "mean": float(lens.mean()), "median": float(np.median(lens)),
            "p95": float(np.percentile(lens, 95)), "max": int(lens.max())}


# =========================================================================== main
def prepare(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    enc = LabelEncoder(cfg["labels"])
    raw, label_map = load_raw(cfg)
    n_raw = len(raw)
    if raw["id"].duplicated().any():
        raise ValueError(f"Duplicate record ids: {raw.loc[raw['id'].duplicated(), 'id'].tolist()[:5]}")
    print(f"Loaded {n_raw} raw records ({cfg['dataset']['name']})")
    vc = cfg["dataset"]["validation"]
    cache_side = cfg["dataset"].get("image_cache_max_side")
    cache_dir = resolve_path("data/cache") / f"{cfg['dataset']['name']}_{cache_side}" if cache_side else None
    valid, removed = validate(raw, label_map, enc, vc, cache_dir, cache_side or 0)
    print(f"Valid: {len(valid)}  removed: {len(removed)}")
    valid, dup_removed, dup_stats = deduplicate(valid)
    removed = pd.concat([removed, dup_removed], ignore_index=True)
    valid = valid.reset_index(drop=True)
    valid["group_id"] = assign_groups(valid)

    sc = cfg["dataset"]["split"]
    source_info = None
    official = valid["official_split"]
    if sc["use_official_split"] and official.notna().all() and set(official) <= {"train", "val", "test"}:
        valid["split"] = official
        split_source = "official"
        leaking = valid.groupby("group_id")["split"].nunique()
        leak_groups = int((leaking > 1).sum())
    elif sc.get("strategy", "group_stratified") == "source_disjoint":
        if (valid["source"] == "").all():
            raise ValueError("source_disjoint split needs a `source` column (available for MemoSen).")
        valid["split"], source_info = split_by_source(valid, sc["val_size"], sc["test_size"], cfg["labels"])
        split_source = "source-disjoint (sources ranked by label-distribution JS divergence)"
        # The same meme re-posted by two pages would leak across splits: drop the val/test copies
        # of any near-duplicate group that already occurs in an earlier split (train < val < test).
        rank = valid["split"].map({"train": 0, "val": 1, "test": 2})
        first = rank.groupby(valid["group_id"]).transform("min")
        leaked = rank > first
        source_info["cross_split_duplicates_removed"] = int(leaked.sum())
        removed = pd.concat([removed, valid[leaked].assign(reason="cross-source duplicate of an earlier split")],
                            ignore_index=True)
        valid = valid[~leaked].reset_index(drop=True)
        leak_groups = int((valid.groupby("group_id")["split"].nunique() > 1).sum())
    else:
        valid["split"] = split_groups(valid, sc["val_size"], sc["test_size"], cfg["seed"])
        split_source = f"group-stratified random (seed={cfg['seed']})"
        leak_groups = int((valid.groupby("group_id")["split"].nunique() > 1).sum())

    proc = resolve_path(cfg["paths"]["processed_dir"])
    if proc.exists():
        shutil.rmtree(proc)
    proc.mkdir(parents=True)
    valid["image_source"] = [display_path(p) for p in valid["image_abs"]]
    valid["image_path"] = [display_path(c or p) for c, p in zip(valid["image_cached"], valid["image_abs"])]
    cols = ["id", "text", "image_path", "image_source", "label", "label_id", "has_text", "split", "group_id", "source", "script",
            "scripts", "code_mixed", "n_chars", "n_words", "image_w", "image_h", "image_hash", "synthetic"]
    valid = valid[cols]
    for s in ("train", "val", "test"):
        valid[valid["split"] == s].to_csv(proc / f"{s}.csv", index=False)
    valid.to_csv(proc / "all.csv", index=False)
    if len(removed):
        removed.drop(columns=["image_abs"], errors="ignore").to_csv(proc / "removed.csv", index=False)

    stats = {
        "dataset": cfg["dataset"]["name"], "synthetic": bool(valid["synthetic"].any()),
        "prepared_at": now_iso(), "seed": cfg["seed"], "split_source": split_source,
        "raw_records": n_raw, "usable_records": len(valid),
        "removed_by_reason": removed["reason"].map(lambda s: s.split(":")[0]).value_counts().to_dict()
        if len(removed) else {},
        **dup_stats,
        "groups": int(valid["group_id"].nunique()),
        "sources": {src: {**valid[valid["source"] == src]["label"].value_counts().reindex(cfg["labels"]).fillna(0)
                          .astype(int).to_dict(), "split": "/".join(sorted(valid[valid["source"] == src]["split"].unique()))}
                    for src in sorted(valid["source"].unique())},
        "source_split_info": source_info,
        "multi_member_groups": int((valid.groupby("group_id").size() > 1).sum()),
        "groups_spanning_splits": leak_groups,
        "missing_text": int((~valid["has_text"]).sum()),
        "split_sizes": valid["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_dict(),
        "label_distribution": {s: valid[valid["split"] == s]["label"].value_counts().reindex(cfg["labels"]).fillna(0)
                               .astype(int).to_dict() for s in ("train", "val", "test")},
        "label_distribution_total": valid["label"].value_counts().reindex(cfg["labels"]).fillna(0).astype(int).to_dict(),
        "scripts": valid["scripts"].replace("", "None").value_counts().to_dict(),
        "code_mixed_fraction": float(valid["code_mixed"].mean()),
        "text_words": {"mean": float(valid["n_words"].mean()), "median": float(valid["n_words"].median()),
                       "max": int(valid["n_words"].max())},
        "token_length": token_length_stats(valid["text"].tolist(), cfg["text"]["model_name"]),
        "image_size": {"median_w": float(valid["image_w"].median()), "median_h": float(valid["image_h"].median()),
                       "min_side": int(valid[["image_w", "image_h"]].min().min())},
    }
    save_json(stats, proc / "stats.json")
    rep = resolve_path(cfg["paths"]["reports_dir"]) / "dataset"
    make_figures(valid, cfg["labels"], rep)
    save_json(stats, rep / "stats.json")
    print(f"Split sizes: {stats['split_sizes']}  | groups spanning splits: {leak_groups}")
    print(f"Wrote {proc} and figures to {rep}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", choices=["memosen", "csv", "sample"], default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    prepare(copy.deepcopy(cfg))


if __name__ == "__main__":
    main()
