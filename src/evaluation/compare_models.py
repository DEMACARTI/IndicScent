"""Aggregate per-experiment evaluation outputs into comparison tables and plots.

Everything here is computed from artifacts/reports/<experiment>/{metrics.json,predictions.csv};
no number is hard-coded. Uncertainty is quantified with a non-parametric bootstrap over
test examples (95 % CI) and a *paired* bootstrap for differences between models, because
on a test set of a few hundred memes a 1-2 point gap can easily be noise.

    python -m src.evaluation.compare_models
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..models.factory import EXPERIMENTS
from ..utils import load_config, load_json, resolve_path, save_json

DISPLAY = {"text": "Text-only (XLM-R)", "image": "Image-only (ResNet-50)",
           "multimodal_concat": "Multimodal: concat", "multimodal_attention": "Multimodal: cross-attention",
           "baseline_majority": "Baseline: majority class",
           "baseline_source_prior": "Baseline: source-page prior*"}
COLORS = {"text": "#4C72B0", "image": "#DD8452", "multimodal_concat": "#55A868", "multimodal_attention": "#8172B3",
          "baseline_majority": "#B0B0B0", "baseline_source_prior": "#7F7F7F"}


def reference_baselines(cfg: dict, ids: list[str]) -> dict[str, np.ndarray]:
    """Content-free reference predictions for the test ids.

    * majority: the most frequent training label for every meme.
    * source prior*: the most frequent training label of the meme's *source page* (falls back to
      the majority label for pages unseen in training). It uses metadata a deployed model never
      sees, so it is NOT a competitor: it measures how much of the task can be solved by
      recognising where a meme came from.
    """
    from ..data.dataset import load_split
    lab2id = {l: i for i, l in enumerate(cfg["labels"])}
    tr = load_split(cfg, "train")
    te = load_split(cfg, "test").set_index("id").loc[ids]
    maj = tr["label"].value_counts().idxmax()
    out = {"baseline_majority": np.full(len(te), lab2id[maj])}
    if "source" in tr.columns and (tr["source"] != "").any():
        src_major = tr.groupby("source")["label"].agg(lambda s: s.value_counts().idxmax())
        out["baseline_source_prior"] = te["source"].map(src_major).fillna(maj).map(lab2id).to_numpy()
    return out


def _macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    acc, f1 = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        acc.append((y[i] == p[i]).mean())
        f1.append(_macro_f1(y[i], p[i]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"accuracy_ci95": q(acc), "macro_f1_ci95": q(f1)}


def paired_bootstrap(y, p_a, p_b, n_boot: int = 1000, seed: int = 0) -> dict:
    """Distribution of macroF1(a) - macroF1(b) under resampling of test examples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(_macro_f1(y[i], p_a[i]) - _macro_f1(y[i], p_b[i]))
    d = np.asarray(d)
    return {"observed_diff": float(_macro_f1(y, p_a) - _macro_f1(y, p_b)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_a_not_better": float((d <= 0).mean())}  # one-sided bootstrap p-value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    labels = cfg["labels"]
    rep = resolve_path(cfg["paths"]["reports_dir"])
    out = rep / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    metrics, preds = {}, {}
    for exp in EXPERIMENTS:
        mp = rep / exp / "metrics.json"
        if mp.exists():
            metrics[exp] = load_json(mp)
            preds[exp] = pd.read_csv(rep / exp / "predictions.csv").set_index("id")
    if not metrics:
        print("No evaluation outputs found. Run `python -m src.evaluation.evaluate` first.")
        return
    lab2id = {l: i for i, l in enumerate(labels)}

    # all models must have been evaluated on identical test examples for a fair comparison
    id_sets = {e: set(p.index) for e, p in preds.items()}
    common = sorted(set.intersection(*id_sets.values()))
    if any(len(s) != len(common) for s in id_sets.values()):
        print("WARNING: experiments were evaluated on different example sets; using the intersection.")
    y = preds[next(iter(preds))].loc[common, "true"].map(lab2id).to_numpy()
    P = {e: p.loc[common, "pred"].map(lab2id).to_numpy() for e, p in preds.items()}

    rows = []
    for exp, m in metrics.items():
        ci = bootstrap_ci(y, P[exp], args.n_boot)
        m["bootstrap"] = ci
        row = {"experiment": exp, "model": DISPLAY[exp], "n_test": m["n"],
               "accuracy": m["accuracy"], "acc_ci95": ci["accuracy_ci95"],
               "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
               "macro_f1": m["macro_f1"], "macro_f1_ci95": ci["macro_f1_ci95"], "weighted_f1": m["weighted_f1"]}
        for l in labels:
            row[f"f1_{l}"] = m["per_class"][l]["f1-score"]
        row["trainable_params_M"] = m["parameters"]["trainable"] / 1e6
        row["training_min"] = m.get("training_seconds", float("nan")) / 60
        row["epochs_run"] = m.get("epochs_run")
        row["best_epoch"] = m["checkpoint_epoch"]
        row["latency_median_ms"] = m.get("latency", {}).get("median_ms")
        row["latency_device"] = m.get("latency", {}).get("device")
        rows.append(row)
    from .evaluate import compute_metrics
    for name, pb in reference_baselines(cfg, common).items():
        m = compute_metrics(y, pb, labels)
        ci = bootstrap_ci(y, pb, args.n_boot)
        P[name] = pb
        rows.append({"experiment": name, "model": DISPLAY[name], "n_test": m["n"], "accuracy": m["accuracy"],
                     "acc_ci95": ci["accuracy_ci95"], "macro_precision": m["macro_precision"],
                     "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"],
                     "macro_f1_ci95": ci["macro_f1_ci95"], "weighted_f1": m["weighted_f1"],
                     **{f"f1_{l}": m["per_class"][l]["f1-score"] for l in labels},
                     "trainable_params_M": 0.0, "training_min": float("nan"), "epochs_run": None,
                     "best_epoch": None, "latency_median_ms": None, "latency_device": None})
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison_table.csv", index=False)

    # significance of multimodal vs unimodal
    tests = {}
    for mm in [e for e in P if e.startswith("multimodal")]:
        for uni in [e for e in ("text", "image") if e in P]:
            tests[f"{mm}_vs_{uni}"] = paired_bootstrap(y, P[mm], P[uni], args.n_boot)
    if "baseline_source_prior" in P:
        for e in [e for e in metrics]:
            tests[f"{e}_vs_baseline_source_prior"] = paired_bootstrap(y, P[e], P["baseline_source_prior"], args.n_boot)
    if "multimodal_attention" in P and "multimodal_concat" in P:
        tests["multimodal_attention_vs_multimodal_concat"] = paired_bootstrap(
            y, P["multimodal_attention"], P["multimodal_concat"], args.n_boot)
    ablations = {e: metrics[e]["modality_ablation"] for e in metrics if "modality_ablation" in metrics[e]}
    save_json({"table": rows, "paired_bootstrap": tests, "modality_ablation": ablations,
               "n_common_test_examples": len(common)}, out / "comparison.json")

    _plot_bars(table, out / "accuracy_macro_f1.png")
    _plot_per_class(table, labels, out / "per_class_f1.png")
    _plot_confusions(metrics, labels, out / "confusion_matrices.png")
    _write_markdown(table, tests, ablations, labels, out / "comparison.md")
    print(table[["model", "accuracy", "macro_f1", "weighted_f1", "training_min", "latency_median_ms"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Comparison written to {out}")


def _plot_bars(t: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (col, ci, title) in zip(axes, [("accuracy", "acc_ci95", "Accuracy"),
                                           ("macro_f1", "macro_f1_ci95", "Macro-F1")]):
        vals = t[col].to_numpy()
        lo = vals - np.array([c[0] for c in t[ci]])
        hi = np.array([c[1] for c in t[ci]]) - vals
        bars = ax.bar(t["model"], vals, yerr=[lo, hi], capsize=4,
                      color=[COLORS[e] for e in t["experiment"]])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, 0.03, f"{v:.3f}", ha="center", color="white",
                    fontweight="bold")
        ax.set_title(f"Test {title} (95% bootstrap CI)")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_per_class(t: pd.DataFrame, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(t)
    x = np.arange(len(labels))
    for i, (_, r) in enumerate(t.iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, [r[f"f1_{l}"] for l in labels], w, label=r["model"],
               color=COLORS[r["experiment"]])
    ax.set_xticks(x, [l.capitalize() for l in labels])
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 on the test set")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusions(metrics: dict, labels: list[str], path: Path) -> None:
    import seaborn as sns
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes)
    for ax, (exp, m) in zip(axes, metrics.items()):
        cm = np.asarray(m["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True).clip(min=1)
        sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(DISPLAY[exp], fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_markdown(t: pd.DataFrame, tests: dict, ablations: dict, labels: list[str], path: Path) -> None:
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
    ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
    lines = ["# Model comparison (auto-generated from evaluation outputs)", "",
             f"Test examples: {int(t['n_test'].iloc[0])}", "",
             "| Model | Accuracy (95% CI) | Macro-P | Macro-R | Macro-F1 (95% CI) | Weighted-F1 | "
             + " | ".join(f"F1 {l}" for l in labels) + " | Train time (min) | Latency median (ms) |",
             "|" + "---|" * (8 + len(labels))]
    for _, r in t.iterrows():
        lines.append(f"| {r['model']} | {f(r['accuracy'])} {ci(r['acc_ci95'])} | {f(r['macro_precision'])} | "
                     f"{f(r['macro_recall'])} | {f(r['macro_f1'])} {ci(r['macro_f1_ci95'])} | {f(r['weighted_f1'])} | "
                     + " | ".join(f(r[f'f1_{l}']) for l in labels)
                     + (f" | {r['training_min']:.1f} | {f(r['latency_median_ms'])} ({r['latency_device']}) |"
                        if r["experiment"] in EXPERIMENTS else " | – | – |"))
    if "baseline_source_prior" in set(t["experiment"]):
        lines += ["", "\\* The source-page prior uses the meme's source page (metadata, not content) and is "
                  "a shortcut reference, not a competing model."]
    if tests:
        lines += ["", "## Paired bootstrap: macro-F1 difference (A − B)", "",
                  "| Comparison | Observed Δ | 95% CI | P(A not better) |", "|---|---|---|---|"]
        for k, v in tests.items():
            lines.append(f"| {k.replace('_vs_', ' vs ')} | {v['observed_diff']:+.4f} | {ci(v['ci95'])} | "
                         f"{v['p_a_not_better']:.3f} |")
    if ablations:
        lines += ["", "## Modality ablation of the multimodal models", "",
                  "| Model | Condition | Accuracy | Macro-F1 | Predictions changed |", "|---|---|---|---|---|"]
        for e, ab in ablations.items():
            for cond, v in ab.items():
                lines.append(f"| {DISPLAY[e]} | {cond} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | "
                             f"{v['prediction_changed_rate']:.1%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
