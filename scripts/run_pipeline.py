"""Run the full IndicSent-M pipeline end to end (cross-platform).

    python scripts/run_pipeline.py                       # everything, MemoSen
    python scripts/run_pipeline.py --dataset sample --epochs 1   # SYNTHETIC smoke run
    python scripts/run_pipeline.py --skip download prepare        # resume

Steps: check -> download -> prepare -> train (text, image, multimodal concat, multimodal
attention) -> evaluate -> compare -> explain. Each step is a separate subprocess so a
failure stops the pipeline with a clear message.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["check", "download", "prepare", "train", "evaluate", "compare", "explain"]


def run(args: list[str]) -> None:
    print(f"\n$ python {' '.join(args)}", flush=True)
    t = time.perf_counter()
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"Step failed (exit {r.returncode}): python {' '.join(args)}")
    print(f"  ({time.perf_counter() - t:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["memosen", "csv", "sample"], default=None,
                    help="override dataset.name from the config")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    ap.add_argument("--only", nargs="*", default=None, choices=STEPS)
    a = ap.parse_args()
    steps = [s for s in (a.only or STEPS) if s not in a.skip]
    cfg = ["--config", a.config] if a.config else []
    ep = ["--epochs", str(a.epochs)] if a.epochs else []
    if a.dataset and a.dataset != "memosen" and "download" in steps:
        steps.remove("download")
    if a.dataset and a.dataset != "memosen" and not a.config:
        print("NOTE: --dataset only affects `prepare`; training reads the prepared split files.")

    if "check" in steps:
        run(["scripts/check_environment.py"])
    if "download" in steps:
        run(["-m", "src.data.download", "--dataset", "memosen"])
    if "prepare" in steps:
        run(["-m", "src.data.prepare", *cfg, *(["--dataset", a.dataset] if a.dataset else [])])
    if "train" in steps:
        run(["-m", "src.training.train_text", *cfg, *ep])
        run(["-m", "src.training.train_image", *cfg, *ep])
        run(["-m", "src.training.train_multimodal", "--fusion", "concat", *cfg, *ep])
        run(["-m", "src.training.train_multimodal", "--fusion", "attention", *cfg, *ep])
    if "evaluate" in steps:
        run(["-m", "src.evaluation.evaluate", *cfg])
    if "compare" in steps:
        run(["-m", "src.evaluation.compare_models", *cfg])
    if "explain" in steps:
        run(["-m", "src.explainability.generate_examples", *cfg])
    print("\nPipeline finished. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
