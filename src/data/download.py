"""Obtain raw data.

    python -m src.data.download --dataset memosen     # public Google Drive link from the authors' repo
    python -m src.data.download --dataset memotion3   # prints registration instructions (gated)
    python -m src.data.download --dataset sample      # regenerates the SYNTHETIC sample

MemoSen: Hossain, Sharif & Hoque, "MemoSen: A Multimodal Dataset for Sentiment Analysis of
Memes", LREC 2022. Link taken from https://github.com/eftekhar-hossain/Multimodal-Sentiment-LREC2022
The dataset carries no explicit licence in that repository: use it for research only and cite
the paper. Do not redistribute the files.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

from ..utils import resolve_path, save_json, now_iso

MEMOSEN_DRIVE_ID = "12jxFuMz7jtE1kN9fi1ckspgHcKZhkT_B"
MEMOSEN_SOURCE = "https://github.com/eftekhar-hossain/Multimodal-Sentiment-LREC2022"

MEMOTION3_INSTRUCTIONS = """\
Memotion 3 (Hindi-English code-mixed memes; Mishra et al., 2023) is distributed by the
DeFactify-2 @ AAAI-2023 organisers after registration:
  1. Register: https://aiisc.ai/defactify2/Memotion.html  (form linked on that page)
  2. Download the data you are granted access to into data/raw/custom/
  3. Point configs/config.yaml -> dataset.name: csv and set dataset.csv.* to the column
     names / label strings of the released annotation file (inspect it first; this project
     has NOT seen those files, so no column names are assumed).
  4. python -m src.data.prepare
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zip_path: Path, dest: Path) -> int:
    """Extract a zip refusing absolute paths / '..' components (zip-slip)."""
    dest = dest.resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest)):
                raise RuntimeError(f"Unsafe path in archive: {info.filename}")
            if info.filename.startswith("__MACOSX/") or Path(info.filename).name.startswith("._"):
                continue
            zf.extract(info, dest)
            n += 1
    return n


def download_memosen(force: bool = False) -> Path:
    raw = resolve_path("data/raw")
    dl_dir = raw / "memosen_download"
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / "memosen.zip"
    if not zip_path.exists() or force:
        import gdown
        print("Downloading MemoSen from Google Drive (this can be slow) ...")
        gdown.download(id=MEMOSEN_DRIVE_ID, output=str(zip_path), quiet=False)
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"{zip_path} is not a valid zip archive (download incomplete or blocked). "
                           "Delete it and retry, or download manually from " + MEMOSEN_SOURCE)
    out = raw / "memosen"
    if out.exists() and force:
        shutil.rmtree(out)
    if not out.exists():
        n = safe_extract(zip_path, out)
        print(f"Extracted {n} entries to {out}")
    save_json({"dataset": "MemoSen", "source": MEMOSEN_SOURCE, "google_drive_id": MEMOSEN_DRIVE_ID,
               "archive_sha256": sha256(zip_path), "archive_bytes": zip_path.stat().st_size,
               "downloaded_at": now_iso()}, raw / "memosen_provenance.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["memosen", "memotion3", "sample"], default="memosen")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.dataset == "memosen":
        download_memosen(args.force)
    elif args.dataset == "memotion3":
        print(MEMOTION3_INSTRUCTIONS)
    else:
        from .make_sample import main as make_sample
        make_sample()


if __name__ == "__main__":
    main()
