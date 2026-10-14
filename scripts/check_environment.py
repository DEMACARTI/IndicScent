"""Check Python/package versions, hardware, cached model weights and data availability.

    python scripts/check_environment.py
"""
from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = ["torch", "torchvision", "transformers", "sentencepiece", "numpy", "pandas", "PIL", "sklearn",
            "yaml", "matplotlib", "seaborn", "streamlit", "fastapi", "openpyxl", "pytest"]


def main() -> int:
    ok = True
    print(f"Python      {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  !! Python >= 3.10 required")
        ok = False
    print(f"Platform    {platform.platform()}")
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f"  ok  {mod:<14} {getattr(m, '__version__', '')}")
        except Exception as e:
            print(f"  !!  {mod:<14} missing ({e.__class__.__name__})")
            ok = False
    try:
        import torch
        print(f"CUDA        {torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
        print(f"MPS         {torch.backends.mps.is_available()}")
        from src.utils import get_device
        print(f"Device used {get_device('auto')}")
    except Exception as e:
        print(f"  !! torch device check failed: {e}")
        ok = False
    try:
        import psutil  # optional
        print(f"RAM         {psutil.virtual_memory().total / 2**30:.1f} GB")
    except ImportError:
        pass
    free = shutil.disk_usage(ROOT).free / 2**30
    print(f"Disk free   {free:.1f} GB" + ("  (!! < 5 GB: models + data need ~3 GB)" if free < 5 else ""))

    from src.utils import load_config
    cfg = load_config()
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(cfg["text"]["model_name"], "model.safetensors")
        print(f"XLM-R cache {'ok' if isinstance(hit, str) else 'not cached (downloads ~1.1 GB on first use)'}")
    except Exception:
        pass
    import torch
    resnet = list((Path(torch.hub.get_dir()) / "checkpoints").glob("resnet50-*.pth"))
    print(f"ResNet-50   {'ok' if resnet else 'not cached (downloads ~100 MB on first use)'}")
    raw = ROOT / "data" / "raw" / "memosen" / "Dataset" / "multi-sent.xlsx"
    print(f"MemoSen     {'ok' if raw.exists() else 'missing -> python -m src.data.download --dataset memosen'}")
    proc = ROOT / "data" / "processed" / "train.csv"
    print(f"Processed   {'ok' if proc.exists() else 'missing -> python -m src.data.prepare'}")
    for exp in ["text", "image", "multimodal_concat", "multimodal_attention"]:
        c = ROOT / "artifacts" / "checkpoints" / exp / "metadata.json"
        print(f"Checkpoint  {exp:<22} {'ok' if c.exists() else 'not trained'}")
    print("\nEnvironment OK" if ok else "\nEnvironment has problems (see !! above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
