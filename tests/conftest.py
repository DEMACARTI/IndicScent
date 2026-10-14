import copy
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # tests never download anything
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.utils import load_config  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config()
    c["device"] = "cpu"
    return copy.deepcopy(c)


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(load_config()["text"]["model_name"])
    except Exception as e:  # not cached locally
        pytest.skip(f"XLM-R tokenizer not available offline: {e}")


@pytest.fixture
def tiny_text_config():
    """XLM-R architecture with real vocab size but tiny dimensions (random weights)."""
    from transformers import XLMRobertaConfig
    return XLMRobertaConfig(vocab_size=250002, hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
                            intermediate_size=64, max_position_embeddings=130, pad_token_id=1)


def pretrained_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        import torch
        hub = Path(torch.hub.get_dir()) / "checkpoints"
        has_resnet = any(hub.glob("resnet50-*.pth"))
        has_xlmr = isinstance(try_to_load_from_cache(load_config()["text"]["model_name"], "model.safetensors"), str)
        return has_resnet and has_xlmr
    except Exception:
        return False
