"""Inference API used by the Streamlit app, FastAPI service and evaluation of single inputs.

Only real trained checkpoints are used. If a checkpoint is missing, ``ModelNotAvailable``
is raised with setup instructions; the predictor never falls back to random weights.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from PIL import Image

from .data.image_utils import InvalidImageError, load_image
from .data.text_processing import detect_scripts, normalize_text
from .models.factory import EXPERIMENTS, checkpoint_dir, load_trained
from .models.image_model import build_image_transforms
from .utils import get_device, load_config, load_json, sync_device

MODE_TO_EXPERIMENTS = {
    "text": ["text"],
    "image": ["image"],
    "multimodal": ["multimodal_concat", "multimodal_attention"],
}

SETUP_INSTRUCTIONS = (
    "No trained checkpoint found for this model. Prepare the data and train it first:\n"
    "  python -m src.data.download --dataset memosen\n"
    "  python -m src.data.prepare\n"
    "  python -m src.training.train_text | train_image | train_multimodal --fusion concat\n"
    "or run everything with: python scripts/run_pipeline.py"
)


class ModelNotAvailable(RuntimeError):
    pass


class InputError(ValueError):
    pass


@dataclass
class Prediction:
    experiment: str
    label: str
    confidence: float
    probabilities: dict
    latency_ms: float
    device: str
    normalized_text: str | None = None
    script_info: dict | None = None
    warnings: list = field(default_factory=list)
    text_attribution: dict | None = None   # Integrated Gradients
    text_occlusion: dict | None = None     # leave-one-word-out
    gradcam: object | None = None          # PIL.Image overlay
    gradcam_raw: object | None = None      # np.ndarray 7x7
    checkpoint: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("gradcam")
        d["gradcam_raw"] = None if self.gradcam_raw is None else self.gradcam_raw.tolist()
        return d


class SentimentPredictor:
    def __init__(self, config_path: str | None = None, device: str | None = None, max_loaded: int = 2,
                 cfg: dict | None = None):
        self.cfg = cfg if cfg is not None else load_config(config_path)
        self.device = get_device(device or self.cfg["device"])
        self.max_loaded = max_loaded
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    # ------------------------------------------------------------------ discovery
    def available(self) -> dict[str, dict]:
        """experiment -> short metadata for checkpoints that exist on disk."""
        out = {}
        for exp in EXPERIMENTS:
            d = checkpoint_dir(self.cfg, exp)
            if (d / "metadata.json").exists():
                m = load_json(d / "metadata.json")
                out[exp] = {"epoch": m["epoch"], "val_macro_f1": m["val_macro_f1"], "dataset": m["dataset"],
                            "saved_at": m.get("saved_at"), "fusion": m.get("fusion")}
        return out

    def _get(self, experiment: str):
        if experiment in self._cache:
            self._cache.move_to_end(experiment)
            return self._cache[experiment]
        d = checkpoint_dir(self.cfg, experiment)
        if not (d / "metadata.json").exists():
            raise ModelNotAvailable(f"[{experiment}] {SETUP_INSTRUCTIONS}")
        try:
            loaded = load_trained(d, self.device)
        except Exception as e:  # corrupt / incompatible checkpoint
            raise ModelNotAvailable(f"[{experiment}] checkpoint could not be loaded: {e}") from e
        self._cache[experiment] = loaded
        while len(self._cache) > self.max_loaded:
            self._cache.popitem(last=False)
            if self.device.type == "mps":
                torch.mps.empty_cache()
        return loaded

    # ------------------------------------------------------------------ predict
    def predict(self, text: str | None = None, image=None, experiment: str = "multimodal_concat",
                explain: bool = False, ig_steps: int | None = None) -> Prediction:
        if experiment not in EXPERIMENTS:
            raise InputError(f"Unknown model '{experiment}'.")
        modality = EXPERIMENTS[experiment][0]
        warnings: list[str] = []

        clean, script = None, None
        if modality in {"text", "multimodal"}:
            clean = normalize_text(text)
            if not clean:
                raise InputError("Please enter some text — this mode needs the meme's text/caption.")
            script = detect_scripts(clean)
            if not script["supported"]:
                warnings.append(f"Text contains a script the model was not designed for ({', '.join(script['scripts'])}); "
                                "the prediction may be unreliable.")
            if script["dominant"] == "None":
                warnings.append("Text has no letters (only emojis/digits/punctuation); prediction relies on those.")
        pil = None
        if modality in {"image", "multimodal"}:
            if image is None:
                raise InputError("Please upload an image — this mode needs the meme image.")
            if isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                try:
                    pil = load_image(image)
                except InvalidImageError as e:
                    raise InputError(str(e)) from e

        model, meta, tok = self._get(experiment)
        mcfg = meta["config"]
        labels = meta["labels"]
        if meta["dataset"] == "sample":
            warnings.append("This checkpoint was trained on the SYNTHETIC sample data — outputs are not meaningful.")
        inputs = {}
        if clean is not None:
            enc = tok(clean, truncation=True, max_length=mcfg["text"]["max_length"], return_tensors="pt")
            if len(tok(clean)["input_ids"]) > mcfg["text"]["max_length"]:
                warnings.append(f"Text was truncated to {mcfg['text']['max_length']} tokens.")
            inputs["input_ids"] = enc["input_ids"].to(self.device)
            inputs["attention_mask"] = enc["attention_mask"].to(self.device)
        if pil is not None:
            tfm = build_image_transforms(mcfg["image"]["image_size"], train=False)
            inputs["pixel_values"] = tfm(pil)[None].to(self.device)

        sync_device(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).float(), -1)[0].cpu()
        sync_device(self.device)
        latency = (time.perf_counter() - t0) * 1000
        idx = int(probs.argmax())

        pred = Prediction(
            experiment=experiment, label=labels[idx], confidence=float(probs[idx]),
            probabilities={l: float(probs[i]) for i, l in enumerate(labels)}, latency_ms=latency,
            device=str(self.device), normalized_text=clean, script_info=script, warnings=warnings,
            checkpoint={"epoch": meta["epoch"], "val_macro_f1": meta["val_macro_f1"], "dataset": meta["dataset"]},
        )
        if explain:
            from .explainability.image_explainer import grad_cam, overlay
            from .explainability.text_explainer import integrated_gradients, occlusion
            steps = ig_steps or self.cfg["explainability"]["ig_steps"]
            if clean is not None:
                args = (model, tok, clean, idx, self.device, mcfg["text"]["max_length"])
                pv = inputs.get("pixel_values")
                pred.text_occlusion = occlusion(*args, pixel_values=pv)
                pred.text_attribution = integrated_gradients(*args, pixel_values=pv, steps=steps)
            if pil is not None:
                cam = grad_cam(model, inputs, idx)
                pred.gradcam_raw = cam
                pred.gradcam = overlay(pil, cam)
        return pred


def best_multimodal(available: dict) -> str | None:
    mm = {k: v for k, v in available.items() if k.startswith("multimodal")}
    return max(mm, key=lambda k: mm[k]["val_macro_f1"]) if mm else None
