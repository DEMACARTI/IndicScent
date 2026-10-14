"""Grad-CAM (Selvaraju et al., 2017) on the last ResNet-50 stage (layer4, 7x7 grid).

Works for both the image-only and the multimodal models (for the latter the text input is
held fixed and gradients flow through the fusion module into the image branch).

The heatmap shows which image regions the target-class score is most *sensitive* to at
7x7 resolution. It is a coarse, gradient-based sensitivity map, not a causal explanation,
and it can highlight rendered caption text as well as visual content.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


def _image_encoder(model):
    return model.image_encoder if hasattr(model, "image_encoder") else model.encoder


def grad_cam(model, inputs: dict, target: int) -> np.ndarray:
    """Return a [0, 1] heatmap of shape (7, 7) for ``target``."""
    model.eval()
    layer = _image_encoder(model).gradcam_layer
    store = {}

    def fwd_hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            pv = inputs["pixel_values"].clone().requires_grad_(True)
            logits = model(**{**inputs, "pixel_values": pv})
            model.zero_grad(set_to_none=True)
            logits[0, target].backward()
    finally:
        h.remove()
    act, grad = store["act"].detach()[0], store["grad"].detach()[0]       # (C, h, w)
    weights = grad.mean(dim=(1, 2))                                         # (C,)
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.float().cpu().numpy()


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> Image.Image:
    """Upsample the CAM to the original image size and blend it over the image."""
    w, h = image.size
    cam_t = torch.from_numpy(cam)[None, None]
    cam_up = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    heat = (colormaps[cmap](cam_up)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    blend = (1 - alpha) * base + alpha * heat.astype(np.float32)
    return Image.fromarray(blend.clip(0, 255).astype(np.uint8))
