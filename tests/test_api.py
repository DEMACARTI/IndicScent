"""REST API error handling (no trained model needed)."""
import io

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import src.api as api  # noqa: E402
from src.inference import SentimentPredictor  # noqa: E402


@pytest.fixture
def client(cfg, tmp_path, monkeypatch):
    cfg["paths"]["checkpoints_dir"] = str(tmp_path)
    monkeypatch.setattr(api, "_predictor", SentimentPredictor(cfg=cfg, device="cpu"))
    return TestClient(api.app)


def _png():
    b = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(b, format="PNG")
    return b.getvalue()


def test_health_lists_no_models(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["models"] == {}


def test_missing_text_is_422(client):
    r = client.post("/predict", data={"model": "text", "text": ""})
    assert r.status_code == 422


def test_corrupt_image_is_422(client):
    r = client.post("/predict", data={"model": "image"}, files={"image": ("x.png", b"nope", "image/png")})
    assert r.status_code == 422


def test_untrained_model_is_503_not_a_prediction(client):
    r = client.post("/predict", data={"model": "image"}, files={"image": ("x.png", _png(), "image/png")})
    assert r.status_code == 503
    assert "train" in r.json()["detail"]
