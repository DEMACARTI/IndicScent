"""Optional REST API around the same predictor the Streamlit app uses.

    uvicorn src.api:app --port 8000
    curl -F "text=আজকের দিনটা দারুণ" -F "image=@meme.jpg" -F "model=multimodal_concat" localhost:8000/predict
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .data.image_utils import DEFAULT_MAX_BYTES
from .inference import InputError, ModelNotAvailable, SentimentPredictor

app = FastAPI(title="IndicSent-M", version="1.0",
              description="Multimodal sentiment analysis for Indian-language memes (positive/neutral/negative).")
_predictor: SentimentPredictor | None = None


def predictor() -> SentimentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor


@app.get("/health")
def health():
    return {"status": "ok", "models": predictor().available(), "device": str(predictor().device)}


@app.post("/predict")
async def predict(model: str = Form("multimodal_concat"), text: str | None = Form(None),
                  image: UploadFile | None = File(None), explain: bool = Form(False)):
    data = None
    if image is not None:
        data = await image.read(DEFAULT_MAX_BYTES + 1)
        if len(data) > DEFAULT_MAX_BYTES:
            raise HTTPException(413, "Image larger than 10 MB.")
    try:
        r = predictor().predict(text, data, model, explain=explain)
    except InputError as e:
        raise HTTPException(422, str(e))
    except ModelNotAvailable as e:
        raise HTTPException(503, str(e))
    return {**r.to_dict(), "note": "confidence = predicted-class probability, not a guarantee of correctness"}
