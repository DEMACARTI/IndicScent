# IndicSent-M — CPU image for the Streamlit app (and FastAPI via an alternate command).
# Build:  docker build -t indicsent-m .
# Run:    docker run -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts" -v "$(pwd)/data:/app/data" indicsent-m
# Model weights and data are NOT baked into the image: mount artifacts/ (checkpoints + reports) and data/.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface TORCH_HOME=/app/.cache/torch TOKENIZERS_PARALLELISM=false
WORKDIR /app

# CPU-only PyTorch wheels keep the image ~1.5 GB smaller than the default CUDA build
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1 \
 && pip install -r requirements.txt

COPY . .
# Pre-fetch the pretrained backbones that frozen checkpoint weights are rebuilt from
RUN python -c "from transformers import AutoModel, AutoTokenizer as T; m='FacebookAI/xlm-roberta-base'; AutoModel.from_pretrained(m); T.from_pretrained(m); import torchvision as tv; tv.models.resnet50(weights='IMAGENET1K_V2')"

RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8501 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
