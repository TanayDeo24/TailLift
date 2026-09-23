# SparseQuery serving image: FastAPI backend + static demo UI, single process.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System deps for torch/sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-serving.txt

# Copy only what the running service needs -- see .dockerignore for the
# original repo's large data/model files (text8, *.txt logs, cosModel.model,
# w2vNet.model, my_model.h5) which are NOT needed at serving time.
COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/frozen_config.json artifacts/final_metrics.json ./artifacts/
COPY artifacts/splits/ ./artifacts/splits/
COPY artifacts/latency_benchmark.json ./artifacts/latency_benchmark.json

# Pre-download and cache the frozen embedding model at BUILD time, not at
# first request -- avoids a fragile network-dependent cold start in
# production (Phase 8). The rejected cross-encoder is intentionally NOT
# pre-cached here: it is not on the default serving path and is only loaded
# lazily if a caller explicitly opts into the experimental comparison mode.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    import json; \
    cfg = json.load(open('artifacts/frozen_config.json')); \
    SentenceTransformer(cfg['embed_model'], device='cpu')"

# Pre-compute and cache the 12,261 head-query embeddings at build time too,
# so the very first request after a cold container start does not pay for
# it (Phase 8 caching requirement).
RUN python -c "from app.service import MappingService; s = MappingService(); assert s.ready, s.load_error"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:'+__import__('os').environ.get('PORT','8000')+'/health', timeout=3)" || exit 1

# One worker: this is a CPU-bound ML service and each worker duplicates the
# full model + 12,261-vector index in memory (see docs/DEPLOYMENT.md for the
# memory measurement backing this default). Increase only after measuring
# per-worker memory against the target instance size.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
