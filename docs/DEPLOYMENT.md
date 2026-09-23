# Deployment

This document covers running the SparseQuery serving layer locally, in
Docker, and on Render. It is downstream of the frozen model (see
`docs/FINAL_REPORT.md`) — nothing here changes any relevance metric, split,
or hyperparameter.

## 1. Local development

```bash
pip install -r requirements.txt -r requirements-serving.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/` for the demo UI, or `http://localhost:8000/health`
to check readiness. The first request after a cold start triggers model
loading if it hasn't already happened at import time (it has — `app/main.py`
constructs `MappingService()` at module import, before the server starts
accepting requests).

## 2. Docker run

```bash
docker build -t sparsequery .
docker run -p 8000:8000 -e PORT=8000 sparsequery
```

Then visit `http://localhost:8000/`.

**Verification status**: Docker is not installed in the environment this
project was built in, so `docker build`/`docker run` could not be executed
and verified end-to-end here. What *was* verified locally, outside a
container: the exact service construction path the Docker `HEALTHCHECK` and
build-time cache step use
(`python -c "from app.service import MappingService; ..."`, which is the
same call used in the Dockerfile's `RUN` step) succeeds, loads the frozen
model, and serves correct results (see `tests/test_serving.py`, all passing).
The Dockerfile's file paths were checked to exist. If you build the image
and hit an issue, it is most likely a Python-version or system-library
mismatch between this development machine (macOS, Apple Silicon, Python
3.12, conda) and the `python:3.11-slim` (Debian, x86_64/arm64 per your
build host) image — see Troubleshooting below.

## 3. Render setup

Render can build directly from `render.yaml` (Docker runtime) in this
repository:

1. Push this repository to a Git remote Render can access (GitHub/GitLab).
2. In the Render dashboard: **New > Blueprint**, point it at the repo. Render
   reads `render.yaml` and creates the `sparsequery` web service automatically
   (Docker runtime, health check at `/health`).
   - Alternative, without a Blueprint: **New > Web Service**, connect the
     repo, set **Runtime: Docker**, leave the Dockerfile path as `./Dockerfile`,
     and set the health check path to `/health`.
3. No environment variables or secrets are required for ordinary inference.
4. Deploy. Render injects `$PORT`; the container's `CMD` already binds to it.

**This repository has not been deployed to Render** — no Render account/API
credentials are available in the environment this project was built in. See
`docs/FINAL_REPORT.md` "Deployment" for the exact manual steps you need to
perform, and the freeze verdict this implies (**READY FOR DEPLOYMENT**, not
**DEPLOYED**).

## 4. Expected startup behavior

1. Container starts, `uvicorn` imports `app.main`, which constructs
   `MappingService()`.
2. The service loads `artifacts/frozen_config.json`, `artifacts/final_metrics.json`,
   `artifacts/splits/head_index.json`, `artifacts/splits/train.json`.
3. It loads the frozen bi-encoder (`multi-qa-MiniLM-L6-cos-v1`) — from the
   local HF cache populated at Docker build time (`local_files_only=True`
   first; falls back to a network download only if the cache is missing,
   e.g. in local dev without a prior build).
4. It loads/computes the 12,261 head-query embeddings — from a cached
   `.npy` file under `artifacts/` if present (built once, reused across
   restarts), computed fresh (~2-3s) otherwise.
5. It builds the lexical structures and the **train-split-only** support
   prior (`artifacts/splits/train.json` — never val/test).
6. `/health` reports `status: ok`, `model_loaded: true` once this completes.
   Measured locally (model already cached): **~3 seconds**. With a cold HF
   cache (first-ever run, no Docker build-time pre-fetch): ~15-20 seconds
   (network-dependent).

The rejected cross-encoder (Level 4) is **not** loaded at startup — it is
loaded lazily, once, only if a caller explicitly sets
`include_experimental_reranker: true` on `/api/compare`. Ordinary requests
never pay its memory or load-time cost.

## 5. Model download/cache behavior

- The Dockerfile pre-downloads and caches the bi-encoder at **build time**
  (`RUN python -c "...SentenceTransformer(...)"`), so a fresh container
  start does not depend on network access to Hugging Face.
- The Dockerfile also pre-computes and caches the head-query embeddings at
  **build time** (`RUN python -c "...MappingService()..."`), so the image
  ships ready to serve immediately.
- `HF_HOME` is set to `/app/.cache/huggingface` inside the image so the
  build-time download and the runtime load use the same cache location.

## 6. Memory considerations

The dominant resident memory cost is the bi-encoder model
(`multi-qa-MiniLM-L6-cos-v1`, ~90MB on disk) plus the 12,261×384 float32
head-embedding matrix (~18MB) plus PyTorch/transformers runtime overhead.
**Each `uvicorn` worker duplicates all of this** — this is why the Dockerfile
defaults to a single worker (`--workers 1`). Do not increase worker count
without first measuring per-worker RSS against your target Render instance's
memory limit; a 512MB instance is likely too small once PyTorch's own
overhead is included, a 1GB+ instance is the safer starting point.

## 7. Cold-start considerations

- **Docker image build**: pre-fetches the model and pre-computes head
  embeddings, so container start is fast (~3s to healthy, per §4).
- **Render's own cold start** (spin-down on free/low tiers, or a fresh
  deploy): the container itself still needs to start (image pull + `uvicorn`
  boot + the ~3s service load) — expect single-digit seconds to first
  healthy response after a deploy, longer after a platform-level cold start
  from an idle/sleeping instance (outside this project's control).

## 8. Health-check path

`GET /health` — returns `{"status": "ok", "model_loaded": true, "head_index_size": 12261, "frozen_config_version": "<hash>"}`
when ready, or `status: "degraded"` with an `error` field if model loading
failed. `render.yaml` and the Dockerfile's `HEALTHCHECK` both point at this
path.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` returns `degraded` | `artifacts/frozen_config.json` or `artifacts/splits/*.json` missing from the image | Confirm the Dockerfile's `COPY artifacts/...` lines match what's actually in the build context; check `.dockerignore` isn't excluding something needed |
| Container build hangs on the model-download `RUN` step | No network access during build (some CI/registries) | Pre-populate the HF cache another way (e.g. mount a volume with the cache, or vendor the model files) and set `HF_HUB_OFFLINE=1` |
| Cross-encoder scores come back as `null`/non-finite if you enable the experimental comparison mode | The known NaN bug this project found and fixed (`src/mapping.py::materialize_model_`, see `docs/CHANGE_REPORT.md`) — if it recurs on a different platform, verify `materialize_model_` is actually being called (`app/service.py::_rerank_experimental`) | Run `tests/test_mapping_bugfix.py` — it fails loudly if this regresses |
| High P99 latency on first few requests after deploy | JIT/lazy CPU kernel warmup, not a cold-start-per-request issue | Expected; the benchmark protocol (`scripts/latency_benchmark.py`, `tests/test_api_latency.py`) already excludes cold start by using warmup requests before measuring |
| 503 from every endpoint except `/health` | Model failed to load (`service.ready == False`) | Check `/health`'s `error` field, check container logs for the exception raised in `MappingService._load` |
