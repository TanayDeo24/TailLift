"""
FastAPI serving layer for the frozen SparseQuery Level 5 system.

Single deployable application: this backend also serves the static demo UI
(app/static/) so there is one process and one deployment target -- no
separate frontend build/deploy step. See docs/DEPLOYMENT.md.
"""
import hashlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import CompareRequest, HealthResponse, MapRequest
from app.service import MappingService

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger("sparsequery")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SparseQuery: Long-Tail Query Mapping", version="1.0.0")

# Loaded once at process startup (Phase 3): model, head embeddings, lexical
# structures, and the train-only support prior all stay resident for the
# life of the process. No per-request reloading, no network calls at
# inference time.
service = MappingService()


def _frozen_config_version() -> str:
    cfg_path = ROOT / "artifacts" / "frozen_config.json"
    if not cfg_path.exists():
        return "unknown"
    return hashlib.sha1(cfg_path.read_bytes()).hexdigest()[:10]


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if service.ready else "degraded",
        model_loaded=service.ready,
        head_index_size=len(service.head_index) if service.ready else None,
        frozen_config_version=_frozen_config_version(),
        error=service.load_error if not service.ready else None,
    )


def _require_ready():
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail=f"Model service is not ready: {service.load_error or 'still loading'}",
        )


@app.post("/api/map")
def map_query(req: MapRequest):
    _require_ready()
    try:
        return service.map_query(req.query, req.k)
    except Exception as exc:
        logger.exception("map_query failed")
        raise HTTPException(status_code=500, detail="Internal error while mapping the query.") from exc


@app.post("/api/compare")
def compare_query(req: CompareRequest):
    _require_ready()
    try:
        return service.compare_query(req.query, req.k, req.include_experimental_reranker)
    except Exception as exc:
        logger.exception("compare_query failed")
        raise HTTPException(status_code=500, detail="Internal error while comparing mappers.") from exc


@app.get("/api/metadata")
def metadata():
    _require_ready()
    return service.metadata()


# ---------------- static demo UI ----------------
# Mounted last so it does not shadow the /api and /health routes above.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
