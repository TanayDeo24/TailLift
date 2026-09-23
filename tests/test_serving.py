"""
Serving-layer tests (Section 11 of the deployment brief). These test the
FastAPI wrapper around the frozen Level 5 system -- they do NOT retrain,
retune, or re-evaluate the model. Session-scoped fixtures load the model
once (it is slow) and reuse it across tests.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import app, service  # noqa: E402
from mapping import LexicalMapper, FusionMapper, SupportPriorMapper  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------- service initialization / health ----------------

def test_service_loaded_successfully():
    assert service.ready is True
    assert service.load_error is None


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["head_index_size"] == 12261
    assert body["frozen_config_version"]


def test_metadata_endpoint(client):
    r = client.get("/api/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["head_index_size"] == 12261
    assert body["primary_k"] == 5
    assert "evaluation_note" in body
    assert "OFFLINE" in body["evaluation_note"].upper()


def test_static_ui_route(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "SparseQuery" in r.text


# ---------------- /api/map validation ----------------

def test_map_empty_query_rejected(client):
    r = client.post("/api/map", json={"query": "", "k": 5})
    assert r.status_code == 422


def test_map_whitespace_only_query_rejected(client):
    r = client.post("/api/map", json={"query": "   ", "k": 5})
    assert r.status_code == 422


def test_map_k_out_of_bounds_rejected(client):
    assert client.post("/api/map", json={"query": "aol", "k": 0}).status_code == 422
    assert client.post("/api/map", json={"query": "aol", "k": 11}).status_code == 422


def test_map_k_within_bounds_accepted(client):
    r = client.post("/api/map", json={"query": "nfldraftcountdown", "k": 3})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3


def test_map_valid_query_returns_ranked_results(client):
    r = client.post("/api/map", json={"query": "nfldraftcountdown", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "nfldraftcountdown"
    assert len(body["results"]) == 5
    ranks = [item["rank"] for item in body["results"]]
    assert ranks == [1, 2, 3, 4, 5]
    scores = [item["final_score"] for item in body["results"]]
    assert scores == sorted(scores, reverse=True)  # descending order


def test_map_result_has_all_score_fields(client):
    r = client.post("/api/map", json={"query": "aol update", "k": 1})
    item = r.json()["results"][0]
    for field in ("rank", "head_query", "lexical_score", "semantic_score", "support_score", "final_score"):
        assert field in item


def test_map_latency_and_model_fields_present(client):
    r = client.post("/api/map", json={"query": "aol update", "k": 1})
    body = r.json()
    assert body["latency_ms"] > 0
    assert body["model"]["embedding_model"] == service.embed_model_name
    assert body["model"]["alpha"] == service.alpha
    assert body["model"]["beta"] == service.beta


def test_map_deterministic_for_fixed_input(client):
    r1 = client.post("/api/map", json={"query": "nfldraftcountdown", "k": 5}).json()
    r2 = client.post("/api/map", json={"query": "nfldraftcountdown", "k": 5}).json()
    heads1 = [r["head_query"] for r in r1["results"]]
    heads2 = [r["head_query"] for r in r2["results"]]
    assert heads1 == heads2
    scores1 = [r["final_score"] for r in r1["results"]]
    scores2 = [r["final_score"] for r in r2["results"]]
    assert scores1 == scores2


# ---------------- /api/compare ----------------

def test_compare_returns_three_systems_by_default(client):
    r = client.post("/api/compare", json={"query": "species of trees", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert set(["lexical_baseline", "semantic_only", "final_system"]).issubset(body.keys())
    assert "experimental_rejected_reranker" not in body


def test_compare_lexical_endpoint_shape(client):
    r = client.post("/api/compare", json={"query": "species of trees", "k": 3})
    block = r.json()["lexical_baseline"]
    assert block["system"] == "lexical_baseline"
    assert len(block["results"]) == 3
    assert block["latency_ms"] > 0


def test_compare_semantic_endpoint_shape(client):
    r = client.post("/api/compare", json={"query": "species of trees", "k": 3})
    block = r.json()["semantic_only"]
    assert block["system"] == "semantic_only"
    assert len(block["results"]) == 3


def test_compare_final_endpoint_shape(client):
    r = client.post("/api/compare", json={"query": "species of trees", "k": 3})
    block = r.json()["final_system"]
    assert block["system"] == "final_system_level5"
    assert len(block["results"]) == 3


def test_compare_experimental_reranker_opt_in_only(client):
    r = client.post("/api/compare", json={"query": "aol update", "k": 3, "include_experimental_reranker": True})
    body = r.json()
    assert "experimental_rejected_reranker" in body
    assert "rejected" in body["experimental_rejected_reranker"]["note"].lower()


# ---------------- API ranking must exactly match the frozen mapper ----------------

def test_api_matches_frozen_mapper():
    """The most important serving test: /api/map must reproduce EXACTLY the
    same ranking and scores as scripts/run_ladder.py's Level 5 pipeline
    (LexicalMapper + FusionMapper + SupportPriorMapper), independently
    reconstructed here from the frozen artifacts -- not by calling the
    service internals, so this genuinely catches drift."""
    frozen_config = json.loads((ROOT / "artifacts" / "frozen_config.json").read_text())
    head_index = json.loads((ROOT / "artifacts" / "splits" / "head_index.json").read_text())
    train_rows = json.loads((ROOT / "artifacts" / "splits" / "train.json").read_text())

    from collections import Counter
    train_pos_counts = Counter()
    for r in train_rows:
        for h in r["positive_heads"]:
            train_pos_counts[h] += 1

    lex_mapper = LexicalMapper(head_index)
    fusion = FusionMapper(frozen_config["alpha_fusion"])
    prior = SupportPriorMapper(head_index, train_pos_counts, frozen_config["beta_support_prior"])

    query = "nfldraftcountdown"
    norm_q = query.strip().lower()

    tail_emb = service.model.encode(
        [norm_q], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
    )[0]
    semantic_scores = service.head_embeddings @ tail_emb
    lexical_scores = lex_mapper.score_all(norm_q)
    fused = fusion.fuse(lexical_scores, semantic_scores)
    expected_final = prior.add_prior(fused)
    expected_order = np.argsort(-expected_final)[:5]
    expected_heads = [head_index[i] for i in expected_order]

    from fastapi.testclient import TestClient
    client = TestClient(app)
    api_body = client.post("/api/map", json={"query": query, "k": 5}).json()
    api_heads = [r["head_query"] for r in api_body["results"]]

    assert api_heads == expected_heads
    for i, r in enumerate(api_body["results"]):
        assert r["final_score"] == pytest.approx(float(expected_final[expected_order[i]]), rel=1e-6)


def test_service_uses_frozen_config_not_hardcoded_copies():
    frozen_config = json.loads((ROOT / "artifacts" / "frozen_config.json").read_text())
    assert service.alpha == frozen_config["alpha_fusion"]
    assert service.beta == frozen_config["beta_support_prior"]
    assert service.embed_model_name == frozen_config["embed_model"]
    assert service.primary_k == frozen_config["primary_k"]


def test_service_support_prior_built_from_train_split_only():
    """Guards against the serving layer accidentally leaking val/test labels
    into the support prior."""
    train_rows = json.loads((ROOT / "artifacts" / "splits" / "train.json").read_text())
    val_only_heads = set()
    val_rows = json.loads((ROOT / "artifacts" / "splits" / "val.json").read_text())
    train_heads = set()
    for r in train_rows:
        train_heads.update(r["positive_heads"])
    for r in val_rows:
        val_only_heads.update(r["positive_heads"])
    val_only_heads -= train_heads
    if not val_only_heads:
        pytest.skip("no val-only positive heads available to probe with")
    probe_head = next(iter(val_only_heads))
    idx = service.head_index.index(probe_head)
    assert service.prior.log_prior[idx] == 0.0
