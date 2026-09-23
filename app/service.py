"""
Serving wrapper around the FROZEN Level 5 query-mapping system.

This module does not reimplement ranking logic. It loads the exact classes
from src/mapping.py and the exact frozen hyperparameters from
artifacts/frozen_config.json, and reuses the training-split-only support
counts that were used for the frozen held-out evaluation
(artifacts/splits/train.json). No relevance metric, split membership, or
hyperparameter is changed here -- see docs/FINAL_REPORT.md "Frozen
Evaluation Confirmation".

Startup responsibilities (Phase 3):
  1. load frozen config
  2. load the selected semantic bi-encoder (multi-qa-MiniLM-L6-cos-v1)
  3. load/precompute the 12,261 head-query embeddings (cached to disk)
  4. initialize lexical structures
  5. load the TRAIN-ONLY support prior (never touches val/test labels)
  6. keep everything resident in memory for inference

The rejected cross-encoder (Level 4) is intentionally NOT loaded into the
primary serving path -- see docs/FINAL_REPORT.md and
docs/BENCHMARK_REPORT.md Section 11 for why it was rejected on validation.
"""
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapping import LexicalMapper, FusionMapper, SupportPriorMapper, materialize_model_, tokenize  # noqa: E402


def _slugify_model_name(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()[:12]


class MappingService:
    """Holds the frozen Level 5 system resident in memory for inference.
    One instance is created at process startup and reused for every request
    -- no network calls, no recomputation, no per-request model loading."""

    def __init__(self):
        self.ready = False
        self.load_error: Optional[str] = None
        self._load()

    def _load(self):
        t0 = time.time()
        try:
            from sentence_transformers import SentenceTransformer

            self.frozen_config = json.loads((ROOT / "artifacts" / "frozen_config.json").read_text())
            self.final_metrics = json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())
            self.head_index = json.loads((ROOT / "artifacts" / "splits" / "head_index.json").read_text())
            train_rows = json.loads((ROOT / "artifacts" / "splits" / "train.json").read_text())

            # TRAIN-ONLY support prior counts (no val/test label ever touches this)
            train_pos_counts = Counter()
            for r in train_rows:
                for h in r["positive_heads"]:
                    train_pos_counts[h] += 1

            self.embed_model_name = self.frozen_config["embed_model"]
            self.alpha = self.frozen_config["alpha_fusion"]
            self.beta = self.frozen_config["beta_support_prior"]
            self.primary_k = self.frozen_config["primary_k"]

            # Prefer the local cache (populated at image build time, see
            # Dockerfile) so ordinary requests never trigger a network call.
            # Falls back to a normal (network) load on a fresh environment
            # where the model hasn't been cached/downloaded yet.
            try:
                self.model = SentenceTransformer(
                    self.embed_model_name, device="cpu", local_files_only=True,
                )
            except Exception:
                self.model = SentenceTransformer(self.embed_model_name, device="cpu")
            self.lex_mapper = LexicalMapper(self.head_index)
            self.fusion = FusionMapper(self.alpha)
            self.prior = SupportPriorMapper(self.head_index, train_pos_counts, self.beta)

            self.head_embeddings = self._load_or_compute_head_embeddings()

            self.ready = True
            self.load_time_s = round(time.time() - t0, 3)
        except Exception as exc:  # keep the service constructible even if loading fails
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.ready = False
            self.load_time_s = round(time.time() - t0, 3)

    def _cache_path(self) -> Path:
        slug = _slugify_model_name(self.embed_model_name)
        return ROOT / "artifacts" / f"head_embeddings_{slug}.npy"

    def _load_or_compute_head_embeddings(self) -> np.ndarray:
        """Phase 8: avoid recomputation on every process start. The 12,261
        head-query embeddings are a pure function of the frozen embedding
        model and the frozen head index, so they are safe to cache to disk
        and reload -- this changes nothing about the model's outputs."""
        cache_path = self._cache_path()
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape[0] == len(self.head_index):
                return cached
        embs = self.model.encode(
            self.head_index, batch_size=256, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embs)
        except OSError:
            pass  # read-only filesystem at runtime (e.g. some containers) -- non-fatal
        return embs

    # ---------------- inference ----------------

    def _encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]

    def _normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def map_query(self, query: str, k: int) -> dict:
        """The single, canonical frozen Level 5 pipeline: lexical + semantic
        fusion + train-only support prior, ranked over the full head index.
        This must match scripts/run_ladder.py::eval_prior exactly -- see
        tests/test_serving.py::test_api_matches_frozen_mapper."""
        t0 = time.time()
        norm_q = self._normalize(query)
        tail_emb = self._encode_query(norm_q)
        semantic_scores = self.head_embeddings @ tail_emb
        lexical_scores = self.lex_mapper.score_all(norm_q)
        fused = self.fusion.fuse(lexical_scores, semantic_scores)
        final_scores = self.prior.add_prior(fused)

        order = np.argsort(-final_scores)[:k]
        results = []
        for rank, idx in enumerate(order, start=1):
            results.append({
                "rank": rank,
                "head_query": self.head_index[idx],
                "lexical_score": float(lexical_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "support_score": float(self.prior.log_prior[idx]),
                "final_score": float(final_scores[idx]),
            })
        latency_ms = (time.time() - t0) * 1000.0
        return {
            "query": query,
            "normalized_query": norm_q,
            "results": results,
            "latency_ms": latency_ms,
            "model": {
                "embedding_model": self.embed_model_name,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        }

    def compare_query(self, query: str, k: int, include_experimental_reranker: bool = False) -> dict:
        """Default comparison: lexical baseline vs. semantic-only vs. the
        frozen final system (Level 5). The rejected cross-encoder (Level 4)
        is not included by default -- it was rejected from the frozen
        architecture on validation evidence (docs/BENCHMARK_REPORT.md
        Section 11) and is only ever exposed behind an explicit opt-in flag,
        never as part of the default demo comparison."""
        norm_q = self._normalize(query)

        t0 = time.time()
        lexical_scores = self.lex_mapper.score_all(norm_q)
        lexical_latency_ms = (time.time() - t0) * 1000.0
        lexical_order = np.argsort(-lexical_scores)[:k]

        t0 = time.time()
        tail_emb = self._encode_query(norm_q)
        semantic_scores = self.head_embeddings @ tail_emb
        semantic_latency_ms = (time.time() - t0) * 1000.0
        semantic_order = np.argsort(-semantic_scores)[:k]

        t0 = time.time()
        fused = self.fusion.fuse(lexical_scores, semantic_scores)
        final_scores = self.prior.add_prior(fused)
        final_latency_ms = (time.time() - t0) * 1000.0 + semantic_latency_ms + lexical_latency_ms
        final_order = np.argsort(-final_scores)[:k]

        def build(order, score_arr, label):
            return {
                "system": label,
                "latency_ms": None,  # filled by caller
                "results": [
                    {"rank": r, "head_query": self.head_index[i], "score": float(score_arr[i])}
                    for r, i in enumerate(order, start=1)
                ],
            }

        out = {
            "query": query,
            "normalized_query": norm_q,
            "lexical_baseline": {**build(lexical_order, lexical_scores, "lexical_baseline"),
                                  "latency_ms": lexical_latency_ms},
            "semantic_only": {**build(semantic_order, semantic_scores, "semantic_only"),
                               "latency_ms": lexical_latency_ms + semantic_latency_ms},
            "final_system": {**build(final_order, final_scores, "final_system_level5"),
                              "latency_ms": final_latency_ms},
        }

        if include_experimental_reranker:
            out["experimental_rejected_reranker"] = self._rerank_experimental(
                norm_q, lexical_scores, semantic_scores, k
            )
        return out

    def _rerank_experimental(self, norm_q, lexical_scores, semantic_scores, k):
        """Level 4, exposed ONLY behind an explicit opt-in flag. Rejected
        from the frozen architecture -- see docs/BENCHMARK_REPORT.md Section
        11 (lower validation/test MRR than fusion alone, 5x higher P95
        latency). Loads the cross-encoder lazily so the default serving path
        never pays its memory cost."""
        t0 = time.time()
        if not hasattr(self, "_cross_encoder"):
            from sentence_transformers import CrossEncoder
            try:
                self._cross_encoder = CrossEncoder(
                    self.frozen_config["cross_encoder"], device="cpu", local_files_only=True,
                )
            except Exception:
                self._cross_encoder = CrossEncoder(self.frozen_config["cross_encoder"], device="cpu")
            # Required fix -- see src/mapping.py::materialize_model_ docstring
            # and docs/CHANGE_REPORT.md "Level 4 cross-encoder bug fix": this
            # model silently emits NaN on this platform without it.
            materialize_model_(self._cross_encoder.model)
        fused = self.fusion.fuse(lexical_scores, semantic_scores)
        depth = self.frozen_config["rerank_depth"]
        top_idx = np.argsort(-fused)[:depth]
        cand_heads = [self.head_index[i] for i in top_idx]
        ce_scores = self._cross_encoder.predict(
            [(norm_q, h) for h in cand_heads], show_progress_bar=False,
        )
        order = np.argsort(-ce_scores)[:k]
        latency_ms = (time.time() - t0) * 1000.0
        return {
            "system": "experimental_rejected_reranker_level4",
            "note": "Rejected from the frozen architecture on validation evidence; shown for explainability only.",
            "latency_ms": latency_ms,
            "results": [
                {"rank": r, "head_query": cand_heads[i], "score": float(ce_scores[i])}
                for r, i in enumerate(order, start=1)
            ],
        }

    def metadata(self) -> dict:
        fm = self.final_metrics
        p95_local = None
        latency_path = ROOT / "artifacts" / "latency_benchmark.json"
        if latency_path.exists():
            lb = json.loads(latency_path.read_text())
            selected = lb.get("latency_ms", {}).get("selected_final_level5_no_reranker")
            if selected:
                p95_local = selected.get("p95")
        return {
            "p95_local_inference_benchmark_ms": p95_local,
            "head_index_size": len(self.head_index),
            "frozen_model": self.embed_model_name,
            "alpha": self.alpha,
            "beta": self.beta,
            "primary_k": self.primary_k,
            "held_out_test_size": fm["split_summary"]["n_test"],
            "baseline_mrr": fm["baseline_tail_metric"],
            "final_mrr": fm["final_tail_metric"],
            "relative_improvement_pct": fm["relative_improvement_pct"],
            "final_level": fm["final_level"],
            "baseline_level": fm["baseline_level"],
            "evaluation_note": "All MRR/improvement figures are OFFLINE HELD-OUT EVALUATION RESULTS, computed once on a frozen test split. They are not live production metrics and are not affected by demo queries typed into this UI.",
        }
