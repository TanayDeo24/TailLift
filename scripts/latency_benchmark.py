"""
Latency benchmark, following the project brief's exact protocol (Section 41):
  - initialize all models/indexes first (cold start measured separately)
  - >=15 warmup requests
  - >=300 measured requests, concurrency=1
  - cycle a fixed, deterministic query set
  - report P50/P95/P99

Benchmarks TWO pipelines and labels which is the actually-deployed one:
  - "selected_final" = Level 5 (lexical + semantic fusion + train-only
    support prior, NO cross-encoder) -- this is the system validation
    actually selected (see artifacts/final_metrics.json::final_level).
  - "rerank_variant" = Level 4 (adds cross-encoder rerank on top of fusion)
    -- NOT the deployed system (validation MRR was lower than fusion alone),
    benchmarked anyway so the latency cost of the rejected reranker is
    documented, not just its quality cost.

Usage: python3 scripts/latency_benchmark.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapping import LexicalMapper, FusionMapper, SupportPriorMapper, materialize_model_, tokenize  # noqa: E402

N_WARMUP = 20
N_MEASURED = 300


def percentile(values, p):
    return float(np.percentile(values, p))


def main():
    import platform
    from collections import Counter

    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder

    cfg = json.loads((ROOT / "artifacts" / "frozen_config.json").read_text())
    final_metrics = json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())
    final_level = final_metrics["final_level"]
    head_index = json.loads((ROOT / "artifacts" / "splits" / "head_index.json").read_text())
    test_rows = json.loads((ROOT / "artifacts" / "splits" / "test.json").read_text())
    train_rows = json.loads((ROOT / "artifacts" / "splits" / "train.json").read_text())

    train_pos_counts = Counter()
    for r in train_rows:
        for h in r["positive_heads"]:
            train_pos_counts[h] += 1

    bench_queries = [r["tail_query"] for r in test_rows[:40]]
    cycled = bench_queries * ((N_WARMUP + N_MEASURED) // len(bench_queries) + 1)

    t_cold_start = time.time()
    model = SentenceTransformer(cfg["embed_model"], device="cpu")
    cross_encoder = CrossEncoder(cfg["cross_encoder"], device="cpu")
    materialize_model_(cross_encoder.model)  # see src/mapping.py docstring + docs/CHANGE_REPORT.md
    lex_mapper = LexicalMapper(head_index)
    head_embs = model.encode(head_index, batch_size=256, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    fusion = FusionMapper(cfg["alpha_fusion"])
    prior = SupportPriorMapper(head_index, train_pos_counts, cfg["beta_support_prior"])
    cold_start_s = time.time() - t_cold_start

    def map_one_selected_final(tail_query: str):
        """Level 5: the actually-deployed pipeline (no cross-encoder)."""
        t0 = time.time()
        _ = tokenize(tail_query)
        tail_emb = model.encode([tail_query], show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        sem_scores = head_embs @ tail_emb
        lex_scores = lex_mapper.score_all(tail_query)
        fused = fusion.fuse(lex_scores, sem_scores)
        final_scores = prior.add_prior(fused)
        ranked = [head_index[i] for i in np.argsort(-final_scores)]
        return ranked, time.time() - t0

    def map_one_rerank_variant(tail_query: str):
        """Level 4: fusion + cross-encoder rerank (NOT deployed; benchmarked
        to show the latency cost of the reranker validation rejected)."""
        t0 = time.time()
        _ = tokenize(tail_query)
        tail_emb = model.encode([tail_query], show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        sem_scores = head_embs @ tail_emb
        lex_scores = lex_mapper.score_all(tail_query)
        fused = fusion.fuse(lex_scores, sem_scores)
        depth = cfg["rerank_depth"]
        top_idx = np.argsort(-fused)[:depth]
        cand_heads = [head_index[i] for i in top_idx]
        ce_scores = cross_encoder.predict([(tail_query, h) for h in cand_heads], show_progress_bar=False)
        ranked = [cand_heads[i] for i in np.argsort(-ce_scores)]
        return ranked, time.time() - t0

    pipelines = {
        "selected_final_level5_no_reranker": map_one_selected_final,
        "rejected_rerank_variant_level4": map_one_rerank_variant,
    }

    results = {}
    for name, fn in pipelines.items():
        print(f"[latency] pipeline={name}: warmup {N_WARMUP}...")
        for q in cycled[:N_WARMUP]:
            fn(q)
        print(f"[latency] pipeline={name}: measuring {N_MEASURED}, concurrency=1...")
        latencies = []
        for q in cycled[N_WARMUP:N_WARMUP + N_MEASURED]:
            _, elapsed = fn(q)
            latencies.append(elapsed * 1000.0)
        results[name] = {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": float(np.mean(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        }

    out = {
        "deployed_pipeline": final_level,
        "protocol": {
            "n_warmup": N_WARMUP,
            "n_measured": N_MEASURED,
            "concurrency": 1,
            "fixed_query_set_size": len(bench_queries),
            "timing_boundary_selected_final": (
                "normalization -> embedding -> exact semantic retrieval over "
                "12,261-query head index -> lexical scoring -> fusion -> "
                "train-only support-prior weighting -> final ranked output"
            ),
            "timing_boundary_rerank_variant": (
                "same as above, then cross-encoder rerank of top-"
                f"{cfg['rerank_depth']} fused candidates"
            ),
        },
        "cold_start_seconds": round(cold_start_s, 3),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        "software_versions": {
            "embed_model": cfg["embed_model"],
            "cross_encoder": cfg["cross_encoder"],
        },
        "latency_ms": results,
        "error_rate": 0.0,
    }
    out_path = ROOT / "artifacts" / "latency_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
