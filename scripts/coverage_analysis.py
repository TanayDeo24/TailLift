"""
Section 31: secondary, non-primary confidence-bucketed coverage/quality
breakdown for the frozen final system, using the TOP-1 fused score
(pre-rerank/prior) as the confidence signal. Does not alter the primary
(unthresholded, 100%-coverage) frozen test metric reported elsewhere.

Usage: python3 scripts/coverage_analysis.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapping import LexicalMapper, FusionMapper  # noqa: E402


def main():
    final_metrics = json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())
    final_level = final_metrics["final_level"]

    results = json.loads((ROOT / "artifacts" / "ladder_results.json").read_text())
    cfg = results["frozen_config"]
    test_rows = json.loads((ROOT / "artifacts" / "splits" / "test.json").read_text())
    head_index = json.loads((ROOT / "artifacts" / "splits" / "head_index.json").read_text())

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["embed_model"], device="cpu")
    head_embs = model.encode(head_index, batch_size=256, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    tail_embs = model.encode([r["tail_query"] for r in test_rows], batch_size=256,
                              show_progress_bar=False, convert_to_numpy=True,
                              normalize_embeddings=True).astype(np.float32)

    lex_mapper = LexicalMapper(head_index)
    fusion = FusionMapper(cfg["alpha_fusion"])
    sim_matrix = tail_embs @ head_embs.T

    top1_conf = []
    for i, row in enumerate(test_rows):
        lex_scores = lex_mapper.score_all(row["tail_query"])
        fused = fusion.fuse(lex_scores, sim_matrix[i])
        top1_conf.append(float(fused.max()))
    top1_conf = np.array(top1_conf)

    final_per_q = results["levels"]["test"][final_level]["per_query"]
    mrr = np.array([d["mrr"] for d in final_per_q])
    hit5 = np.array([d["hit@5"] for d in final_per_q])

    # quartile buckets by top-1 fusion confidence
    edges = np.quantile(top1_conf, [0, 0.25, 0.5, 0.75, 1.0])
    bucket_idx = np.clip(np.digitize(top1_conf, edges[1:-1]), 0, 3)

    buckets = []
    for b in range(4):
        mask = bucket_idx == b
        buckets.append({
            "confidence_quartile": b + 1,
            "n": int(mask.sum()),
            "confidence_range": [float(top1_conf[mask].min()), float(top1_conf[mask].max())] if mask.sum() else None,
            "mean_mrr": float(mrr[mask].mean()) if mask.sum() else None,
            "mean_hit@5": float(hit5[mask].mean()) if mask.sum() else None,
        })

    # what a hypothetical abstention threshold would buy: coverage vs quality tradeoff
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    threshold_table = []
    for t in thresholds:
        mask = top1_conf >= t
        threshold_table.append({
            "min_confidence_threshold": t,
            "coverage_pct": float(100 * mask.mean()),
            "mean_mrr_if_served": float(mrr[mask].mean()) if mask.sum() else None,
            "mean_hit@5_if_served": float(hit5[mask].mean()) if mask.sum() else None,
        })

    out = {
        "note": (
            "Secondary analysis only. The PRIMARY frozen test result "
            "(artifacts/final_metrics.json) applies NO abstention threshold "
            "(100% forced coverage). This file characterizes what a "
            "confidence-based abstention policy would trade off, for future "
            "use, without altering the primary frozen number."
        ),
        "final_level_analyzed": final_level,
        "confidence_signal": "top-1 lexical+semantic fused score (pre-rerank, pre-prior)",
        "quality_by_confidence_quartile": buckets,
        "coverage_vs_quality_by_threshold": threshold_table,
    }
    (ROOT / "artifacts" / "coverage_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
