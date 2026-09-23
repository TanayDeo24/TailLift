"""
Section 13: does weaker historical evidence correspond to worse mapping
quality? Correlates each held-out test tail query's evidence volume (its row
"degree" in newMedium.txt -- how many head-query candidates it was ever
compared against under the original click-URL-sharing method) against this
project's own measured per-query MRR (Level 4 rerank, the frozen final
system) on the test split.

Requires artifacts/ladder_results.json to already exist (run scripts/run_ladder.py first).

Usage: python3 scripts/sparsity_correlation.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def norm(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def main():
    degree = Counter()
    with (ROOT / "newMedium.txt").open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("||")
            if len(parts) != 3:
                continue
            degree[norm(parts[0])] += 1

    results = json.loads((ROOT / "artifacts" / "ladder_results.json").read_text())
    test_block = results["levels"]["test"]
    order = test_block["tail_query_order"]
    final_per_q = test_block["level4_rerank"]["per_query"]
    baseline_per_q = test_block["level0_1_lexical"]["per_query"]

    degrees = np.array([degree[q] for q in order])
    final_mrr = np.array([d["mrr"] for d in final_per_q])
    baseline_mrr = np.array([d["mrr"] for d in baseline_per_q])

    spearman_final = stats.spearmanr(degrees, final_mrr)
    spearman_baseline = stats.spearmanr(degrees, baseline_mrr)

    # decile analysis
    decile_edges = np.quantile(degrees, np.linspace(0, 1, 11))
    deciles = np.digitize(degrees, decile_edges[1:-1])
    decile_stats = []
    for d in range(10):
        mask = deciles == d
        if mask.sum() == 0:
            continue
        decile_stats.append({
            "decile": d,
            "n": int(mask.sum()),
            "degree_range": [float(degrees[mask].min()), float(degrees[mask].max())],
            "mean_final_mrr": float(final_mrr[mask].mean()),
            "mean_baseline_mrr": float(baseline_mrr[mask].mean()),
        })

    out = {
        "n_test_queries": len(order),
        "evidence_metric": "pair_degree_in_newMedium.txt (proxy for available comparison evidence, NOT raw AOL frequency)",
        "spearman_correlation_evidence_vs_final_mrr": {
            "rho": float(spearman_final.correlation), "p_value": float(spearman_final.pvalue),
        },
        "spearman_correlation_evidence_vs_baseline_mrr": {
            "rho": float(spearman_baseline.correlation), "p_value": float(spearman_baseline.pvalue),
        },
        "quality_by_evidence_decile": decile_stats,
        "interpretation_caveat": (
            "pair_degree here counts how many (tail,head) rows a query has in "
            "newMedium.txt, which reflects how many head queries shared a "
            "clicked URL with it under the original method -- it is a proxy "
            "for available evidence density, not the query's raw AOL search "
            "frequency (which is not recoverable, see docs/DATA_PROVENANCE.md)."
        ),
    }
    (ROOT / "artifacts" / "sparsity_correlation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
