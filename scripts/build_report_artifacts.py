"""
Consumes artifacts/ladder_results.json and produces the final, report-ready
machine-readable metrics bundle: gap analysis, coverage, paired bootstrap CI,
and offline click/target-recovery framing (Section 33-36, 46).

Usage: python3 scripts/build_report_artifacts.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import paired_bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_K = 5
PRIMARY_METRIC = "mrr"


def per_query_value(level_block, metric):
    return [d[metric] for d in level_block["per_query"]]


def main():
    results = json.loads((ROOT / "artifacts" / "ladder_results.json").read_text())
    split_summary = json.loads((ROOT / "artifacts" / "splits" / "split_summary.json").read_text())

    test = results["levels"]["test"]
    ceiling = results["head_self_retrieval_ceiling"]["agg"][PRIMARY_METRIC]

    baseline_level = "level0_1_lexical"  # Level 0/1: original repo's Jaccard method, reproduced

    # Architecture selection is VALIDATION-ONLY (Section 20/38/39): pick the
    # ladder level with the highest val MRR among the candidate levels, not a
    # hardcoded assumption. This is decided before any test-split number is
    # inspected below.
    val_block = results["levels"]["val"]
    candidate_levels = ["level2_semantic", "level3_fusion", "level4_rerank", "level5_support_prior"]
    val_mrr_by_level = {lvl: val_block[lvl]["agg"]["mrr"] for lvl in candidate_levels}
    final_level = max(val_mrr_by_level, key=val_mrr_by_level.get)

    baseline_vals = per_query_value(test[baseline_level], PRIMARY_METRIC)
    final_vals = per_query_value(test[final_level], PRIMARY_METRIC)

    baseline_tail_metric = sum(baseline_vals) / len(baseline_vals)
    final_tail_metric = sum(final_vals) / len(final_vals)

    baseline_gap = ceiling - baseline_tail_metric
    final_gap = ceiling - final_tail_metric
    gap_reduction_abs = baseline_gap - final_gap
    gap_reduction_pct = (gap_reduction_abs / baseline_gap) if baseline_gap else None

    observed, lo, hi, n, resamples, seed = paired_bootstrap_ci(baseline_vals, final_vals, n_resamples=5000)

    # coverage: every eligible test tail query IS mapped (no abstention
    # threshold is applied to the primary reported numbers above -- this
    # project always returns a ranked list). We additionally report
    # confidence-bucketed quality using the fusion score of the top-1
    # result as a confidence signal, and what coverage a threshold would buy,
    # WITHOUT changing the primary (unthresholded) frozen result.
    coverage = {
        "eligible_test_tail_queries": len(baseline_vals),
        "mapped_test_tail_queries": len(baseline_vals),
        "abstained_test_tail_queries": 0,
        "coverage_pct": 100.0,
        "note": (
            "Primary evaluation applies no abstention threshold (forced "
            "coverage = 100%). See docs/BENCHMARK_REPORT.md 'Confidence "
            "bucketing' for a secondary, non-primary breakdown by top-1 "
            "fusion-score confidence bucket, computed only to characterize "
            "where errors concentrate -- not used to alter the frozen primary "
            "test metric."
        ),
    }

    levels_test_summary = {
        lvl: results["levels"]["test"][lvl]["agg"]
        for lvl in ("level0_1_lexical", "level2_semantic", "level3_fusion", "level4_rerank", "level5_support_prior")
    }
    levels_val_summary = {
        lvl: results["levels"]["val"][lvl]["agg"]
        for lvl in ("level0_1_lexical", "level2_semantic", "level3_fusion", "level4_rerank", "level5_support_prior")
    }

    out = {
        "primary_k": PRIMARY_K,
        "primary_metric": PRIMARY_METRIC,
        "split_summary": split_summary,
        "frozen_config": results["frozen_config"],
        "ladder_val": levels_val_summary,
        "ladder_test": levels_test_summary,
        "head_self_retrieval_ceiling": results["head_self_retrieval_ceiling"]["agg"],
        "architecture_selection": {
            "method": "highest MRR on full validation split, Section 20/38/39",
            "val_mrr_by_candidate_level": val_mrr_by_level,
            "selected": final_level,
        },
        "baseline_level": baseline_level,
        "final_level": final_level,
        "baseline_tail_metric": baseline_tail_metric,
        "final_tail_metric": final_tail_metric,
        "absolute_delta": final_tail_metric - baseline_tail_metric,
        "relative_improvement_pct": (
            (final_tail_metric - baseline_tail_metric) / baseline_tail_metric * 100
            if baseline_tail_metric else None
        ),
        "paired_bootstrap": {
            "observed_delta": observed, "ci_low": lo, "ci_high": hi,
            "n": n, "n_resamples": resamples, "seed": seed,
        },
        "baseline_gap": baseline_gap,
        "final_gap": final_gap,
        "gap_reduction_absolute": gap_reduction_abs,
        "gap_reduction_pct": gap_reduction_pct * 100 if gap_reduction_pct is not None else None,
        "coverage": coverage,
    }
    (ROOT / "artifacts" / "final_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
