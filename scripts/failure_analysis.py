"""
Section 43: sample sanitized, real (not fabricated) examples from the test
split for each failure/success category the brief asks about, using the
frozen ladder results. Categories with zero real occurrences are reported as
such, not invented.

Usage: python3 scripts/failure_analysis.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_PER_CATEGORY = 4


def main():
    results = json.loads((ROOT / "artifacts" / "ladder_results.json").read_text())
    final_metrics = json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())
    baseline_level = final_metrics["baseline_level"]
    final_level = final_metrics["final_level"]  # validation-selected, Section 20/38/39

    test_rows = json.loads((ROOT / "artifacts" / "splits" / "test.json").read_text())
    test_block = results["levels"]["test"]
    order = test_block["tail_query_order"]
    row_by_query = {r["tail_query"]: r for r in test_rows}

    baseline = test_block[baseline_level]["per_query"]
    final = test_block[final_level]["per_query"]

    categories = {
        "baseline_wins_final_loses": [],   # original lexical method beats modern reranker
        "final_wins_baseline_loses": [],   # modern reranker beats lexical baseline
        "both_fail": [],                   # neither recovers the correct head at all (mrr=0 for both)
        "both_succeed_at_1": [],           # both get it right at rank 1
    }

    for i, q in enumerate(order):
        b, f = baseline[i], final[i]
        entry = {
            "tail_query": q,
            "true_positive_heads": row_by_query[q]["positive_heads"],
            "baseline_mrr": b["mrr"],
            "final_mrr": f["mrr"],
        }
        if b["mrr"] == 0.0 and f["mrr"] == 0.0:
            categories["both_fail"].append(entry)
        elif b["hit@1"] == 1.0 and f["hit@1"] == 1.0:
            categories["both_succeed_at_1"].append(entry)
        elif b["mrr"] > f["mrr"]:
            categories["baseline_wins_final_loses"].append(entry)
        elif f["mrr"] > b["mrr"]:
            categories["final_wins_baseline_loses"].append(entry)

    out = {"baseline_level": baseline_level, "final_level": final_level}
    for cat, entries in categories.items():
        out[cat] = {
            "total_count_in_test_set": len(entries),
            "pct_of_test_set": round(100 * len(entries) / len(order), 2),
            "sample": entries[:N_PER_CATEGORY],
        }

    (ROOT / "artifacts" / "failure_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
