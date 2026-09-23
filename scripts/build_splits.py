"""
Builds a leakage-safe, query-grouped train/val/test split over newMedium.txt.

Design decision (see docs/SPLIT_METHODOLOGY.md):
  - No timestamps or user IDs exist in this repository (see docs/DATA_PROVENANCE.md),
    so a temporal or user-grouped split is not possible.
  - The original repo's out (1).txt/train_data.txt/test_data.txt split is a
    contiguous 80/20 line-order split with confirmed query overlap leakage
    (4.3% of test query1 values also appear in train -- see artifacts/data_audit.json).
  - We instead group by TAIL QUERY (the prediction unit) and assign every row
    for a given tail query to exactly one split. This guarantees no tail query
    is ever seen in more than one split.
  - Only "eligible" tail queries (>=1 positive-labeled head in newMedium.txt)
    are usable for the primary mapping task; the rest are recorded as
    ineligible (no ground-truth relevant head exists under the original
    relevance definition) and excluded from train/val/test, never silently
    dropped without being counted.

Usage: python3 scripts/build_splits.py
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260922  # frozen seed, date-derived, documented in docs/REPRODUCIBILITY.md
random.seed(SEED)

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def main():
    path = ROOT / "newMedium.txt"
    tail_to_heads = defaultdict(dict)  # tail -> {head: label}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("||")
            if len(parts) != 3:
                continue
            tail, head, label = norm(parts[0]), norm(parts[1]), parts[2].strip()
            if label not in ("0", "1"):
                continue
            # keep the max label if a (tail,head) pair repeats (positive wins)
            prev = tail_to_heads[tail].get(head)
            tail_to_heads[tail][head] = max(prev or "0", label, key=int)

    eligible, ineligible = [], []
    for tail, heads in tail_to_heads.items():
        if any(l == "1" for l in heads.values()):
            eligible.append(tail)
        else:
            ineligible.append(tail)

    eligible_sorted = sorted(eligible)  # deterministic order before shuffling
    random.shuffle(eligible_sorted)
    n = len(eligible_sorted)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_q = set(eligible_sorted[:n_train])
    val_q = set(eligible_sorted[n_train:n_train + n_val])
    test_q = set(eligible_sorted[n_train + n_val:])

    assert train_q.isdisjoint(val_q) and train_q.isdisjoint(test_q) and val_q.isdisjoint(test_q)

    # Head-query candidate index: fixed vocabulary (head150.txt), independent
    # of which split a tail query belongs to -- see docs/SPLIT_METHODOLOGY.md
    # for why this does not constitute leakage.
    head_index = sorted(
        {norm(l) for l in (ROOT / "head150.txt").read_text(errors="replace").splitlines() if l.strip()}
    )

    out_dir = ROOT / "artifacts" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump_split(name, q_set):
        rows = []
        for tail in q_set:
            heads = tail_to_heads[tail]
            positives = sorted([h for h, l in heads.items() if l == "1"])
            negatives = sorted([h for h, l in heads.items() if l == "0"])
            rows.append({"tail_query": tail, "positive_heads": positives, "negative_heads": negatives})
        (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2))
        return rows

    train_rows = dump_split("train", train_q)
    val_rows = dump_split("val", val_q)
    test_rows = dump_split("test", test_q)
    (out_dir / "head_index.json").write_text(json.dumps(head_index, indent=2))
    (out_dir / "ineligible_tail_queries.json").write_text(
        json.dumps(sorted(ineligible), indent=2)
    )

    summary = {
        "seed": SEED,
        "split_unit": "tail_query (query-grouped, no query appears in >1 split)",
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "n_eligible_tail_queries": n,
        "n_ineligible_tail_queries": len(ineligible),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "n_head_index": len(head_index),
        "leakage_check_disjoint_sets": True,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
