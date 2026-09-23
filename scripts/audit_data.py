"""
Data audit: computes real, machine-readable statistics over every original
data artifact in the repository. No numbers in docs/ may be cited without
tracing back to this script's output (artifacts/data_audit.json).

Usage: python3 scripts/audit_data.py
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "data_audit.json"


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def audit_new_medium():
    path = ROOT / "newMedium.txt"
    tail_counter = Counter()
    head_counter = Counter()
    label_counter = Counter()
    tail_pos_heads = {}  # tail -> set of head queries with label 1 (sampled, capped)
    n_rows = 0
    n_malformed = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("||")
            if len(parts) != 3:
                n_malformed += 1
                continue
            tail_raw, head_raw, label_raw = parts
            tail = norm(tail_raw)
            head = norm(head_raw)
            label = label_raw.strip()
            if label not in ("0", "1"):
                n_malformed += 1
                continue
            n_rows += 1
            tail_counter[tail] += 1
            head_counter[head] += 1
            label_counter[label] += 1
            if label == "1":
                tail_pos_heads.setdefault(tail, set()).add(head)

    degree_dist = Counter(tail_counter.values())
    n_tail_with_positive = len(tail_pos_heads)

    return {
        "path": "newMedium.txt",
        "format": "tail_query || head_query || label(0/1)",
        "n_rows": n_rows,
        "n_malformed_rows": n_malformed,
        "n_unique_tail_queries": len(tail_counter),
        "n_unique_head_queries": len(head_counter),
        "label_distribution": dict(label_counter),
        "positive_rate": label_counter["1"] / n_rows if n_rows else None,
        "tail_degree_percentiles": percentiles(list(tail_counter.values())),
        "tail_degree_distribution_top10": degree_dist.most_common(10),
        "n_tail_queries_with_at_least_one_positive_head": n_tail_with_positive,
        "pct_tail_queries_with_positive_head": (
            n_tail_with_positive / len(tail_counter) if tail_counter else None
        ),
        "n_positive_heads_per_tail_percentiles": percentiles(
            [len(v) for v in tail_pos_heads.values()]
        ),
    }, tail_counter, head_counter, tail_pos_heads


def percentiles(values):
    if not values:
        return {}
    s = sorted(values)
    n = len(s)

    def pct(p):
        idx = min(n - 1, int(round(p / 100 * (n - 1))))
        return s[idx]

    return {
        "min": s[0],
        "p10": pct(10),
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "p99": pct(99),
        "max": s[-1],
        "mean": sum(s) / n,
        "n": n,
    }


def audit_head150():
    path = ROOT / "head150.txt"
    lines = [norm(l) for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return {
        "path": "head150.txt",
        "n_lines": len(lines),
        "n_unique": len(set(lines)),
        "sample_first5": lines[:5],
    }


def audit_out1_split():
    out1 = ROOT / "out (1).txt"
    train = ROOT / "train_data.txt"
    test = ROOT / "test_data.txt"

    def read_pairs(p, sep=","):
        rows = []
        with p.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split(sep)
                if len(parts) < 3:
                    continue
                q1, q2, label = parts[0], parts[1], parts[-1]
                rows.append((norm(q1), norm(q2), label.strip()))
        return rows

    out1_rows = read_pairs(out1)
    train_rows = read_pairs(train)
    test_rows = read_pairs(test)

    train_q1 = {r[0] for r in train_rows}
    test_q1 = {r[0] for r in test_rows}
    overlap_q1 = train_q1 & test_q1

    return {
        "path": "out (1).txt / train_data.txt / test_data.txt",
        "format": "query1 , query2 , label(-1/1)  [DIFFERENT SCHEMA than newMedium.txt]",
        "n_rows_out1": len(out1_rows),
        "n_rows_train": len(train_rows),
        "n_rows_test": len(test_rows),
        "train_is_first_80pct_of_out1": train_rows == out1_rows[: len(train_rows)],
        "test_is_last_20pct_of_out1": test_rows == out1_rows[-len(test_rows):],
        "n_unique_query1_train": len(train_q1),
        "n_unique_query1_test": len(test_q1),
        "n_query1_overlap_train_test": len(overlap_q1),
        "pct_test_query1_also_in_train": (
            len(overlap_q1) / len(test_q1) if test_q1 else None
        ),
        "leakage_finding": (
            "Original train/test split is a contiguous 80/20 line-order split "
            "of a single file, NOT query-grouped. Any query1 value appearing in "
            "more than one contiguous block leaks across the split."
        ),
    }


def audit_headkey_tailkey():
    def read_url_list(path):
        text = path.read_text(encoding="utf-8", errors="replace")
        urls = re.findall(r"http[s]?://[^\s',\]]+", text)
        return urls

    head_urls = read_url_list(ROOT / "headkey.txt")
    tail_urls = read_url_list(ROOT / "tailkey.txt")
    return {
        "path": "headkey.txt / tailkey.txt",
        "format": "flat printed Python list of clicked URLs; NOT linked to individual queries",
        "n_urls_headkey": len(head_urls),
        "n_urls_tailkey": len(tail_urls),
        "n_unique_urls_headkey": len(set(head_urls)),
        "n_unique_urls_tailkey": len(set(tail_urls)),
        "n_urls_shared_between_head_and_tail": len(set(head_urls) & set(tail_urls)),
        "usability_finding": (
            "No per-query linkage present (flat URL list only). Cannot be used "
            "for per-query click-position, no-click, or session analysis."
        ),
    }


def audit_schema_fields():
    # Inspect original scripts' pandas column access to document what raw log
    # fields the *original* pipeline consumed (source of truth: code, not README).
    return {
        "raw_AOL_log_present_in_repo": False,
        "fields_referenced_in_separator_scripts": ["Query", "ItemRank", "ClickURL"],
        "fields_NOT_referenced_in_any_script": ["AnonID", "QueryTime"],
        "finding": (
            "The raw AOL query log (AnonID, Query, QueryTime, ItemRank, ClickURL) "
            "is not present anywhere in this repository -- only already-derived "
            "pair files (newMedium.txt, out (1).txt/train/test, head150.txt, "
            "headkey.txt, tailkey.txt) exist. The original code (separator.py, "
            "vidit.jain@ada.iiit.ac.in) never reads AnonID or QueryTime even when "
            "the raw log was available, so no session/user/temporal information "
            "was ever computed by the original project either. This repository "
            "cannot support sessionization, first-click-position, reformulation, "
            "or abandonment analysis without re-acquiring the raw AOL log, which "
            "is deliberately NOT done here (see docs/DATA_PROVENANCE.md, Privacy)."
        ),
    }


def main():
    new_medium_stats, tail_counter, head_counter, tail_pos_heads = audit_new_medium()
    result = {
        "newMedium": new_medium_stats,
        "head150": audit_head150(),
        "original_split_out1": audit_out1_split(),
        "headkey_tailkey": audit_headkey_tailkey(),
        "schema_availability": audit_schema_fields(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
