# Search-Quality Diagnosis

Every number below is from `artifacts/data_audit.json`,
`artifacts/sparsity_correlation.json`, or `artifacts/final_metrics.json` —
produced by `scripts/audit_data.py`, `scripts/sparsity_correlation.py`,
`scripts/build_report_artifacts.py` respectively.

## 1. How large is the long tail?

1,409,790 unique tail-query strings vs. 12,261 unique head-query strings in
`newMedium.txt` — the tail vocabulary outnumbers the head vocabulary by
**115:1**. Of those tail queries, 116,655 (8.27%) have at least one
positive-labeled head-query target under the original relevance rule; the
remaining 91.73% have no recorded relevant head at all under that rule.

## 2. What percentage of "traffic" does it represent?

This repository does not retain raw occurrence counts (only the
already-thresholded output survives — see `docs/DATA_PROVENANCE.md`), so a
true traffic-share number (queries weighted by how often they were searched)
cannot be computed. What can be measured: **row share** in the pair corpus —
tail queries generate 1,439,823 total rows against only 12,261 distinct head
queries, i.e. a small head vocabulary is the target of the overwhelming
majority of mapping attempts. This is a row-count statement, not a traffic
statement, and is reported as such.

## 3. How sparse is tail evidence?

Among the 116,655 eligible tail queries, the number of recorded relevant
head targets is heavily concentrated at 1: median 1, p90 1, p99 3, max 12
(`artifacts/data_audit.json` -> `newMedium.n_positive_heads_per_tail_percentiles`).
Compare this to the head-query side: every head query, by construction, is a
member of a 150+-occurrence class in the original raw log (per
`separator.py`'s threshold) — a categorically larger evidence base than any
individual tail query can have (tail queries were capped at occurrence count
< 5 by the same original method). This asymmetry — tail queries limited to
observing at most a handful of relevant signals vs. head queries built from
150+ occurrences — **is** the sparsity this project addresses; it is
definitional to the original head/tail construction, not independently
re-derived here (see `docs/HEAD_TAIL_METHODOLOGY.md` for why raw counts are
unavailable for a from-scratch re-derivation).

## 4. Does tail-query mapping quality differ from a reference ceiling?

Yes, substantially. Using this project's own held-out test evaluation:

| | MRR@5-class metric (mean reciprocal rank) |
|---|---|
| Head-query self-retrieval reference ceiling | **0.998** |
| Tail→head mapping, original method reproduced (Level 0/1 lexical) | **0.147** |
| Tail→head mapping, final selected system (Level 5) | **0.266** |

The reference ceiling is a head query matched against the same index it was
drawn from (near-certain self-identity retrieval) — see
`docs/BENCHMARK_REPORT.md` for the exact definition and why it is a ceiling
reference, not an independently-measured production head-query metric (no
raw head-only click logs survive to measure that directly). The **gap**
between that ceiling and tail-query mapping quality is 0.85 (baseline) /
0.73 (after this project's intervention) — see `docs/BENCHMARK_REPORT.md`
§Gap.

## 5. What behavior suggests greater search difficulty for tail queries?

The only evidence-density signal available in this repository (row "degree"
in `newMedium.txt` — how many head-query candidates a tail query was ever
compared against under the original click-URL-sharing method) shows a
**near-zero, statistically insignificant-in-practice correlation** with
final mapping quality (Spearman ρ = -0.026, p<0.001, n=17,499 — statistically
significant only because of the large sample size, not because the effect
is meaningfully sized) and a small positive correlation with baseline
lexical quality (ρ = 0.068). This is a genuine **negative/null result**,
reported honestly rather than omitted: 13,823 of the 17,499 held-out tail
queries (79%) have the minimum possible degree value (1), leaving too
little variance in this particular evidence-density proxy to explain
quality differences. The proxy itself is a weak stand-in for true historical
search frequency (which is not recoverable — see
`docs/DATA_PROVENANCE.md`), and this project does not overstate what it
shows.

## 6. Which failure modes dominate?

See `docs/FAILURE_ANALYSIS.md` for the full breakdown with sanitized
examples. Headline: the final system outright beats the reproduced original
baseline on 81.3% of held-out tail queries, loses on 7.2%, and both
methods reach a correct top-1 answer together on 10.4% — see that document
for concrete (non-identifying) examples of each.

## 7. What intervention opportunity follows from these facts?

Because the head-query self-retrieval ceiling (0.998) is far above even the
best tail-query mapping result (0.266), there is substantial legitimate
headroom for tail-query search quality to improve without any claim that the
gap can be closed to zero — a tail query with a genuinely ambiguous or
under-specified information need cannot be mapped to a single "correct" head
query with certainty, ceiling or not. This project demonstrates a concrete,
measured, statistically validated step in that direction (Section 34's gap
metric) using only evidence and methodology this repository can actually
support, and documents rather than hides the two evidentiary limits that
constrain how far the diagnosis can go: no raw traffic/frequency data, and a
weak historical evidence-density proxy.
