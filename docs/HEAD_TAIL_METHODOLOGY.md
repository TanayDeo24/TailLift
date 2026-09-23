# Head / Tail Query Definition

## Why this definition is INHERITED, not re-derived from raw frequency

The project brief's default procedure (Section 11) is to build a frequency
histogram from raw query-log occurrence counts and select thresholds from
percentiles/traffic-share curves, independent of model performance. That
procedure requires the raw, per-occurrence query log.

As established in `docs/DATA_PROVENANCE.md` and `docs/QUERY_LOG_AUDIT.md`,
**the raw log is not present in this repository** — only files that are
already the *output* of a head/tail thresholding step the original authors
ran on data we no longer have access to (`head150.txt`, and the tail-side
vocabulary implicit in `newMedium.txt`). We cannot rebuild the frequency
histogram, because we do not have the counts.

Given that constraint, re-deriving a fresh, independently-validated threshold
(as Section 11 Steps 1–7 describe) is not honestly possible here. This is a
data-availability limitation, not a shortcut — documented explicitly rather
than silently inheriting the number without saying so.

## What is inherited, and from where

- **Head-query vocabulary**: the 12,261 queries in `head150.txt`, which is
  exactly the head-query column vocabulary of `newMedium.txt` (verified,
  `artifacts/data_audit.json`). Per `separator.py`, this corresponds to
  queries with historical occurrence count > 150 in the (now-absent) raw
  log. (`vidit.jain@ada.iiit.ac.in`'s divergent copy used a threshold of 100
  instead — see `docs/QUERY_LOG_AUDIT.md`. Since `head150.txt`'s size
  (12,261) and the file's own name both align with the `separator.py`/150
  variant, and the 150 variant matches the published README, this project
  treats 150 as the operative historical threshold — an inference from
  artifact evidence, not a re-derivation from raw counts.)
- **Tail-query population**: every distinct string in `newMedium.txt`'s
  tail column (1,409,790 unique queries) — by the same original logic,
  queries with historical occurrence count `< 5`.
- **Torso queries** (occurrence count strictly between the tail and head
  cutoffs) were, per the original method, simply never assigned to either
  bucket and never appear in `newMedium.txt`. This project inherits that
  omission; torso-query search quality is out of scope here, exactly as it
  was in the original project.

## Eligibility (this project's own addition, not present in the original)

Not every tail query in `newMedium.txt` has a ground-truth relevant head
query recorded. Per `artifacts/data_audit.json`:

- **116,655 tail queries (8.27%)** have ≥1 positive-labeled head query.
  These are **ELIGIBLE** for the primary tail→head mapping task.
- **1,293,135 tail queries (91.73%)** appear only in negative-labeled pairs
  — under the original relevance rule (shared clicked URL, rank ≤5 both
  sides), no head query was ever found to be relevant to them. These are
  **INELIGIBLE**: there is no defensible target to evaluate a mapping
  against, so they are excluded from train/val/test (never silently
  dropped — every one is accounted for in `artifacts/splits/ineligible_tail_queries.json`,
  and the exclusion is reported as a coverage fact, not hidden).

This eligible/ineligible split is decided purely from label structure
(whether a positive target exists), before any model is trained or
evaluated — it does not depend on, and is not tuned against, model
performance, satisfying the brief's independence requirement even though the
underlying threshold itself is inherited rather than re-derived.

## Sample-size adequacy (Section 12 gate)

Of the 116,655 eligible tail queries, this project's query-grouped 70/15/15
split (`docs/SPLIT_METHODOLOGY.md`) yields:

- 81,658 training tail queries
- 17,498 validation tail queries
- 17,499 held-out test tail queries

All three populations comfortably exceed what is needed for stable
aggregate-metric estimation and paired bootstrap confidence intervals
(Section 42 requires the method to work at all; 17K+ paired observations is
far beyond the minimum for a stable bootstrap CI on a proportion-like metric
such as MRR or Recall@K). **Adequacy gate: PASSED**, no small-sample caveat
required for the primary held-out test population.

## Head-query candidate index

The retrieval candidate set for every mapping is the full 12,261-query
`head150.txt` vocabulary (`artifacts/splits/head_index.json`), held fixed
across train/val/test. This is not a leakage risk: the index is a static
candidate catalog (which head queries exist to be mapped to), not
information about which specific head query is correct for any held-out
tail query. See `docs/SPLIT_METHODOLOGY.md` for the leakage argument in
full.
