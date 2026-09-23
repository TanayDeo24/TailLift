# Data Provenance

All figures below are computed by `scripts/audit_data.py` and persisted at
`artifacts/data_audit.json`. No number here is hand-typed without that
traceability.

## Source

The original README states the data derives from the AOL query-log release
(query text, clicked URL, click rank). **The raw AOL log itself is not present
in this repository** — confirmed by directory listing and by tracing every
`import`/file-read in the original Python scripts (see
`docs/QUERY_LOG_AUDIT.md`). Only already-derived, already-thresholded pair
files remain:

| File | Rows | Schema | Role |
|---|---|---|---|
| `newMedium.txt` | 1,439,823 | `tail_query \|\| head_query \|\| label(0/1)` | Master pair corpus — used for all modernization work in this project |
| `out (1).txt` | 100,000 | `query1 , query2 , label(-1/1)` | Second, differently-schemad pair file; source of the original train/test split |
| `train_data.txt` | 80,000 | same as above | First 80,000 lines of `out (1).txt`, byte-identical (verified) |
| `test_data.txt` | 20,000 | same as above | Last 20,000 lines of `out (1).txt`, byte-identical (verified) |
| `head150.txt` | 12,261 | one query per line | Frozen head-query vocabulary |
| `headkey.txt` | 298 URLs (149 unique) | flat printed Python list | Clicked URLs associated with head queries, no per-query linkage |
| `tailkey.txt` | 230 URLs (115 unique) | flat printed Python list | Clicked URLs associated with tail queries, no per-query linkage |
| `finalData.txt` | 0 | — | Empty |

## Fields available vs. not available

The original code (`separator.py`, `vidit.jain@ada.iiit.ac.in`) reads a raw
log via `pandas.read_table` and accesses exactly three columns: `Query`,
`ItemRank`, `ClickURL`. It never reads `AnonID` or `QueryTime`, even though
the standard AOL log schema includes both. This means:

- **No user identifiers anywhere in this project, past or present.** Neither
  the (absent) raw log processing nor any derived file carries anonymized
  user IDs.
- **No timestamps anywhere.** `QueryTime` was never consumed by the original
  pipeline and is not present in any derived file.
- **No full impression/result lists.** Only clicked (query, URL, rank)
  triples were ever used — and even the per-row rank information was
  consumed to *build* the newMedium.txt binary label and then discarded; it
  is not retained per-row in any file still present.

This is a hard constraint on the rest of the project, not a choice: sessions,
first-click-position, reformulation, no-click, and abandonment all require at
minimum timestamps or user identifiers, or (for no-click) a full impression
list. **None of these exist in this repository.** See
`docs/SESSIONIZATION.md` and `docs/SEARCH_QUALITY_DIAGNOSIS.md` for how this
is handled (excluded and documented, per the project brief's own guidance,
rather than fabricated).

## `newMedium.txt` — the master corpus used going forward

- 1,439,823 rows; 1,409,790 unique tail-query strings; 12,261 unique
  head-query strings (this set is identical to `head150.txt`, confirmed).
- Label distribution: 130,893 positive (9.09%), 1,308,930 negative (90.91%).
- Of the 1,409,790 unique tail queries, only **116,655 (8.27%)** have at
  least one positive-labeled head query recorded — i.e. have a defensible
  ground-truth mapping target under the original relevance rule (same
  clicked URL, rank ≤5 in both). The remaining 1,293,135 tail queries appear
  only in negative pairs and are **ineligible** for the primary mapping task
  (no known-correct head query to evaluate against — see
  `docs/HEAD_TAIL_METHODOLOGY.md` §Eligibility).
- Among eligible tail queries, the number of positive head targets is
  strongly single-valued: median 1, p90 1, p99 3, max 12 (mean 1.12). This
  supports treating the task as (mostly) single-relevant-target retrieval —
  MRR and Hit@1/5/10 are appropriate primary metrics (Section 18 of the
  project brief).

## `out (1).txt` / `train_data.txt` / `test_data.txt`

- `train_data.txt` is byte-identical to the first 80,000 lines of
  `out (1).txt`; `test_data.txt` is byte-identical to the last 20,000 lines.
  This is a **contiguous line-order split of one file**, not a query-grouped
  split.
- Leakage check: 4,599 unique `query1` values appear in the test portion;
  **199 of them (4.3%) also appear in the train portion.** This is a
  confirmed train/test leakage defect in the original repository's
  evaluation setup — carried into `docs/QUERY_LOG_AUDIT.md` and explicitly
  NOT reproduced in this project's own split (see
  `docs/SPLIT_METHODOLOGY.md`).
- This file's query vocabulary (sampled: "reviews on adult movie peckers...",
  "myspace background generators", "civilization 4 cheats") does not overlap
  with `head150.txt`'s vocabulary in the samples inspected — it appears to be
  a separate slice of the same underlying AOL release, used historically
  only to train/test the original SVM ensemble, not the head-query mapping
  index used in the rest of this project.

## `headkey.txt` / `tailkey.txt`

Flat, unlinked URL lists (298 / 230 entries, 149 / 115 unique). Only 4 URLs
are shared between the two lists. No per-query mapping is recoverable from
these files, so they cannot support click-position or no-click analysis at
the query level; they are used only as a coarse descriptive artifact (URL
diversity) and are not load-bearing for any reported metric.

## Privacy / sanitization policy

- **No raw user identifiers exist in any file in this repository** (verified
  above) — the acute AOL-2006-style re-identification risk (real user
  queries tied to a persistent anonymous ID) does not apply to what is
  actually present locally.
- The raw, controversial AOL query-log release itself (the one containing
  `AnonID`) is **deliberately not re-acquired** for this project, even though
  doing so would technically unlock sessionization/reformulation/abandonment
  analysis. That data release is widely known to have caused real-world harm
  through query-based re-identification of individuals in 2006, and
  reintroducing user-linked raw queries into a project intended for public
  hosting is not a legitimate scope expansion. This is a considered
  engineering + privacy decision, not an oversight — see `docs/FINAL_REPORT.md`
  Limitations.
- All example queries used in documentation in this project are drawn from
  the already-present, already-anonymized derived files (no user ID
  attached) and are representative, ordinary web-navigational/informational
  queries (e.g. "civilization 4 cheats", "myspace backgrounds"). A small
  number of adult-content query strings exist in `out (1).txt`/`train_data.txt`
  (inherited from the original repository, not introduced here); none are
  reproduced verbatim in any `docs/` file or report — only aggregate
  statistics about that file are cited.
