# Split Methodology / Leakage Audit

## Leakage audit performed

| Leakage vector | Checked? | Finding |
|---|---|---|
| Exact query duplication across splits | Yes | Prevented by construction (query-grouped split, see below) |
| Normalized-query duplication (case/whitespace) | Yes | Normalization (`lower()`, whitespace-collapse) applied before grouping, so `"AOL Update"` and `"aol   update"` are treated as one entity and cannot land in two splits |
| Near-duplicate query leakage (fuzzy) | Partially | Not deduplicated beyond exact-normalized match; token-level near-duplicates (e.g. plural/singular variants) could theoretically appear on both sides. Not treated as material given the observed <5% original-split overlap this design already improves on by orders of magnitude (0% vs 4.3%). |
| User leakage | N/A | No user IDs exist in any file (`docs/DATA_PROVENANCE.md`) — nothing to leak |
| Session leakage | N/A | No session data exists |
| Future click/frequency/head-support leakage | Yes | The head-query candidate index (`head150.txt`) is a fixed, pre-existing vocabulary independent of any specific tail query's split assignment; no held-out tail query's label ever informs which heads exist in the index or any training-time statistic (support-prior counts are computed from the TRAIN split only — see `src/mapping.py::SupportPriorMapper`, populated only from `artifacts/splits/train.json`) |
| Mapping-pair leakage | Yes | Every `(tail, head, label)` row for a given tail query is assigned to exactly one split as a unit — no tail query has some of its pairs in train and others in test |

## Why temporal / user-grouped / session-grouped splits are not used

The project brief's default preference order is chronological > user-grouped
> session-grouped > query-grouped. As established repeatedly
(`docs/DATA_PROVENANCE.md`), there are no timestamps and no user/session
identifiers anywhere in this repository. Chronological, user-grouped, and
session-grouped splits are therefore not possible. **Query-grouped is not
just preferred here — it is the only leakage-resistant option the data
supports**, and it directly targets the leakage vector that matters most for
this task: the same tail query being both a training example and a test
example (which would let a model "memorize" the correct head for a specific
query string rather than generalize).

## Design

- **Split unit**: tail query. Every `(tail_query, {positive_heads},
  {negative_heads})` record is assigned to exactly one of train/val/test.
- **Fractions**: 70% train / 15% val / 15% test, applied only to the 116,655
  ELIGIBLE tail queries (see `docs/HEAD_TAIL_METHODOLOGY.md`).
- **Randomization**: queries sorted deterministically, then shuffled with a
  fixed seed (`20260922`, `random.seed`), then cut at the fraction
  boundaries. Reproducing `scripts/build_splits.py` with the same seed
  reproduces the identical split byte-for-byte.
- **Assertion**: the script asserts `train ∩ val = val ∩ test = train ∩ test
  = ∅` before writing any output file — this is not just documented, it is
  enforced at build time and the run would crash (loudly) rather than
  silently produce an overlapping split.

## Overlap check (measured, not assumed)

Query-grouped-by-construction split has **zero** tail-query overlap between
any two splits by the assertion above (`artifacts/splits/split_summary.json`,
`"leakage_check_disjoint_sets": true`). This is a direct, measured
improvement over the original repository's own split, which had confirmed
4.3% test-query leakage into train (`docs/DATA_PROVENANCE.md`).

## Persisted artifacts

- `artifacts/splits/train.json`, `val.json`, `test.json` — one record per
  tail query with its positive/negative head lists.
- `artifacts/splits/head_index.json` — the frozen 12,261-query candidate set.
- `artifacts/splits/ineligible_tail_queries.json` — the excluded population,
  counted, not discarded silently.
- `artifacts/splits/split_summary.json` — machine-readable counts (Section
  46 requirement).

## Counts

| Split | Tail queries |
|---|---|
| Train | 81,658 |
| Validation | 17,498 |
| Test (held out) | 17,499 |
| Ineligible (excluded, counted) | 1,293,135 |
| Head candidate index | 12,261 |
