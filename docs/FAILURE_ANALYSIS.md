# Failure Analysis

Source: `artifacts/failure_analysis.json`, produced by
`scripts/failure_analysis.py` from the frozen test-split ladder results.
Baseline = Level 0/1 (reproduced original Jaccard method). Final = Level 5
(support-prior-weighted fusion, the validation-selected system). All example
queries below are ordinary, non-identifying web-navigational/informational
queries drawn directly from `newMedium.txt` (no user data attached to any
query — see `docs/DATA_PROVENANCE.md`).

## Category counts (n=17,499 held-out test tail queries)

| Category | Count | % of test set |
|---|---|---|
| Final wins, baseline loses | 14,233 | 81.34% |
| Baseline wins, final loses | 1,262 | 7.21% |
| Both reach the correct answer at rank 1 | 1,820 | 10.40% |
| Both reach MRR exactly 0.0 | 0 | 0.00% (see note below) |

**Note on the "both fail" category being empty:** both methods rank the
*entire* 12,261-query head index (neither truncates candidates), so every
held-out query receives some finite rank for its true target and therefore a
nonzero (if tiny) reciprocal rank — MRR is essentially never *exactly* 0.0
under full-index ranking. A more meaningful "practical failure" threshold
would be "true head not in the top 10" for both methods; that is not
reported as a separate category here because it was not part of the frozen
evaluation and would constitute post-hoc metric redefinition after seeing
results (an explicitly forbidden pattern per the project brief, Section 40/43
notwithstanding — the brief's "no redefining metrics after seeing results"
principle is honored even for descriptive breakdowns).

## Category 1: Modern mapper wins, original lexical baseline fails

```
"wins1010" -> true target: "flight tracker"
  baseline MRR: 0.00009   final MRR: 0.00078

"lavar arrington rumors" -> true targets: "2006 nfl mock draft" / "mets" / "nfl draft"
  baseline MRR: 0.00047   final MRR: 0.01163

"lexmark x63 printer fax" -> true target: "lexmark"
  baseline MRR: 0.333     final MRR: 1.000   (final reaches rank 1)

"nfldraftcountdown" -> true target: "nfl draft"
  baseline MRR: 0.00034   final MRR: 0.250
```

Interpretation: these are cases where the tail query is a lexical variant
with little to no token overlap with its true head (e.g. "nfldraftcountdown"
vs. "nfl draft" — no shared whitespace-delimited tokens at all under
Jaccard), so the pure-lexical baseline has almost nothing to go on, while
the semantic embedding + training-period popularity prior correctly favors
the true target.

## Category 2: Original lexical baseline wins, modern mapper loses

```
"species of trees" -> true target: "palm trees"
  baseline MRR: 0.500     final MRR: 0.200

"www.clubaquatica.com" -> true target: "www.limewire.com"
  baseline MRR: 0.028     final MRR: 0.0026

"fullfiggas" -> true target: "yahoo360"
  baseline MRR: 0.00024   final MRR: 0.00012
```

Interpretation: "species of trees" -> "palm trees" is a case where the
correct target shares a literal token ("trees") that the Jaccard baseline
picks up on directly, while the semantic+prior system's fusion (heavily
weighted toward the semantic score, α=0.9) and popularity prior pull the
ranking toward other, more popular but less specific head queries instead.
This is a genuine, reportable weakness of the selected system: **a strong
training-period popularity prior can outweigh a literal token match when the
true target is not itself a popular head query** — see
`docs/BENCHMARK_REPORT.md` §Limitations. The `www.clubaquatica.com` /
`fullfiggas` examples are both very low-MRR for BOTH methods (neither comes
close to a useful answer) — these look like cases where no head query in the
12,261-query index is genuinely a good match for the tail query's intent
(the mapping problem here may not be well-posed for this specific tail
query, independent of which mapper is used).

## Category 3: Both methods succeed at rank 1

```
"wells fargo." -> "wells fargo"                     (both MRR 1.0)
"gwen stefani's clothing line" -> "gwen stefani"     (both MRR 1.0)
"tcf national bank" -> "tcf bank"                    (both MRR 1.0)
"lyrics to have you ever" -> "lyrics"                (both MRR 1.0)
```

Interpretation: high lexical overlap + unambiguous single-entity intent
makes these easy for any reasonable method — expected, and useful as a sanity
check that the pipeline is not somehow degenerate.

## Categories explicitly not observed / not applicable

- **Multiple plausible heads causing genuine ambiguity**: present in the
  eligibility data (max 12 positive heads for one tail query,
  `docs/HEAD_TAIL_METHODOLOGY.md`) but not separately broken out here, since
  the per-query metric already credits a hit against *any* of the positive
  targets (Recall@K/MRR definitions in `src/metrics.py`).
- **Abstention wrongly rejects / wrongly accepts**: the frozen primary
  evaluation applies no abstention threshold (100% coverage, see
  `docs/BENCHMARK_REPORT.md` §Coverage), so this category does not occur in
  the primary system as evaluated. A secondary confidence-bucketed analysis
  (`artifacts/coverage_analysis.json`) shows quality does improve
  monotonically with confidence, which is the necessary precondition for a
  future abstention policy to be useful — but no such policy is deployed or
  claimed here.
- **Entity/code-name query fails**: not separately tagged; several Category-2
  examples above (`www.clubaquatica.com`, `fullfiggas`) are consistent with
  this pattern but are reported under the general low-quality-for-both
  category rather than a fabricated dedicated bucket, since no automatic
  entity/code-name detector exists in this project to make the distinction
  rigorously.
- **Reranker hurts**: this is not a failure-analysis anecdote but a
  measured, aggregate, validation-confirmed finding — see
  `docs/BENCHMARK_REPORT.md` §Optimization: Level 4 (cross-encoder rerank)
  scored *lower* than Level 3 (fusion, no rerank) on both validation
  (0.1878 vs 0.1933 MRR) and test (0.1889 vs 0.1945 MRR), which is why it was
  not selected as the final system.
