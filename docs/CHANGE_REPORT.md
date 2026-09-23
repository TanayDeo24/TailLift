# Change Report — Original / Repaired / Newly Implemented

## ORIGINAL CAPABILITY (unmodified, carried forward as-is)

- The core problem framing: map long-tail queries to head queries to inherit
  richer historical evidence (original README content, preserved in full
  beneath a modernized summary added during the final closure pass — see
  `docs/FINAL_REPORT.md` §35 "Files modified").
- The relevance definition for a tail→head pair: same clicked URL, rank ≤5
  both sides (from `separator.py`, algorithmically preserved in this
  project's task definition — see `docs/HEAD_TAIL_METHODOLOGY.md`).
- The head-query vocabulary (`head150.txt`, 12,261 queries) and the master
  pair corpus (`newMedium.txt`, 1,439,823 rows) — used as-is, not
  regenerated (the raw log needed to regenerate them is absent).
- The Jaccard token-overlap similarity formula (`separator.py::calcJaccard`)
  — same formula, same semantics.

## REPAIRED CAPABILITY (original intent restored under modern tooling)

- **Jaccard lexical mapper**: original was a Python 2, pairwise-classifier
  formulation with a hardcoded threshold and no ranking; repaired as a
  Python 3 retrieval-ranking function (`src/mapping.py::LexicalMapper`) that
  scores the query against the *entire* head-query index rather than one
  pair at a time — same underlying similarity metric, modernized application
  (a query-mapping product surface needs a ranked list, not a binary
  yes/no). Files: `src/mapping.py`.
- **Train/test split**: original (`out (1).txt` -> `train_data.txt`/`test_data.txt`)
  had confirmed 4.3% query leakage from a contiguous line-order split.
  Repaired as a query-grouped, leakage-asserted split with 0% measured
  overlap. Files: `scripts/build_splits.py`, `docs/SPLIT_METHODOLOGY.md`.
  Measurable effect: leakage reduced from 4.3% to 0% (assertion-enforced).

## NEWLY IMPLEMENTED CAPABILITY (did not exist in the original repository)

- **Query-grouped, leakage-audited split with persisted membership and an
  automated disjointness assertion.** `scripts/build_splits.py`,
  `tests/test_pipeline.py::test_splits_are_disjoint`.
- **Eligibility accounting** (eligible vs. ineligible tail queries, with the
  ineligible population counted and persisted rather than silently
  dropped). `scripts/build_splits.py`, `docs/HEAD_TAIL_METHODOLOGY.md`.
- **Semantic bi-encoder mapper** (sentence-transformers), selected from a
  bounded 2-candidate comparison on validation only.
  `src/mapping.py::SemanticMapper`, `scripts/run_ladder.py`.
- **Lexical+semantic fusion mapper** with a tunable, validation-selected
  blend weight. `src/mapping.py::FusionMapper`.
- **Cross-encoder second-stage reranking** over the fused top-N candidates.
  `scripts/run_ladder.py::eval_rerank`.
- **Training-split-only historical-support prior**, explicitly firewalled
  from val/test labels (see `tests/test_pipeline.py::test_no_future_leakage_support_prior_only_from_train`).
  `src/mapping.py::SupportPriorMapper`.
- **A full 5-level modernization comparison ladder with a frozen, documented
  optimization procedure** (validation-only search, single frozen
  configuration, one held-out test pass). `scripts/run_ladder.py`,
  `artifacts/optimization_search_log.json`, `artifacts/frozen_config.json`.
- **Head-tail gap quantification** via a defined self-retrieval reference
  ceiling (no such concept existed in the original project — the original
  never measured a "gap," only a single pairwise classification accuracy
  number). `scripts/build_report_artifacts.py`, `docs/BENCHMARK_REPORT.md`.
- **Paired bootstrap statistical reliability** on the primary metric delta.
  `src/metrics.py::paired_bootstrap_ci`.
- **Exact-protocol latency benchmark** (warmup/measured/concurrency=1/P50-P99).
  `scripts/latency_benchmark.py` — did not exist at all originally; the
  original project has no notion of production latency.
- **25-test correctness suite** covering tokenization, similarity math,
  fusion, support-prior leakage-firewalling, ranking metrics, bootstrap CI,
  and split-integrity assertions. `tests/test_pipeline.py` — the original
  repository had zero tests.
- **This entire `docs/` provenance/audit/methodology trail** — none of these
  fourteen documents existed originally.

## Components explicitly NOT modernized (and why)

- Entity linking (Dexter API): dead external dependency, cannot be repaired
  without standing up new infrastructure the project brief does not call
  for; superseded functionally by semantic embeddings in the ladder.
- WordNet sentence similarity / raw Word2Vec cosine: algorithmically sound
  but functionally superseded by the sentence-transformer bi-encoder, which
  captures the same "semantic closeness" signal with a stronger model; kept
  out of the ladder to respect Section 21's bound on model/feature shopping
  rather than stacking every legacy feature for its own sake.
- LSTM/dense neural pairwise classifiers: original code has unresolved
  syntax errors (`lstm_rnn.py`) or depends on an absent multi-GB pretrained
  embedding file (`neuralNet.py`); not repaired, since the retrieval-ranking
  reformulation (Level 2-5 of the ladder) already supersedes what these were
  trying to achieve, and reviving broken code with no evaluable target
  serves no engineering purpose per the brief's own "don't spend excessive
  time reviving obsolete infrastructure" guidance (Section 19).

## Level 4 cross-encoder bug fix (engineering correction, post-freeze)

**Discovered**: while building the serving layer's `/api/compare` endpoint
(experimental cross-encoder comparison mode), `cross-encoder/ms-marco-MiniLM-L-6-v2`
was found to emit `NaN` for every input on this platform.

**Root cause**: isolated by forward-hooking every submodule of the model's
first transformer layer and manually re-running the matmul with freshly
cloned copies of the exact same tensors. Parameter tensors as loaded by
`transformers` 5.17.0's newer fast-weight-loading path report as finite,
NaN-free, and contiguous under direct inspection
(`torch.isnan(p).any() == False`), yet a plain `torch.matmul` using those
tensors in place produces NaN — while the identical values, cloned into
fresh contiguous storage, compute correctly (verified down to float64
precision, ruling out a genuine numerical-instability explanation). This
points to a storage-level artifact (consistent with known issues in
mmap/lazy-materialization-based fast checkpoint loading) specific to this
`transformers`/PyTorch/platform combination, not a corrupted checkpoint or
an algorithmic issue.

**Fix**: `src/mapping.py::materialize_model_(model)` clones every parameter
and buffer into fresh, contiguous storage once, immediately after
`from_pretrained(...)`. Applied at every cross-encoder load site:
`scripts/rerun_level4_corrected.py`, `scripts/latency_benchmark.py`,
`app/service.py::MappingService._rerank_experimental`.

**Regression test**: `tests/test_mapping_bugfix.py` (5 tests) — asserts
finite outputs on sanity pairs, correct relative ranking, idempotence of the
fix, and that parameter *values* are unchanged (only their storage).

**Correction applied, following the pre-agreed decision rule**:
1. Corrected Level 4 evaluated on the full validation split: MRR 0.1755
   (< frozen Level 5 val MRR 0.2648) → Level 5 stays the final architecture.
2. Corrected Level 4 then evaluated on the held-out test split exactly once
   (not touched before the rule above was satisfied): MRR 0.1749.
3. `artifacts/final_metrics.json`'s `ladder_val`/`ladder_test`/
   `architecture_selection.val_mrr_by_candidate_level` entries for
   `level4_rerank` were replaced with these corrected values (a
   `level4_correction_note` field documents why). All other fields in that
   file — `baseline_tail_metric`, `final_tail_metric`, `baseline_gap`,
   `final_gap`, `paired_bootstrap`, `coverage` — are byte-identical to their
   pre-correction values (verified by direct comparison).
4. `artifacts/ladder_results.json` (the original run) was left numerically
   unmodified but annotated with a `KNOWN_BUG_NOTICE_level4_rerank` field
   pointing to the corrected artifact, preserving the audit trail without
   letting the invalid numbers be mistaken for current results.
5. Every document referencing the old Level 4 numbers or the "reranking
   hurts" causal claim was updated: `docs/BENCHMARK_REPORT.md`,
   `docs/FINAL_REPORT.md`, `docs/PROJECT_EVIDENCE.md`,
   `docs/FAILURE_ANALYSIS.md` (this document).

**What did not change**: Level 0/1, Level 2, Level 3, Level 5 (the deployed
system), train/val/test split membership, labels, the relevance definition,
K, head/tail definitions, and every frozen Level 5 hyperparameter.
