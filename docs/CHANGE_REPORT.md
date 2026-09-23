# Change Report — Original / Repaired / Newly Implemented

## ORIGINAL CAPABILITY (unmodified, carried forward as-is)

- The core problem framing: map long-tail queries to head queries to inherit
  richer historical evidence (README, unchanged).
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
