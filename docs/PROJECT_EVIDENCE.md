# Project Evidence

Factual engineering evidence, each statement cited to the code/config/split/
artifact/command that produces it. Not resume language.

1. **1,439,823** tail→head query pairs analyzed, from `newMedium.txt`.
   (`scripts/audit_data.py` -> `artifacts/data_audit.json::newMedium.n_rows`)

2. **1,409,790** unique tail-query strings; **12,261** unique head-query
   strings. (`artifacts/data_audit.json::newMedium.n_unique_tail_queries`,
   `n_unique_head_queries`)

3. **91.73%** of unique tail queries (1,293,135 of 1,409,790) have no
   recorded ground-truth relevant head query under the original relevance
   rule and are excluded from the mapping task as ineligible, counted not
   discarded. (`artifacts/splits/split_summary.json`,
   `artifacts/splits/ineligible_tail_queries.json`)

4. Eligible tail-query evidence is thin: median **1** positive head target
   per query (p90 = 1, p99 = 3, max = 12).
   (`artifacts/data_audit.json::newMedium.n_positive_heads_per_tail_percentiles`)

5. The original repository's own train/test split had confirmed **4.3%**
   query-level leakage (199 of 4,599 unique test `query1` values also
   appeared in train). This project's own split has **0%** measured leakage,
   enforced by an assertion in `scripts/build_splits.py` and verified by
   `tests/test_pipeline.py::test_splits_are_disjoint`.
   (`artifacts/data_audit.json::original_split_out1`,
   `artifacts/splits/split_summary.json::leakage_check_disjoint_sets`)

6. Held-out test population: **17,499** eligible tail queries (15% of
   116,655 eligible), well above the adequacy threshold needed for a stable
   paired bootstrap (Section 12 gate — passed).
   (`artifacts/splits/split_summary.json`)

7. Baseline (original Jaccard similarity function, faithfully preserved and
   adapted from the original pairwise classifier into full-index retrieval
   ranking) held-out tail MRR = **0.1474**.
   (`artifacts/final_metrics.json::baseline_tail_metric`)

8. Final system (validation-selected Level 5: lexical+semantic fusion +
   train-only support prior) held-out tail MRR = **0.2661**.
   (`artifacts/final_metrics.json::final_tail_metric`)

9. Absolute tail-quality improvement = **+0.1187** MRR; relative improvement
   = **+80.5%**. (`artifacts/final_metrics.json::absolute_delta`,
   `relative_improvement_pct`)

10. Paired bootstrap 95% CI on that delta: **[+0.1150, +0.1225]**, 5,000
    resamples, seed 20260922, n=17,499 — excludes zero by a wide margin.
    (`artifacts/final_metrics.json::paired_bootstrap`)

11. Head-query self-retrieval reference ceiling MRR = **0.9981**.
    (`artifacts/final_metrics.json::head_self_retrieval_ceiling`)

12. Baseline head-tail gap = **0.8506**; final gap = **0.7319**; absolute gap
    reduction = **0.1187**; relative gap reduction = **13.95%**.
    (`artifacts/final_metrics.json::baseline_gap`, `final_gap`,
    `gap_reduction_absolute`, `gap_reduction_pct`)

13. Intervention coverage = **100%** (no abstention threshold applied in the
    primary evaluation). Secondary confidence-bucket analysis shows quality
    rises with confidence (top quartile mean MRR 0.332).
    (`artifacts/final_metrics.json::coverage`, `artifacts/coverage_analysis.json`)

14. Inference-path latency, Level 5 (the validation-selected system), run
    locally: P50 = **18.9ms**, P95 = **21.8ms**, P99 = **25.1ms**, cold start
    = **13.2s**, measured with 20 warmup + 300 measured requests at
    concurrency=1 on a fixed query set. (`artifacts/latency_benchmark.json`)

15. A validation-tested but rejected cross-encoder reranking stage (Level 4)
    was both lower-quality (test MRR 0.1889 vs. 0.1945 for fusion alone) AND
    5x slower at P95 (116.7ms vs. 21.8ms) than the selected system — a
    disclosed negative result, not omitted. (`artifacts/ladder_results.json`,
    `artifacts/latency_benchmark.json`)

16. **25/25** correctness tests pass, covering tokenization, similarity
    math, fusion behavior at boundary weights, support-prior train-only
    leakage-firewalling, ranking-metric correctness, bootstrap CI behavior,
    and split-disjointness assertions. (`python3 -m pytest tests/ -q`,
    `tests/test_pipeline.py`)

17. Zero raw AOL query-log rows (with `AnonID`/`QueryTime`) exist anywhere in
    this repository, verified by full directory inventory and full
    trace of every `import`/file-read in the original code.
    (`docs/DATA_PROVENANCE.md`, `docs/QUERY_LOG_AUDIT.md`)

18. The original repository's `Entity_linkage.py`'s external dependency
    (`http://dexterdemo.isti.cnr.it:8080`) is confirmed dead (connection
    failure, `curl` exit code, HTTP code 000, tested 2026-09-22).
    (`docs/QUERY_LOG_AUDIT.md`)

19. The original repository's `lstm_rnn.py` contains an unresolvable Python
    syntax error (`tokenizer,embedding_matrix,head_data,tail_data,,word_index =`)
    and multiple undefined-name references — it could not have executed
    successfully as committed, in Python 2 or 3.
    (`docs/QUERY_LOG_AUDIT.md`, direct inspection of `lstm_rnn.py`)
