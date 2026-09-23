# Reproducibility

## Environment

```
Python 3.12.7 (/opt/anaconda3/bin/python3)
numpy 1.26.4
pandas 2.2.2
scikit-learn 1.5.1
gensim 4.3.3
sentence-transformers 6.1.0   # installed during this effort: pip install sentence-transformers
torch (installed as a sentence-transformers dependency)
```

Hardware: Apple Silicon Mac (see `artifacts/latency_benchmark.json` ->
`hardware` for the exact `platform.platform()` string recorded at benchmark
time). All model inference in this project runs on **CPU** (`device="cpu"`)
by explicit choice — MPS (Apple GPU) inference was tried first and hit an
out-of-memory error when loading two sentence-transformer models
sequentially; CPU is slower per-request but stable for the volumes here
(tens of thousands of queries) and was fully sufficient to keep the median
latency well within an interactive budget (see `docs/BENCHMARK_REPORT.md`).

## Fixed seed

`20260922` (used everywhere: `scripts/build_splits.py` train/val/test
shuffling, `scripts/run_ladder.py` validation-sample selection, `src/metrics.py`
paired bootstrap resampling). Using the same seed with the commands below
reproduces every split and metric exactly.

## Exact commands, in order

```bash
# 1. Data audit (Section 22)
python3 scripts/audit_data.py
#   -> artifacts/data_audit.json

# 2. Leakage-safe query-grouped split (Section 10-12)
python3 scripts/build_splits.py
#   -> artifacts/splits/{train,val,test,head_index,ineligible_tail_queries,split_summary}.json

# 3. Modernization ladder: validation-only model/hyperparameter search,
#    then ONE frozen held-out val+test evaluation pass (Section 20, 37-39)
python3 scripts/run_ladder.py
#   -> artifacts/frozen_config.json
#   -> artifacts/optimization_search_log.json
#   -> artifacts/ladder_results.json

# 4. Gap / coverage / statistical-reliability report generation
#    (also selects the final architecture: highest val MRR among candidates)
python3 scripts/build_report_artifacts.py
#   -> artifacts/final_metrics.json

# 5. Sparsity correlation (Section 13)
python3 scripts/sparsity_correlation.py
#   -> artifacts/sparsity_correlation.json

# 6. Failure analysis (Section 43)
python3 scripts/failure_analysis.py
#   -> artifacts/failure_analysis.json

# 7. Confidence/coverage analysis (Section 31, secondary)
python3 scripts/coverage_analysis.py
#   -> artifacts/coverage_analysis.json

# 8. Latency benchmark (Section 41, exact protocol) + tests
python3 scripts/latency_benchmark.py
#   -> artifacts/latency_benchmark.json
python3 -m pytest tests/ -q
```

A single orchestration entry point is provided:

```bash
bash scripts/reproduce_all.sh
```

which runs steps 1-8 in order and exits non-zero if any step fails.

## What is NOT reproducible, and why

- The original Model 1 (Entity Linking) cannot be re-run: its external
  dependency (`http://dexterdemo.isti.cnr.it:8080`) is confirmed dead
  (`docs/QUERY_LOG_AUDIT.md`).
- The original claimed 78% SVM accuracy cannot be independently reproduced:
  no result artifact exists to check against, and the script was never
  actually wired to a data file present in the repository
  (`docs/QUERY_LOG_AUDIT.md`).
- Re-deriving the head/tail frequency threshold from raw counts is not
  reproducible by anyone, ever, using only this repository's contents — the
  raw log does not exist here (`docs/HEAD_TAIL_METHODOLOGY.md`).

## Artifact map

| Artifact | Produced by | Consumed by |
|---|---|---|
| `artifacts/data_audit.json` | `scripts/audit_data.py` | `docs/DATA_PROVENANCE.md`, `docs/QUERY_LOG_AUDIT.md` |
| `artifacts/splits/*.json` | `scripts/build_splits.py` | `scripts/run_ladder.py`, `docs/SPLIT_METHODOLOGY.md`, `docs/HEAD_TAIL_METHODOLOGY.md` |
| `artifacts/frozen_config.json` | `scripts/run_ladder.py` (Phase A) | `scripts/latency_benchmark.py`, `docs/BENCHMARK_REPORT.md` |
| `artifacts/optimization_search_log.json` | `scripts/run_ladder.py` (Phase A) | `docs/BENCHMARK_REPORT.md` §Optimization |
| `artifacts/ladder_results.json` | `scripts/run_ladder.py` (Phase B) | `scripts/build_report_artifacts.py`, `docs/FINAL_REPORT.md` |
| `artifacts/final_metrics.json` | `scripts/build_report_artifacts.py` | `docs/FINAL_REPORT.md`, `docs/PROJECT_EVIDENCE.md` |
| `artifacts/latency_benchmark.json` | `scripts/latency_benchmark.py` | `docs/BENCHMARK_REPORT.md`, `docs/FINAL_REPORT.md` |
