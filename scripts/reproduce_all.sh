#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/8 data audit =="
python3 scripts/audit_data.py > /dev/null

echo "== 2/8 splits =="
python3 scripts/build_splits.py > /dev/null

echo "== 3/8 modernization ladder (long-running, ~20 min on CPU) =="
python3 scripts/run_ladder.py

echo "== 4/8 report artifacts (gap/coverage/bootstrap) =="
python3 scripts/build_report_artifacts.py > /dev/null

echo "== 5/8 sparsity correlation =="
python3 scripts/sparsity_correlation.py > /dev/null

echo "== 6/8 failure analysis =="
python3 scripts/failure_analysis.py > /dev/null

echo "== 7/8 confidence/coverage analysis =="
python3 scripts/coverage_analysis.py > /dev/null

echo "== 8/8 latency benchmark + tests =="
python3 scripts/latency_benchmark.py > /dev/null
python3 -m pytest tests/ -q

echo "Done. See artifacts/ and docs/FINAL_REPORT.md"
