"""
Phase 12: NEW serving-layer benchmark (API round-trip overhead), separate
from the frozen algorithm benchmark in artifacts/latency_benchmark.json.
This does not re-evaluate relevance and must never be used to tune the
frozen model. Run directly (not via pytest -q) to print/save the benchmark:

    python3 -m tests.test_api_latency
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

N_WARMUP = 15
N_MEASURED = 300


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def run_benchmark():
    client = TestClient(app)
    queries = ["nfldraftcountdown", "tcf national bank", "species of trees",
               "lexmark x63 printer fax", "aol update"] * ((N_WARMUP + N_MEASURED) // 5 + 1)

    for q in queries[:N_WARMUP]:
        client.post("/api/map", json={"query": q, "k": 5})

    latencies_ms = []
    for q in queries[N_WARMUP:N_WARMUP + N_MEASURED]:
        t0 = time.time()
        r = client.post("/api/map", json={"query": q, "k": 5})
        assert r.status_code == 200
        latencies_ms.append((time.time() - t0) * 1000.0)

    result = {
        "protocol": {
            "n_warmup": N_WARMUP, "n_measured": N_MEASURED,
            "transport": "FastAPI TestClient (in-process ASGI, no real network hop)",
            "note": "Measures API-layer (request validation + handler + JSON encoding) overhead ON TOP OF the frozen core-inference benchmark in artifacts/latency_benchmark.json. Not a re-evaluation of relevance.",
        },
        "api_latency_ms": {
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "p99": percentile(latencies_ms, 99),
            "mean": statistics.mean(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
        },
    }
    return result


def test_api_latency_benchmark_runs_and_is_reasonable():
    result = run_benchmark()
    assert result["api_latency_ms"]["p95"] < 2000  # sanity bound, not a perf assertion on CI hardware


if __name__ == "__main__":
    result = run_benchmark()
    out_path = ROOT / "artifacts" / "api_latency_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
