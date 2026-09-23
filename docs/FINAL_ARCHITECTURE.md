# Final Architecture

## Pipeline

```
                         newMedium.txt (1,439,823 rows)
                          tail || head || label(0/1)
                                    |
                     scripts/audit_data.py (Section 22)
                                    |
                    scripts/build_splits.py (query-grouped,
                       leakage-asserted, seed=20260922)
                                    |
              +---------------------+----------------------+
              |                     |                       |
         train (81,658)       val (17,498)           test (17,499)
       eligible tail Qs      eligible tail Qs       eligible tail Qs
              |                     |                       |
              |          Phase A: validation-only            |
              |          search (embed model, alpha,          |
              |          beta, rerank depth)                  |
              |                     |                       |
              |            FROZEN CONFIG                     |
              |     (artifacts/frozen_config.json)            |
              |                     |                       |
              |                     +-----------------------+
              |                                              |
    train-only support prior                      Phase B: ONE held-out
   (head popularity counts,                         evaluation pass
    zero val/test leakage --                      (val AND test, same
    tests/test_pipeline.py)                         frozen config)
              |                                              |
              +----------------------+-----------------------+
                                      |
                          Query-Mapping Pipeline (per query, at inference):

  tail query text
        |
        v
  [normalize: lowercase, whitespace-collapse]        <- src/mapping.py::tokenize
        |
        v
  [Level 0/1: Jaccard lexical score vs. all 12,261 head candidates] (exact, in-memory)
        |
  [Level 2: bi-encoder embedding + cosine similarity vs. all 12,261 head candidates]
        |         (sentence-transformers/multi-qa-MiniLM-L6-cos-v1 -- selected
        |          on validation only, see docs/BENCHMARK_REPORT.md)
        v
  [Level 3: min-max normalize both score vectors, fuse: alpha*semantic + (1-alpha)*lexical]
        |
        v
  [candidate generation: top-N by fused score, N = frozen rerank_depth]
        |
        v
  [Level 4: cross-encoder rerank of the top-N candidates]
        |         (cross-encoder/ms-marco-MiniLM-L-6-v2)
        v
  [Level 5 (secondary, reported not primary): + train-only head-popularity
   log-prior, weight beta, applied to the FUSED score before candidate
   generation -- reported separately in the ladder, not stacked with Level 4
   rerank in the frozen primary pipeline, see docs/BENCHMARK_REPORT.md]
        |
        v
  ranked list of head queries (primary output, K=5 reported)
        |
        v
  [offline evaluation only: does the true positive head appear in top-K?]
        MRR / Recall@K / Hit@K -- artifacts/ladder_results.json
```

## Privacy handling

No raw user/session data enters this pipeline at any stage — see
`docs/DATA_PROVENANCE.md`. Query normalization only lowercases and collapses
whitespace; no query text is logged to any external service (both the
bi-encoder and cross-encoder run locally after download, no inference API
calls).

## Candidate index

Exact (brute-force) cosine similarity over 12,261 pre-computed head
embeddings. No ANN library (e.g. FAISS) is introduced: at this scale, exact
search on a `(12261, 384)` float32 matrix is a single fast matrix multiply
per query (dominant cost is model inference, not the retrieval step itself —
confirmed in `artifacts/latency_benchmark.json`), so ANN would add complexity
without a measurable latency benefit. See Section 29's own guidance: "If
exact search is trivial at current size, do not introduce FAISS solely for
appearance."

## Confidence / abstention

The primary frozen evaluation applies no abstention threshold (100% forced
coverage — every eligible tail query receives a ranked mapping). A secondary,
non-primary confidence-bucketed breakdown (by top-1 fusion score) is reported
in `docs/BENCHMARK_REPORT.md` to characterize where errors concentrate,
without altering the frozen primary metric.

## Output

For a given tail query: a ranked list of head queries, each of which carries
the training-period historical click/URL evidence associated with that head
query (`headkey.txt`-style evidence, in spirit). This project's evaluation
remains offline (Section 35's distinction between diagnosed opportunity and
deployed effect still holds — see `docs/FINAL_REPORT.md` "Frozen Evaluation
Confirmation"); a serving layer was added afterward, downstream of the
frozen model, purely to make the already-evaluated system demonstrable — see
below.

## Serving architecture (added after the model freeze)

```
Browser (app/static/index.html, app.js)
   |
   | fetch("/api/map"), fetch("/api/compare")
   v
FastAPI (app/main.py)
   |
   v
Frozen SparseQuery inference service (app/service.py::MappingService,
loaded once at process startup, resident in memory)
   |
   +-- lexical score      (src/mapping.py::LexicalMapper -- Jaccard,
   |                        preserved from the original repository)
   +-- semantic score      (frozen bi-encoder: multi-qa-MiniLM-L6-cos-v1,
   |                        cosine similarity vs. the 12,261-query head index)
   +-- train-only support prior
   |                       (src/mapping.py::SupportPriorMapper, counts
   |                        computed only from artifacts/splits/train.json)
   v
fusion (frozen alpha) -> support prior (frozen beta) -> ranked head queries
   |
   v
JSON response (rank, head_query, lexical_score, semantic_score,
support_score, final_score, latency_ms)
```

This is exactly the frozen Level 5 system (`docs/BENCHMARK_REPORT.md`
Section 11/14) wrapped in a FastAPI service — `app/service.py` imports
`LexicalMapper`, `FusionMapper`, and `SupportPriorMapper` directly from
`src/mapping.py` rather than reimplementing ranking logic, and reads
`alpha`/`beta`/the embedding model name from `artifacts/frozen_config.json`
rather than hardcoding copies. `tests/test_serving.py::test_api_matches_frozen_mapper`
independently reconstructs the Level 5 pipeline from the frozen artifacts
and asserts the API's output matches it exactly.

The rejected cross-encoder (Level 4) is never on this default path; it is
loaded lazily, only if `/api/compare` is called with
`include_experimental_reranker: true`, and is clearly labeled "rejected" in
that response. See `docs/CHANGE_REPORT.md` "Level 4 cross-encoder bug fix"
for a real numerical bug found and fixed in this component while building
this serving layer.

No database is used — see `docs/DEPLOYMENT.md` and
`docs/FINAL_REPORT.md` "Persistence Decision" for why this application is
intentionally stateless.
