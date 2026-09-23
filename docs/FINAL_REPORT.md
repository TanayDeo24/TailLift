# Final Report — Head/Tail Query Mapping Modernization

## 1. Project status

Core engineering work complete. Primary held-out evaluation run once,
without post-test retuning. Several behavioral-diagnostic capabilities
(sessionization, first-click position, reformulation, abandonment) are
**BLOCKED** by a specific, documented data-availability constraint (no raw
AOL log, no timestamps, no user IDs anywhere in this repository) rather than
completed — see §38/§39 below and `docs/SESSIONIZATION.md`. Everything the
available data can honestly support has been built, evaluated, optimized
once (one validation search round), and documented.

## 2. Original repository state

2017 Python 2 project, 32 files, not version-controlled (flat snapshot).
Three loosely-connected "models" claimed in `README.md` (entity linking,
SVM ensemble, neural net) that were never actually wired into one pipeline;
one script (`lstm_rnn.py`) has an unresolvable syntax error; one external
dependency (Dexter entity-linking API) is dead. Full detail:
`docs/PROJECT_STATE_INITIAL.md`, `docs/QUERY_LOG_AUDIT.md`.

## 3. Dataset

- 1,439,823 tail→head query pairs (`newMedium.txt`), binary relevance label.
- 1,409,790 unique tail queries; 12,261 unique head queries.
- No raw log, no timestamps, no user IDs, no impression lists survive
  anywhere in this repository — only already-derived pair files remain.
- Click/rank availability: consumed into the binary label at some prior
  point, not retained per-row in any surviving file.

## 4. Data quality / privacy

No raw user identifiers exist in any file in this repository (verified by
full inventory). The controversial raw AOL log release itself was
deliberately not re-acquired, even though doing so would unlock session
analysis, because of its well-documented 2006 re-identification harm — see
`docs/DATA_PROVENANCE.md` §Privacy. All example queries cited anywhere in
`docs/` are ordinary web-navigational/informational strings drawn from
already-present, already-anonymized files.

## 5. Split methodology

Query-grouped, 70/15/15 (train/val/test), seed 20260922, zero measured
cross-split leakage (assertion-enforced + test-covered). The original
repository's own split had confirmed 4.3% leakage. Full detail:
`docs/SPLIT_METHODOLOGY.md`.

## 6. Sessionization

Not performed — no timestamp/user data exists anywhere in this repository,
confirmed by tracing every field the original code ever read. Documented,
not fabricated. `docs/SESSIONIZATION.md`.

## 7. Frequency definition

Inherited from the original repository (`HEAD_LIMIT=150`, `TAIL_LIMIT<5`),
not independently re-derived — the raw occurrence counts needed to re-derive
it from scratch do not exist in this repository. `docs/HEAD_TAIL_METHODOLOGY.md`.

## 8. Head/tail definition

Head = `head150.txt` (12,261 queries). Tail = `newMedium.txt`'s tail column
(1,409,790 unique queries). Eligible tail (has ≥1 ground-truth relevant
head) = 116,655.

## 9. Head/tail counts

| | Count |
|---|---|
| Head queries | 12,261 |
| Unique tail queries | 1,409,790 |
| Eligible tail queries | 116,655 (8.27% of unique tail queries) |
| Ineligible tail queries | 1,293,135 (91.73%) |
| "Traffic share" (row-count share, not true occurrence-weighted traffic — unavailable) | tail queries generate 1,439,823 of all pair rows against only 12,261 distinct heads |

## 10. Data-sparsity evidence

Eligible tail queries have a median of **1** ground-truth relevant head
(p90=1, p99=3, max=12) — a categorically thinner evidence base than the
150+-occurrence class that defines a head query. See
`docs/SEARCH_QUALITY_DIAGNOSIS.md`.

## 11. Query-log insights

1. The tail vocabulary outnumbers the head vocabulary 115:1.
2. Only 8.27% of unique tail queries have any recorded relevant head at all
   under the original relevance rule — the rest are, by this definition,
   currently unmappable.
3. The only available evidence-density proxy (pair degree) does **not**
   meaningfully correlate with mapping quality (Spearman ρ=-0.026) — a
   genuine null result, not spun into a positive claim.
4. A strong training-period popularity prior materially improves aggregate
   mapping quality (+80.5% relative MRR) but can override a correct literal
   lexical match for less-popular true targets — a real, disclosed tradeoff.

## 12. Primary task

Task A: tail query → relevant head query (query-to-query relevance
mapping), evaluated by retrieval over the fixed 12,261-query head index.

## 13. Primary K

K=5 (secondary K=1,10 also reported). Frozen before test-split evaluation.

## 14. Primary metric

MRR (mean reciprocal rank) — justified by the near-single-relevant-target
structure of the eligible tail-query population.

## 15. Original baselines

Original Jaccard similarity (`separator.py`) faithfully preserved as a
scoring function and adapted to full-index retrieval ranking (the original
was a pairwise binary classifier; this project reuses the identical
similarity formula to rank against all 12,261 head candidates): test MRR =
0.1474. The claimed 78% SVM-ensemble accuracy could not be verified (no
result artifact exists, entity-linking API dead, script never wired to a
present data file).

## 16. Model comparison ladder

| Level | System | Val MRR | Test MRR |
|---|---|---|---|
| 0/1 | Original Jaccard similarity, preserved and adapted to full-index retrieval | 0.1447 | 0.1474 |
| 2 | Semantic bi-encoder | 0.1894 | 0.1897 |
| 3 | Lexical+semantic fusion | 0.1933 | 0.1945 |
| 4 | Fusion + cross-encoder rerank (corrected — see note below) | 0.1755 | 0.1749 |
| **5** | **Fusion + train-only support prior (SELECTED)** | **0.2648** | **0.2661** |

*Level 4 note:* the cross-encoder originally used for reranking silently
emitted NaN for every input on this platform (a real bug, root-caused and
fixed while building the serving layer — see §31 and
`docs/CHANGE_REPORT.md`). The values above are the corrected re-evaluation
(`artifacts/level4_corrected.json`); the original (invalid) 0.1878/0.1889
figures are no longer cited anywhere in this report.

## 17. Baseline head quality

**Not independently measurable as a production metric** — no raw head-only
click logs survive to compute true production head-query search quality.
Used instead, and labeled precisely as such throughout this report: a
defined **head-query self-retrieval reference ceiling** — a head query
matched against the same head-query index it was drawn from (near-certain
self-identity retrieval), MRR = 0.9981. Every gap figure in this report
(§19, §23, §24) is computed against this explicit reference ceiling, not
against an independently measured production number.

## 18. Baseline tail quality

MRR = 0.1474 (Level 0/1, reproduced original method), test split, n=17,499.

## 19. Baseline head-tail gap

0.9981 − 0.1474 = **0.8506**.

## 20. Final intervention architecture

Lexical Jaccard + semantic bi-encoder (`multi-qa-MiniLM-L6-cos-v1`) cosine
similarity, min-max normalized and fused (α=0.9 toward semantic), plus a
training-split-only head-popularity log-prior (β=0.25), ranked over the full
12,261-query head index with no truncation and no reranker. Full diagram:
`docs/FINAL_ARCHITECTURE.md`.

## 21. Optimization procedure

One validation-only search round on a fixed 3,000-query sample of the val
split: 2 embedding models × 4 fusion weights, then 3 support-prior weights ×
2 rerank depths with the winning model/weight held fixed. Config frozen
before Phase B (one full val+test pass, no retuning after test numbers were
seen). `docs/BENCHMARK_REPORT.md` §15.

## 22. Final held-out tail results

| | Value |
|---|---|
| Baseline tail MRR | 0.1474 |
| Final tail MRR | 0.2661 |
| Absolute delta | +0.1187 |
| Relative improvement | +80.5% |
| Paired bootstrap 95% CI (n=17,499, 5,000 resamples, seed 20260922) | [+0.1150, +0.1225] |

## 23. Final head-tail gap

0.9981 − 0.2661 = **0.7319**.

## 24. Gap reduction

Absolute: 0.8506 − 0.7319 = **0.1187**. Relative: **13.95%**.

## 25. Coverage

100% (no abstention threshold in the primary evaluation). Secondary
confidence-bucketed analysis: quality rises monotonically with confidence;
an 0.9-threshold policy would retain 82.2% coverage at mean MRR 0.298 (not
deployed, reported for future reference only).

## 26. Behavioral diagnosis

**Not available.** No session, first-click-position, reformulation, or
abandonment data exists anywhere in this repository (no timestamps, no user
IDs, no full impression lists survive). This is disclosed as a firm
limitation, not worked around. `docs/SESSIONIZATION.md`.

## 27. Offline intervention proxies

The primary MRR/Recall@K/Hit@K results (§22) are themselves this project's
offline intervention proxy, since Task A (query-to-query mapping, not
document/click recovery) was selected as primary. No claim of deployed,
in-production user-behavior change is made anywhere (Section 35 compliance).

## 28. Performance

Inference-path benchmark (Level 5, run locally, not yet deployed at the time
of this measurement): P50=18.9ms, P95=21.8ms, P99=25.1ms, cold start=13.2s,
error rate 0.0%. Protocol: 20 warmup + 300 measured requests, concurrency=1,
fixed query cycling, CPU (Apple Silicon, `device="cpu"`).
`artifacts/latency_benchmark.json`. See §"API-Level Latency" below for the
separate serving-layer (FastAPI) round-trip benchmark added in the serving
phase of this project.

## 29. Statistical stability

Paired bootstrap CI excludes zero by a wide margin (§22). n=17,499 is far
beyond the minimum for a stable percentile bootstrap on a bounded metric —
no small-sample caveat needed.

## 30. Failure analysis

Final system beats the reproduced baseline on 81.3% of held-out tail
queries, loses on 7.2%, both succeed at rank 1 on 10.4%. A disclosed
weakness: the training-period popularity prior can override a correct
literal lexical match for less-popular true targets. Full detail with
sanitized examples: `docs/FAILURE_ANALYSIS.md`.

## 31. Negative results

1. **Cross-encoder reranking underperforms fusion alone** — validation MRR
   0.1755 vs. 0.1933, test MRR 0.1749 vs. 0.1945. This finding was
   originally measured incorrectly: a bug in the cross-encoder's inference
   path (see "Level 4 cross-encoder bug fix" below) caused it to silently
   emit NaN for every input, which — because `numpy.argsort` on an all-NaN
   array preserves input order — made the original "rerank" step a no-op.
   The bug was found, root-caused, and fixed while building the serving
   layer; Level 4 was then correctly re-evaluated on validation (once) and,
   since it still did not beat Level 5, on the held-out test split (once).
   The corrected numbers above are the genuine finding: even a *working*
   off-the-shelf passage-ranking cross-encoder does not transfer cleanly to
   query-query mapping without task-specific fine-tuning. Level 5 remains
   the correctly-selected final architecture throughout — this correction
   changed a secondary ladder entry and its causal explanation, not the
   deployed system or any headline metric (baseline/final MRR, gap,
   coverage, and paired CI are all Level 0/1 and Level 5 numbers, untouched
   by this bug or its fix).
2. The available evidence-density proxy does not meaningfully correlate
   with mapping quality.
3. The support-prior weighting can misfire for high-lexical-overlap,
   low-popularity targets.

None of these were smoothed over to make the headline number look cleaner.

## Level 4 cross-encoder bug fix (engineering correction, added after the model freeze)

**What was found**: while building the serving layer's experimental
reranker comparison endpoint, `cross-encoder/ms-marco-MiniLM-L-6-v2` was
found to emit NaN for every input on this platform, verified at the raw
`transformers.AutoModelForSequenceClassification` level (not a
sentence-transformers-specific issue).

**Root cause**: isolated via forward hooks and manual tensor inspection.
The model's parameter tensors, as loaded by `transformers` 5.17.0's newer
fast-weight-loading path, report as finite, NaN-free, and contiguous under
direct inspection (`torch.isnan(p).any()` is `False`) — but a plain
`torch.matmul` using those exact tensors produces NaN, while the identical
values copied into freshly-allocated storage (`.clone().contiguous()`)
compute correctly. This points to a storage-level artifact (likely
mmap/lazy-materialization related) specific to this `transformers` version
on this PyTorch 2.7.0 / Apple Silicon CPU combination, not a genuine
numerical instability or a corrupted checkpoint.

**Fix**: `src/mapping.py::materialize_model_(model)` — clones every
parameter and buffer into fresh, contiguous storage immediately after
`from_pretrained(...)`, once, at load time. Applied everywhere the
cross-encoder is loaded: `scripts/rerun_level4_corrected.py`,
`scripts/latency_benchmark.py`, `app/service.py`.

**Validation**: `tests/test_mapping_bugfix.py` (5 tests) — asserts finite
outputs, correct relative ranking on sanity pairs, idempotence, and that
the fix does not alter parameter values (only their storage).

**Correction result**: Level 4 val MRR 0.1755 (< Level 5's frozen 0.2648,
so per the pre-agreed decision rule, Level 5 stayed frozen and Level 4 was
evaluated on test exactly once) → Level 4 test MRR 0.1749.

**Confirmation — unaffected by this bug or its fix**: Level 0/1 baseline,
Level 2, Level 3, Level 5 (the deployed system), train/val/test split
membership, labels, the relevance definition, K, head/tail definitions, and
every frozen Level 5 hyperparameter (α=0.9, β=0.25, embed model). Verified:
`artifacts/final_metrics.json::baseline_tail_metric` and `final_tail_metric`
are byte-identical to their pre-correction values.

## 32. Test status

**51/51** tests pass (`python3 -m pytest tests/ -q`): 25 in
`tests/test_pipeline.py` (tokenization, Jaccard math, fusion at boundary
weights, support-prior train-only leakage-firewalling, ranking metrics,
paired bootstrap CI, split-disjointness/eligibility/head-index-size
assertions), 20 in `tests/test_serving.py` (serving layer, including the
critical API-matches-frozen-mapper test), 5 in `tests/test_mapping_bugfix.py`
(Level 4 cross-encoder NaN-bug regression guard), 1 in
`tests/test_api_latency.py` (API-level benchmark sanity check).

## 33. Reproduction commands

```bash
pip install -r requirements.txt
bash scripts/reproduce_all.sh

# serving layer (added after the model freeze)
pip install -r requirements-serving.txt
uvicorn app.main:app --reload --port 8000
python3 -m pytest tests/test_serving.py tests/test_mapping_bugfix.py -q
```

See `docs/REPRODUCIBILITY.md` for the full step-by-step breakdown, seeds,
and artifact map, and `docs/DEPLOYMENT.md` for Docker/Render.

## 34. Files created

**Modeling/evaluation phase**: `docs/*.md` (14 documents), `scripts/audit_data.py`,
`scripts/build_splits.py`, `scripts/run_ladder.py`,
`scripts/build_report_artifacts.py`, `scripts/sparsity_correlation.py`,
`scripts/failure_analysis.py`, `scripts/coverage_analysis.py`,
`scripts/latency_benchmark.py`, `scripts/reproduce_all.sh`,
`src/mapping.py`, `src/metrics.py`, `tests/test_pipeline.py`,
`requirements.txt`, `artifacts/*.json` (machine-readable result files).

**Serving/deployment phase**: `app/main.py`, `app/service.py`,
`app/schemas.py`, `app/static/index.html`, `app/static/app.js`,
`app/static/styles.css`, `tests/test_serving.py`,
`tests/test_mapping_bugfix.py`, `tests/test_api_latency.py`,
`scripts/rerun_level4_corrected.py`, `Dockerfile`, `.dockerignore`,
`render.yaml`, `requirements-serving.txt`, `docs/DEPLOYMENT.md`,
`.gitignore`, `artifacts/level4_corrected.json`,
`artifacts/api_latency_benchmark.json`,
`artifacts/head_embeddings_<hash>.npy`.

## 35. Files modified

**31 of the 32 original repository files remain byte-identical to the
initial snapshot** — verifiable via
`git diff 946b9f9 -- <original files except README.md>` (empty).
**`README.md` is the one deliberate exception**: its top section was
replaced with a short, modern summary (problem, architecture, frozen
results, local demo/API commands, link to `docs/FINAL_REPORT.md`) during
this final closure pass, per an explicit instruction to modernize it. The
original 2017 README content was **not deleted** — it is preserved in full
beneath a clearly labeled divider, so the historical record remains intact
and readable; `docs/QUERY_LOG_AUDIT.md` still traces exactly which of its
claims hold up.

Files **created by this project itself** in the modeling phase and revised
in the serving phase: `src/mapping.py` (added `materialize_model_`, the
Level 4 bugfix — see `docs/CHANGE_REPORT.md`), `scripts/latency_benchmark.py`
(same fix applied), `artifacts/final_metrics.json` and
`artifacts/ladder_results.json` (Level 4 entries corrected/annotated, all
other fields unchanged — see `docs/CHANGE_REPORT.md`), `app/static/{index.html,styles.css,app.js}`
(visual redesign, no API/model change — see the redesign commit), and
several `docs/*.md` files (precision-corrected per the documentation closure
passes, updated with the Level 4 bug-fix trace, and updated with the
product-positioning/deployment-status language from this final closure
pass).

## 36. Original vs. repaired vs. new

Full breakdown: `docs/CHANGE_REPORT.md`.

## 37. Project evidence

Full cited list: `docs/PROJECT_EVIDENCE.md`.

## 38. Limitations

1. **No raw AOL log**: sessionization, first-click-position, reformulation,
   and abandonment analysis are all blocked by this, not merely
   deprioritized. Deliberately not fixed by re-acquiring the raw, privacy-
   controversial release.
2. **Head/tail threshold is inherited, not re-derived**: raw occurrence
   counts needed for an independent Section-11-style derivation do not
   exist in this repository.
3. **"Head quality" is a reference ceiling, not an independent production
   measurement**: no raw head-only click logs survive to measure head-query
   search quality directly; the gap-analysis ceiling is a defined
   self-retrieval reference, disclosed as such throughout.
4. **The evidence-density proxy used for sparsity correlation is weak**:
   most eligible tail queries share the same minimum value, limiting what
   that specific analysis can show (a genuine null result, not hidden).
5. **Cross-encoder reranking was evaluated but rejected** on validation
   evidence — included in the ladder and in latency benchmarking for
   completeness, not deployed.
6. **Original SVM/entity-linking/LSTM methods could not be fully
   reproduced**: dead external API, absent large pretrained files, and one
   unresolvable syntax error in the original code.
7. **CPU-only inference**: MPS (Apple GPU) hit an out-of-memory error during
   development; CPU was used throughout for stability. Latency numbers
   reflect CPU inference, not GPU-accelerated serving.

## 39. JD coverage matrix

| Capability | Status | Evidence |
|---|---|---|
| Query-log analysis | PARTIALLY DEMONSTRATED | Analyzes query-log-derived tail→head relevance data and inherited head/tail artifacts (`scripts/audit_data.py`, `artifacts/data_audit.json`); raw query-event logs, timestamps, users, and impressions are unavailable. |
| Deriving insights from query logs | PARTIALLY DEMONSTRATED | Insights (`docs/SEARCH_QUALITY_DIAGNOSIS.md`) are derived from already-derived tail→head relevance pairs, not raw query-event logs — same underlying data-availability limit as above. |
| Identifying search-quality improvement opportunities | DIRECTLY DEMONSTRATED | §11 above; led directly to the Level 5 intervention |
| Data sparsity | DIRECTLY DEMONSTRATED | `docs/PROJECT_EVIDENCE.md` items 3-4, `artifacts/sparsity_correlation.json` |
| Head-query analysis | DIRECTLY DEMONSTRATED | `docs/HEAD_TAIL_METHODOLOGY.md`, head self-retrieval ceiling |
| Tail-query analysis | DIRECTLY DEMONSTRATED | eligibility accounting, per-query test metrics |
| Long-tail search | DIRECTLY DEMONSTRATED | 115:1 tail:head vocabulary ratio, §9 |
| Click behavior | PARTIALLY DEMONSTRATED | relevance labels derive from historical clicks, but no raw click stream (position, sequence) survives — see Limitation 1 |
| Search-quality segmentation by frequency/support | DIRECTLY DEMONSTRATED | head vs. tail MRR comparison, gap analysis |
| Semantic query understanding | DIRECTLY DEMONSTRATED | Level 2-5 sentence-transformer bi-encoder mapper |
| Sparse-query intervention | DIRECTLY DEMONSTRATED | Level 5 system, +80.5% relative MRR on held-out tail |
| Query mapping | DIRECTLY DEMONSTRATED | the full tail-query → ranked head-query system, §16, §22 |
| Reformulation analysis | NOT DEMONSTRATED / BLOCKED BY DATA | no session/timestamp sequence exists to identify consecutive same-need queries |
| Head-tail search-quality gap | PARTIALLY DEMONSTRATED | Tail-quality gap is quantified against an explicitly defined head-query self-retrieval reference ceiling (§17, §19, §23, §24); true production head-query quality cannot be measured because raw head-query interaction logs are unavailable. The 13.95% gap-reduction calculation is retained and mathematically valid against the stated ceiling — it is not an independently measured production head-vs-tail quality reduction. |
| Measurable search-quality improvement | DIRECTLY DEMONSTRATED | paired bootstrap CI excluding zero, §22 |
| Intervention coverage | DIRECTLY DEMONSTRATED | `artifacts/coverage_analysis.json`, §25 |
| Abstention/confidence handling | PARTIALLY DEMONSTRATED | confidence signal validated (quality rises with confidence) but no abstention policy deployed in the primary system |
| First-click-position analysis | NOT DEMONSTRATED / BLOCKED BY DATA | no rank data survives per-row; documented limitation |
| No-click analysis | NOT DEMONSTRATED / BLOCKED BY DATA | no impression lists exist |
| Session reformulation / abandonment analysis | NOT DEMONSTRATED / BLOCKED BY DATA | no session/timestamp data exists anywhere in this repository |
| Offline click/relevance recovery | PARTIALLY DEMONSTRATED | offline query-mapping recovery fully demonstrated; document/click-level recovery (Task B) not built, Task A chosen as primary instead |
| Production-aware / serving-path latency | DIRECTLY DEMONSTRATED | exact-protocol core-inference benchmark (§28) plus a real FastAPI serving-layer benchmark (§"API-Level Latency") |
| Machine learning | DIRECTLY DEMONSTRATED | bi-encoder, cross-encoder (evaluated and correctly rejected), validation-driven model/hyperparameter selection |

## Serving Architecture (added after the model freeze)

A FastAPI application (`app/`) wraps the frozen Level 5 system for live
demonstration. It reuses `src/mapping.py`'s `LexicalMapper`, `FusionMapper`,
and `SupportPriorMapper` directly and reads `alpha`/`beta`/the embedding
model name from `artifacts/frozen_config.json` — no ranking logic is
duplicated or reimplemented. Full diagram: `docs/FINAL_ARCHITECTURE.md`
§"Serving architecture." The train-only support prior is built exclusively
from `artifacts/splits/train.json`, the same file the frozen evaluation
used — never val/test. The rejected cross-encoder (Level 4) is not on the
default serving path; it loads lazily only behind an explicit opt-in flag.

## Demo UI

A single-page vanilla HTML/CSS/JS interface (`app/static/`), served by the
same FastAPI process (no separate frontend deploy). Features: a query input
with 5 clickable sanitized example queries, ranked mapping results with
per-candidate lexical/semantic/support/final scores, a three-way comparison
view (lexical baseline vs. semantic-only vs. final system), a system-info
panel pulling live values from `/api/metadata` (explicitly labeled "Offline
held-out evaluation"), and a collapsible methodology section. Loading,
error, and empty-input states are handled; output is escaped
(`escapeHtml()` in `app.js`) before insertion into the DOM.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness, model-loaded flag, head-index size, frozen-config version hash |
| `POST /api/map` | the canonical frozen Level 5 mapping (query, k=1-10) → ranked head queries with per-candidate scores |
| `POST /api/compare` | lexical baseline vs. semantic-only vs. final system, side by side; the rejected cross-encoder is available only via `include_experimental_reranker: true` |
| `GET /api/metadata` | factual project metadata (head-index size, frozen model/alpha/beta, held-out test size, baseline/final MRR, relative improvement, local P95 latency), explicitly labeled as offline evaluation results |

Input validation: non-empty/non-whitespace query (max 300 chars), K bounded
to 1-10 (Pydantic `Field` constraints in `app/schemas.py`, enforced as HTTP
422 on violation). No confidence value is presented as a calibrated
probability — scores are labeled precisely (`lexical_score`,
`semantic_score`, `support_score`, `final_score`).

## Deployment

| Layer | Status |
|---|---|
| Local serving (FastAPI + frozen Level 5 pipeline) | **VERIFIED** — a real `uvicorn` process was started and hit over actual HTTP, not just an in-process test client |
| Local interactive demo UI | **VERIFIED** — driven end-to-end by a real headless-browser session against the live API, zero console errors |
| Docker configuration | **PREPARED** — `Dockerfile`, `.dockerignore`; the underlying service-construction path the build exercises was verified directly (Docker itself is not installed in the environment this project was built in, so `docker build` could not be executed here — see `docs/DEPLOYMENT.md` §2) |
| Render configuration | **PREPARED** — `render.yaml`, ready to apply |
| Public standalone deployment | **INTENTIONALLY NOT PERFORMED** |

**Why public deployment was not pursued — a product decision, not a
blocker.** SparseQuery is designed as an upstream relevance component that
sits between an incoming query and a downstream retrieval/ranking system
(see "Intended System Role" below), not as a standalone consumer search
product. Standing up a public website would expose only an internal
query-rewriting component in isolation — not the complete search experience
it is meant to feed into — so it was deliberately not pursued as part of
this closure pass. This is unrelated to, and not blocked by, credential or
environment availability. The Docker and Render configuration remain
prepared and unused-but-ready for whenever SparseQuery is integrated behind
a real downstream system (see `docs/DEPLOYMENT.md` for the exact steps, kept
as a future deployment guide). No live URL is stated anywhere in this
project because none exists and none is currently planned.

## Serving Tests

20 new tests in `tests/test_serving.py` (service init, health, metadata,
static UI route, `/api/map` validation — empty/whitespace/K-bounds,
deterministic output, all score fields present, `/api/compare`'s three
default systems plus the opt-in experimental reranker) plus 5 in
`tests/test_mapping_bugfix.py` (the Level 4 cross-encoder NaN-bug regression
guard) plus 1 in `tests/test_api_latency.py` (the API-level benchmark sanity
check). The single most important serving test,
`test_api_matches_frozen_mapper`, independently reconstructs the frozen
Level 5 pipeline from `artifacts/frozen_config.json` and
`artifacts/splits/train.json` and asserts the live API's ranked output and
scores are identical — this is what guarantees the demo layer cannot
silently drift from the frozen, evaluated system. All 51 tests (25
pre-existing in `tests/test_pipeline.py` + 26 new across the three files
above) pass.

## API-Level Latency

A new, separate benchmark (`tests/test_api_latency.py`), distinct from the
frozen core-inference benchmark (`artifacts/latency_benchmark.json`).
Measures FastAPI request validation + handler + JSON-encoding overhead on
top of core inference, 15 warmup + 300 measured requests, in-process ASGI
transport: **P50 21.8ms / P95 29.8ms / P99 33.3ms**
(`artifacts/api_latency_benchmark.json`) — versus the core-inference-only
P50 18.9ms / P95 21.8ms / P99 25.1ms. This is a serving-overhead
measurement, not a re-evaluation of relevance, and was never used to tune
the frozen model.

## Persistence Decision

**No database was added.** This application has no legitimate persistence
requirement at this stage: it serves a frozen, pre-computed model with no
user accounts, no feedback collection, no query-history feature, and no
usage analytics currently in scope. Adding Neon/PostgreSQL (or any other
database) here would be unjustified complexity with no corresponding
capability. If a real requirement emerges later (explicit user feedback on
mappings, demo query history, model-version experiment logging), it should
be added deliberately at that point, not preemptively. The service remains
fully stateless.

FOUNDATION: git/provenance recorded ✅ · original architecture traced ✅ ·
data provenance documented ✅ · schema audited ✅ · privacy treatment
documented ✅

SESSION/BEHAVIOR: sessions **BLOCKED** (no timestamp/user data exists,
`docs/SESSIONIZATION.md`) · first-click/no-click/reformulation/abandonment
all **BLOCKED**, same root cause

SPLITS: leakage audit ✅ · methodology fixed ✅ · persisted ✅ · future
leakage prevented ✅ (test-covered) · overlap checks ✅ (0% measured)

HEAD/TAIL: frequency definition chosen ✅ (inherited, documented why) ·
distribution analyzed ✅ (from surviving derived data) · threshold
re-derivation from raw counts **BLOCKED** (raw counts absent) · definitions
frozen ✅ · memberships persisted ✅ · sample adequacy checked ✅ (17,499
test, passed)

SPARSITY: behavioral/support sparsity quantified ✅ (with the honest
weak-proxy caveat) · frequency-quality relationship analyzed ✅ (null
result, reported) · gap measured ✅ · insights produced ✅

RELEVANCE: click evidence audited ✅ · click bias **PARTIALLY** documented
(position bias could not be re-audited, per-row rank not retained) ·
primary task/K/metric chosen ✅

BASELINES: original baselines reproduced ✅ (Jaccard; SVM/entity/LSTM
**BLOCKED**, documented why) · best baseline identified ✅ · results stored ✅

MODERNIZATION: lexical/semantic/fusion/reranker/support-aware all evaluated
✅ · model shopping bounded ✅ (2 embed models, 1 cross-encoder) · final
architecture selected on validation ✅

OPTIMIZATION: search space defined ✅ · validation-only tuning ✅ · one round
(no refinement needed) ✅ · config frozen ✅ · no test tuning occurred ✅

INTERVENTION: candidate generation ✅ · mapping ✅ · confidence/abstention
evaluated ✅ (secondary, not deployed) · coverage computed ✅ · future
leakage prevented ✅ (test-covered)

FINAL EVALUATION: n adequate ✅ · baseline/final tail metric ✅ · absolute
delta ✅ · relative improvement ✅ · paired CI ✅

GAP: baseline/final head & tail metrics ✅ (head metric = disclosed ceiling)
· baseline/final gap ✅ · absolute/percentage reduction ✅

BEHAVIOR: historical vs. intervention distinction maintained ✅ (N/A —
no historical behavioral data to conflate) · no false causal claims ✅ ·
offline proxies labeled correctly ✅

LATENCY: models loaded before benchmark ✅ · ≥15 warmups ✅ (20) · ≥300
measured ✅ · concurrency=1 ✅ · timing boundary documented ✅ · P50/P95/P99 ✅
· hardware recorded ✅

QUALITY: tests added ✅ (51, including 20 serving + 5 bugfix regression
added post-freeze) · all pass ✅ · failure analysis complete ✅ · negative
results preserved ✅ (including the corrected Level 4 finding)

REPRODUCIBILITY: setup documented ✅ · preprocessing documented ✅ · splits
reproducible ✅ · versions recorded ✅ · seeds recorded ✅ · exact commands ✅
· structured artifacts saved ✅

DOCUMENTATION: all 14 required documents present ✅ (SESSIONIZATION.md
included, documenting exclusion rather than being skipped)

JD: all 15+ capabilities audited ✅ (§39)

## 41. Final freeze verdict (modeling/evaluation phase)

**READY TO FREEZE**, with the limitations in §38 and the BLOCKED items in
§40 disclosed as firm, evidenced data-availability constraints rather than
unfinished work. Every metric in this report traces to a committed,
re-runnable artifact (`artifacts/*.json`) produced by a committed script
(`scripts/*.py`), with a single frozen configuration evaluated once on
held-out data, a validation-only optimization procedure, an enforced
leakage-free split, a passing test suite, and negative results reported
alongside the positive ones (including a real bug found and fixed in the
rejected Level 4 component after this section was first written — see
"Level 4 cross-encoder bug fix" above; the fix did not change which
architecture is frozen). Nothing here depends on data this repository does
not actually contain.

## Frozen Evaluation Confirmation (serving/deployment phase)

Explicitly confirmed, checked by direct artifact comparison, not assertion:

- **No relevance metric changed.** `baseline_tail_metric` (0.1474331...),
  `final_tail_metric` (0.2661195...), `baseline_gap`, `final_gap`, and
  `paired_bootstrap` in `artifacts/final_metrics.json` are byte-identical
  before and after the serving-phase work.
- **No test split changed.** `artifacts/splits/{train,val,test}.json` were
  not touched; `tests/test_pipeline.py::test_splits_are_disjoint` still
  passes.
- **No model was retuned.** `artifacts/frozen_config.json` (embed model,
  α=0.9, β=0.25, K=5) was not modified; `app/service.py` reads it rather
  than hardcoding copies (`tests/test_serving.py::test_service_uses_frozen_config_not_hardcoded_copies`).
- **No frozen evaluation artifact was altered to improve results.** The one
  correction made — Level 4's numbers — went *down* on the ladder (from an
  invalid, bug-produced 0.1878/0.1889 to a genuine, lower 0.1755/0.1749),
  and the corrected numbers made the frozen Level 5 architecture look
  *more* clearly correct, not less — this was a bug fix restoring accuracy,
  not a tuning pass chasing a better number. The correction is fully
  audited: original bug documented, root cause traced, fix applied and
  regression-tested, validation-first decision rule followed exactly (val
  checked before test was touched), and every document that cited the
  invalid numbers was updated. See `docs/CHANGE_REPORT.md` "Level 4
  cross-encoder bug fix" for the complete trace.

## Intended System Role

SparseQuery is a long-tail query-mapping **component** for search-relevance
systems — it maps sparsely-supported tail queries to better-supported head
queries using lexical, semantic, and train-only historical-support signals.
It is designed to sit **upstream of** a retrieval/ranking system, not to
operate as a complete, standalone search engine:

```
incoming query
    |
    v
sparse / long-tail query handling
    |
    v
SparseQuery mapping   <-- this project
    |
    v
better-supported head-query representation
    |
    v
retrieval / ranking system
    |
    v
search results
```

The FastAPI service and demo UI built in this closure pass demonstrate that
the frozen model can be served through a real API with production-shaped
engineering (health checks, containerization, tested latency) — they are a
**local interactive engineering/demo interface**, not the project's primary
deliverable, and not a claim of search-relevance quality on their own (the
held-out evaluation in this report is that evidence). A future integration
could connect SparseQuery's mapped output to a separate downstream
retrieval/ranking system (for example, a hybrid search project) to form an
end-to-end pipeline — that integration is explicitly **not implemented
here**.

## Serving/Deployment Freeze Verdict

The serving layer, demo UI, API, and containerization are complete, tested
(51/51 passing, including the API-matches-frozen-mapper guarantee), and
documented (`docs/DEPLOYMENT.md`) — see the Deployment status table above.
Public standalone deployment was intentionally not pursued as an
architectural/product decision (see "Intended System Role"), not because of
any missing credential or unresolved engineering issue. This project's
status is **ENGINEERING COMPLETE, SERVING-READY**.

## Final Project Status

- **Core model**: frozen (Level 5 — lexical + semantic fusion + train-only
  support prior), unchanged since freeze except for the documented,
  audited Level 4 comparison-ladder bug correction (§"Level 4 cross-encoder
  bug fix"), which altered no Level 0/1/2/3/5 metric.
- **Evaluation**: held-out, leakage-safe, statistically validated (§16-24,
  §33-34 above).
- **Serving**: built, tested, and verified against a real running server
  and a real browser session (§"Serving Architecture" through
  §"API-Level Latency" above).
- **Deployment**: Docker and Render configuration prepared; public
  deployment intentionally not pursued — an architectural decision, not a
  blocker (§"Intended System Role", §"Deployment").
- **Tests**: 51/51 passing.

**Project status: ENGINEERING COMPLETE · SERVING-READY.**

## Final Freeze Verdict

**READY TO FREEZE.**
