# Project State — Initial (Pre-Modernization Snapshot)

Recorded before any modification. Source of truth for "what existed before this effort."

## Repository

- Source: https://github.com/ranjan019/Head-Query-Tail-Query-Mapping
- Local working directory: `/Users/tanaydeo/Documents/Head-Query-Tail-Query-Mapping-master`
- **Git state at start: not a git repository.** The directory was a flat, un-versioned
  snapshot (a GitHub "download zip"-style extraction, evidenced by the `-master`
  suffix in the folder name and uniform file mtimes of 2017-11-13). `git init` was
  run and the untouched original files were committed as the first commit
  (`946b9f9`, message "Initial snapshot...") so every subsequent change is now
  diffable against the true original state. No history was rewritten; nothing
  was pushed anywhere.
- Branch after init: `main`.

## Runtime environment (as found)

- Python 3.12.7 (`/opt/anaconda3/bin/python3`) — the ORIGINAL code is Python 2
  (uses `print x` statements, `.pyc` files compiled for Python 2). It does not
  run under the installed Python 3 interpreter without modification.
- Installed ML/NLP libraries relevant to this project: `numpy` 1.26.4, `pandas`
  2.2.2, `scikit-learn` 1.5.1, `gensim` 4.3.3. `sentence-transformers` (6.1.0)
  and `torch` were not present and were installed during this effort (see
  `docs/REPRODUCIBILITY.md`).
- No `requirements.txt`, `Pipfile`, `environment.yml`, or `setup.py` present in
  the original repository — dependencies had to be inferred from `import`
  statements in each script.
- Network access to the public internet is available from this environment
  (verified: `https://huggingface.co` reachable). The original project's
  external dependency, the Dexter entity-linking demo API
  (`http://dexterdemo.isti.cnr.it:8080/...`), is **not** reachable (connection
  fails, confirmed via `curl`) — that demo server has been decommissioned since
  2017.

## Original repository contents (32 files)

| File | Type | Role (see docs/QUERY_LOG_AUDIT.md for full trace) |
|---|---|---|
| `README.md` | doc | Project description, claimed methods/results, external links |
| `_config.yml` | config | GitHub Pages Jekyll theme config only |
| `Entity_linkage.py` | Py2 | Model 1 baseline — Dexter entity relatedness |
| `entity_check.py` | Py2 | Entity-overlap feature module (used by SVM) |
| `cosine_similarity.py` | Py2 | Word2Vec average-embedding cosine feature |
| `sentence_similarity.py` | Py2 | WordNet path-similarity feature |
| `SVM.py`, `SVMtemp.py` | Py2 | Model 2 — 4-feature ensemble SVM classifier (two variant scripts) |
| `neuralNet.py` | Py2 | Model 3 Method 1 — dense NN over averaged word2vec pair vectors |
| `lstm_rnn.py` | Py2 | Model 3 Method 2 — LSTM NN (contains a syntax error, never ran as committed — see audit) |
| `mapper.py`, `newMapper.py` | Py2 | Head-query vector index + inference-time lookup helpers |
| `separator.py` | Py2 | Head/tail frequency segmentation + relevance-pair generation (HEAD_LIMIT=150) |
| `vidit.jain@ada.iiit.ac.in` | Py2 | A second, divergent copy of the separator logic (HEAD_LIMIT=100) — see audit for the discrepancy |
| `cosModel.model` (10.5 MB) | artifact | Trained gensim Word2Vec model |
| `w2vNet.model` (33.3 MB) | artifact | Trained gensim Word2Vec model |
| `my_model.h5` (116 KB) | artifact | Trained Keras dense NN weights |
| `text8` (100 MB) | data | Standard Word2Vec training corpus (Wikipedia dump, not query-log data) |
| `newMedium.txt` (63.3 MB, 1,439,823 rows) | data | Master tail→head query-pair file with binary relevance labels |
| `out (1).txt` (4.1 MB, 100,000 rows) | data | A second pair-file, different schema, used for the original SVM train/test |
| `train_data.txt` / `test_data.txt` | data | Contiguous 80/20 split of `out (1).txt` (confirmed leaky — see audit) |
| `head150.txt` (12,261 lines) | data | Frozen head-query vocabulary |
| `headkey.txt` / `tailkey.txt` | data | Flat lists of clicked URLs (298 / 230 entries) — no per-query linkage |
| `finalData.txt` | data | Empty file (0 bytes) |
| `final.png` | image | Result/architecture figure referenced by README |
| `*.pyc` | build artifact | Compiled Python 2 bytecode for 3 modules |

**Raw AOL query log (AnonID, Query, QueryTime, ItemRank, ClickURL) is NOT present
anywhere in this repository.** Only already-derived pair files remain. This is a
hard data-availability constraint that shapes the entire methodology — see
`docs/DATA_PROVENANCE.md`.

## README claims (recorded verbatim intent, not yet verified against code)

- Head/tail thresholds: frequency ≥150 → head, ≤5 → tail.
- Pair relevance rule: same clicked URL for both queries AND rank ≤5 in both.
- Model 2 (SVM ensemble) claimed test accuracy: 78%.
- A Google Forms user study claims average satisfaction scores: Original query
  3.85/5, NN model 2.28/5, Baseline model 2.25/5 (external, unverifiable from
  this repository, not reproducible, not used as evidence in this project).
- Links to a Google Doc technical report, a SlideShare deck, and a second
  mirrored repo (`Viditjn/Head-Tail-Query-Mapping`) — none of these are
  authoritative code artifacts and are not treated as ground truth here.

## Existing tests

None. No `test_*.py`, no CI configuration, no `pytest.ini`, no `.github/workflows`.

## Known legacy/dead dependencies

- `pydexter` (Python client for the Dexter entity-linking API) — API endpoint dead.
- Python 2 syntax throughout (`print` statements, old `keras.layers.normalization`
  import path, `gensim` API from a version predating the 4.x rewrite,
  `model.wv.vocab` — removed in modern gensim).
- `lstm_rnn.py` contains a literal double-comma syntax error in a tuple
  unpack (`tokenizer,embedding_matrix,head_data,tail_data,,word_index =`) and
  references undefined names (`labels`, `MAX_SEQUENCE_LENGTH`, module-level
  `word_index` used inside a function without being passed in) — it could not
  have executed successfully even under Python 2 as committed.

This document is the baseline. Every subsequent document in `docs/` builds on
top of, and must remain consistent with, what is recorded here.
