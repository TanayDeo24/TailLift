# Query-Log / Original-Code Audit

Traces the ACTUAL executable logic in the original repository, as opposed to
what `README.md` claims. Every file listed under "inspect" in the project
brief was read in full.

## README claim vs. actual code

| Claim (README) | Actual code | Verdict |
|---|---|---|
| Head threshold: frequency ≥150 | `separator.py`: `HEAD_LIMIT = 150` ✓. `vidit.jain@ada.iiit.ac.in`: `HEAD_LIMIT = 100` ✗ | **Discrepancy.** Two divergent copies of the same logic exist with different thresholds. `separator.py`'s 150 matches the README; the other file (likely an earlier or alternate iteration, oddly named after an email attachment) uses 100. |
| Tail threshold: frequency ≤5 | Both files: `TAIL_LIMIT = 5`, but comparison operator is `<` not `<=` (`query_Dict[key] < TAIL_LIMIT`), i.e. actual tail rule is **frequency < 5 (≤4)**, not ≤5 | **Off-by-one vs. stated claim.** |
| Relevance rule: same URL, rank ≤5 in both | `separator.py.findScore`: `rankTail<=5 and rankHead<=5` → 1 else **0**. `vidit.jain@ada.iiit.ac.in.findScore`: same condition → 1 else **-1**. Also `MAX_RANK_ALLOWED` is defined (3 in one file, 5 in the other) but **never actually used** in either file's `findScore`. | Matches the rank≤5 claim, but the two files disagree on the negative label's numeric value (0 vs -1), and a defined-but-dead `MAX_RANK_ALLOWED=3` constant in `separator.py` suggests the rank cutoff was being experimented with but the change was never wired in. |
| Negative sampling | `separator.py.mapping`: pads negatives via random head sampling until `coNeg < coPos` (1:1 ratio). `vidit...` variant: pads until `coNeg < 3*coPos` (3:1 ratio). | Two different sampling ratios across the two scripts — another unreconciled divergence. |
| Model 2 (SVM ensemble): 78% accuracy | `SVM.py`/`SVMtemp.py` implement the 4-feature pipeline (Jaccard, WordNet sentence similarity, Word2Vec cosine, Dexter entity score) and call `clf.fit`/`clf.predict`, printing a `precision` figure — **but no result file, log, or saved output containing "78%" (or any number) exists in the repository.** The accuracy claim is not independently verifiable from anything committed. | **Unverifiable.** Treated as an unverified historical claim, not reproduced evidence. |
| Model 3 (NN / LSTM) | `neuralNet.py` is syntactically valid Python 2 (translatable) and depends on a Google-News word2vec binary not present in the repo (`../word2vec__GoogleNews_100Bwords__300Dvectors__3M_vocab.bin`). `lstm_rnn.py` contains a **literal Python syntax error** (`tokenizer,embedding_matrix,head_data,tail_data,,word_index =` — double comma) and references undefined names (`labels` never assigned in `createEmbeddingMatrix`; `MAX_SEQUENCE_LENGTH` and `word_index` used inside `model_LSTM_Classifier` without being passed as parameters; `np` used but only `numpy` imported, not `np`). | `lstm_rnn.py` **could not have run successfully as committed**, in Python 2 or 3. This is a repository defect, not a modernization casualty. |
| Model 1 (Entity Linking baseline) | `Entity_linkage.py` calls a live HTTP demo service (`dexterdemo.isti.cnr.it:8080`). Connectivity test performed 2026-09-22: **connection fails (curl exit, HTTP code 000)**. | External dependency is dead; cannot be executed at all today, faithfully or otherwise. |

## Real data flow (as executed, not as described)

```
newMedium.txt (1,439,823 pre-built pairs, ALREADY head/tail-labeled)
        |
        v
  [no code in this repo regenerates this file from a raw log --
   the raw log and the exact script invocation that produced it
   are both absent; separator.py / vidit.jain@... implement the
   *algorithm* that would have produced a file in this shape, but
   cannot be run against real AOL data because that raw data is
   not present]
        |
        v
SVM.py / SVMtemp.py consume a DIFFERENT file (train_data.txt /
DATA/finalFeature.txt, the latter also absent) -- i.e. the
committed SVM training code was never actually wired to
newMedium.txt at all.
        |
        v
neuralNet.py / mapper.py / newMapper.py consume yet another format
(pipe-delimited "head || tail || label", matching newMedium.txt)
via mapper.py's createHeadVectors(), and depend on an absent
Word2Vec binary.
```

**Finding: the three "models" in the README were never a single connected
pipeline.** Each consumes a different, sometimes-absent input file format.
This project's modernization (see `docs/FINAL_ARCHITECTURE.md`) instead
builds ONE coherent, leakage-audited pipeline on top of the one file that is
actually complete, internally consistent, and large enough to support
held-out evaluation: `newMedium.txt`.

## Working vs. broken vs. outdated (summary)

| Component | Status |
|---|---|
| Jaccard token-overlap (in `separator.py`, `entity_check.py`'s sibling logic) | **Working, faithfully reproduced** in `src/mapping.py::LexicalMapper` (Level 0/1) |
| WordNet sentence similarity (`sentence_similarity.py`) | Algorithmically sound, Python-2-only syntax; not carried into the modern ladder (superseded by semantic embeddings, which subsume this role — see `docs/CHANGE_REPORT.md`) |
| Word2Vec cosine similarity (`cosine_similarity.py`) | Algorithmically sound; superseded by sentence-transformer embeddings for the same reason |
| Entity linking (`Entity_linkage.py`, `entity_check.py`) | **Broken — external API dead**, unreproducible without standing up new infrastructure; not carried into the modern ladder |
| SVM ensemble (`SVM.py`) | Runnable in principle (minus the entity feature) but was never actually connected to `newMedium.txt`; its claimed 78% accuracy is unverifiable |
| Neural net (`neuralNet.py`) | Runnable in principle; depends on an absent large pretrained embedding file |
| LSTM (`lstm_rnn.py`) | **Broken as committed** — syntax error, undefined names |
| `separator.py` head/tail/pair-generation logic | Working; the two threshold-divergent copies are documented above; this project inherits the **frozen output** (`newMedium.txt`, `head150.txt`) rather than re-running either divergent script, since the raw log needed to re-run them is absent |
| Original train/test split (`out (1).txt` → `train_data.txt`/`test_data.txt`) | **Working but leaky** (4.3% query overlap, confirmed) — not reused |

## Evaluation risks identified

1. Original split leakage (above) — not inherited.
2. No raw log means the head/tail *threshold decision itself* cannot be
   re-derived from first principles in this project (Section 11 of the brief
   calls for building a frequency histogram from raw counts — impossible
   here). Handled explicitly in `docs/HEAD_TAIL_METHODOLOGY.md` as an
   inherited, not re-derived, definition.
3. Click-based relevance labels inherit whatever position bias existed in
   the original AOL search results; this project did not have the ability to
   re-audit that bias (no raw rank/impression data survives per-row in
   `newMedium.txt` — it was consumed into the binary label at
   label-construction time and not retained).
