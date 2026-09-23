"""
Engineering CORRECTION, not a new optimization cycle: re-runs ONLY Level 4
(fusion + cross-encoder rerank) with the NaN bug fixed
(src/mapping.py::materialize_model_ -- see docs/CHANGE_REPORT.md "Level 4
cross-encoder bug fix" for the root-cause trace).

Untouched by this script: Level 0/1, Level 2, Level 3, Level 5, train/val/
test split membership, labels, relevance definition, K, head/tail
definitions, and every frozen Level 5 hyperparameter (alpha, beta, embed
model). Level 5 remains the frozen final architecture regardless of this
script's outcome unless the operator (see decision rule below) explicitly
decides otherwise on validation evidence.

Decision rule (do not deviate):
  1. Run corrected Level 4 on the FULL validation split.
  2. If corrected Level 4 val MRR < Level 5 val MRR (0.2648, already frozen):
     Level 5 stays the final architecture. Proceed to run corrected Level 4
     on the held-out test split EXACTLY ONCE, and stop.
  3. If corrected Level 4 val MRR >= Level 5 val MRR: STOP BEFORE TOUCHING
     TEST. Print a clear message and exit without evaluating on test. A
     human decision is required before proceeding further.

Usage: python3 scripts/rerun_level4_corrected.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapping import LexicalMapper, FusionMapper, materialize_model_  # noqa: E402
from metrics import per_query_metrics, aggregate  # noqa: E402

SPLIT_DIR = ROOT / "artifacts" / "splits"
ART_DIR = ROOT / "artifacts"
K_LIST = (1, 5, 10)


def load_split(name):
    return json.loads((SPLIT_DIR / f"{name}.json").read_text())


def load_head_index():
    return json.loads((SPLIT_DIR / "head_index.json").read_text())


def eval_rerank_corrected(rows, head_index, tail_embs, head_embs, lex_mapper, alpha, depth, cross_encoder_model, ks=K_LIST):
    fusion = FusionMapper(alpha)
    sim_matrix = tail_embs @ head_embs.T
    per_q = []
    ce_pairs, ce_meta, ranked_lists = [], [], []
    for i, row in enumerate(rows):
        sem_scores = sim_matrix[i]
        lex_scores = lex_mapper.score_all(row["tail_query"])
        fused = fusion.fuse(lex_scores, sem_scores)
        order = np.argsort(-fused)[:depth]
        cand_heads = [head_index[j] for j in order]
        ranked_lists.append(cand_heads)
        for h in cand_heads:
            ce_pairs.append((row["tail_query"], h))
        ce_meta.append(len(cand_heads))
    ce_scores = cross_encoder_model.predict(ce_pairs, batch_size=256, show_progress_bar=False)
    assert np.isfinite(ce_scores).all(), (
        "Cross-encoder still producing non-finite scores after materialize_model_ fix -- "
        "the bug is not fully resolved, do not trust these results."
    )
    idx = 0
    for i, row in enumerate(rows):
        n = ce_meta[i]
        local_scores = ce_scores[idx: idx + n]
        idx += n
        cand_heads = ranked_lists[i]
        order = np.argsort(-local_scores)
        reranked = [cand_heads[j] for j in order]
        per_q.append(per_query_metrics(reranked, set(row["positive_heads"]), ks))
    return per_q


def main():
    t_start = time.time()
    from sentence_transformers import SentenceTransformer, CrossEncoder

    frozen_config = json.loads((ART_DIR / "frozen_config.json").read_text())
    final_metrics = json.loads((ART_DIR / "final_metrics.json").read_text())
    level5_val_mrr = final_metrics["architecture_selection"]["val_mrr_by_candidate_level"]["level5_support_prior"]

    val_rows = load_split("val")
    head_index = load_head_index()

    model = SentenceTransformer(frozen_config["embed_model"], device="cpu")
    cross_encoder = CrossEncoder(frozen_config["cross_encoder"], device="cpu")
    materialize_model_(cross_encoder.model)  # THE FIX

    lex_mapper = LexicalMapper(head_index)
    head_embs = model.encode(head_index, batch_size=256, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    print("[rerun-l4] Evaluating corrected Level 4 on FULL validation split (17,498 queries)...")
    val_tail_embs = model.encode([r["tail_query"] for r in val_rows], batch_size=256,
                                  show_progress_bar=False, convert_to_numpy=True,
                                  normalize_embeddings=True).astype(np.float32)
    val_per_q = eval_rerank_corrected(
        val_rows, head_index, val_tail_embs, head_embs, lex_mapper,
        frozen_config["alpha_fusion"], frozen_config["rerank_depth"], cross_encoder,
    )
    val_agg = aggregate(val_per_q)
    print(f"[rerun-l4] Corrected Level 4 VAL MRR = {val_agg['mrr']:.4f} "
          f"(Level 5 frozen VAL MRR = {level5_val_mrr:.4f})")

    result = {
        "bugfix": "src/mapping.py::materialize_model_ applied to the cross-encoder before inference",
        "old_buggy_level4_note": "Previous level4_rerank values in artifacts/ladder_results.json were computed with a cross-encoder that silently emitted NaN for every input (numpy argsort on all-NaN preserves input order, so the old 'rerank' was actually a no-op truncated to top-10 fused candidates). Those values are INVALID and are not used for any reported claim.",
        "val_mrr_corrected_level4": val_agg["mrr"],
        "val_agg_corrected_level4": val_agg,
        "val_mrr_frozen_level5": level5_val_mrr,
        "decision": None,
        "test_mrr_corrected_level4": None,
    }

    if val_agg["mrr"] >= level5_val_mrr:
        result["decision"] = (
            "STOP_BEFORE_TEST: corrected Level 4 val MRR >= frozen Level 5 val MRR. "
            "Per the operator's decision rule, this requires a deliberate human "
            "closure decision before touching the test split. Level 5 remains the "
            "frozen final architecture pending that decision."
        )
        (ART_DIR / "level4_corrected.json").write_text(json.dumps(result, indent=2))
        print("\n" + "=" * 70)
        print("STOP: corrected Level 4 val MRR (%.4f) >= frozen Level 5 val MRR (%.4f)."
              % (val_agg["mrr"], level5_val_mrr))
        print("Per the decision rule, the test split is NOT evaluated automatically.")
        print("Level 5 remains frozen as the final architecture pending an explicit")
        print("human decision. Result saved to artifacts/level4_corrected.json.")
        print("=" * 70)
        return

    result["decision"] = (
        "PROCEED_TO_TEST: corrected Level 4 val MRR < frozen Level 5 val MRR. "
        "Level 5 stays the final architecture. Evaluating corrected Level 4 on "
        "the held-out test split exactly once, for an accurate ladder comparison "
        "entry only -- this does not change the final system."
    )
    print("[rerun-l4] Corrected Level 4 does not beat frozen Level 5 on validation.")
    print("[rerun-l4] Level 5 remains the final architecture.")
    print("[rerun-l4] Evaluating corrected Level 4 on the held-out TEST split, exactly once...")

    test_rows = load_split("test")
    test_tail_embs = model.encode([r["tail_query"] for r in test_rows], batch_size=256,
                                   show_progress_bar=False, convert_to_numpy=True,
                                   normalize_embeddings=True).astype(np.float32)
    test_per_q = eval_rerank_corrected(
        test_rows, head_index, test_tail_embs, head_embs, lex_mapper,
        frozen_config["alpha_fusion"], frozen_config["rerank_depth"], cross_encoder,
    )
    test_agg = aggregate(test_per_q)
    result["test_mrr_corrected_level4"] = test_agg["mrr"]
    result["test_agg_corrected_level4"] = test_agg
    result["test_per_query_corrected_level4"] = test_per_q
    result["val_per_query_corrected_level4"] = val_per_q
    result["tail_query_order_val"] = [r["tail_query"] for r in val_rows]
    result["tail_query_order_test"] = [r["tail_query"] for r in test_rows]

    print(f"[rerun-l4] Corrected Level 4 TEST MRR = {test_agg['mrr']:.4f} "
          f"(frozen baseline test MRR = {final_metrics['baseline_tail_metric']:.4f}, "
          f"frozen final test MRR = {final_metrics['final_tail_metric']:.4f})")

    result["elapsed_seconds"] = round(time.time() - t_start, 1)
    (ART_DIR / "level4_corrected.json").write_text(json.dumps(result, indent=2))
    print(f"[rerun-l4] Done in {result['elapsed_seconds']}s. Saved artifacts/level4_corrected.json")


if __name__ == "__main__":
    main()
