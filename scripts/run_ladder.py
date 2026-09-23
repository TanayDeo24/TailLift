"""
Modernization comparison ladder (Section 20) + optimization (Section 37-39).

Phase A (VALIDATION ONLY, on a fixed random sample of the val split):
  - compare embedding model candidates
  - tune fusion alpha, support-prior beta, cross-encoder rerank depth
Phase B (frozen config, full val + full test, single pass):
  - compute Level 0-5 metrics on the complete held-out val and test sets
  - compute the head-query self-retrieval ceiling (reference for gap analysis)

No test-set value is inspected before the config is frozen. See
docs/HEAD_TAIL_METHODOLOGY.md and docs/BENCHMARK_REPORT.md for the frozen
decisions this script produces.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapping import LexicalMapper, FusionMapper, SupportPriorMapper, tokenize  # noqa: E402
from metrics import per_query_metrics, aggregate, paired_bootstrap_ci  # noqa: E402

SPLIT_DIR = ROOT / "artifacts" / "splits"
ART_DIR = ROOT / "artifacts"
VAL_SAMPLE_N = 3000
RERANK_DEPTH_CANDIDATES = [10, 20]
ALPHA_CANDIDATES = [0.3, 0.5, 0.7, 0.9]
BETA_CANDIDATES = [0.0, 0.1, 0.25]
EMBED_MODEL_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
]
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_LIST = (1, 5, 10)
PRIMARY_K = 5
SEED = 20260922


def load_split(name):
    return json.loads((SPLIT_DIR / f"{name}.json").read_text())


def load_head_index():
    return json.loads((SPLIT_DIR / "head_index.json").read_text())


def encode(model, texts, batch_size=256):
    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return embs.astype(np.float32)


def rank_heads(scores, head_index):
    order = np.argsort(-scores)
    return [head_index[i] for i in order]


def eval_lexical(rows, head_index, lex_mapper, ks=K_LIST):
    per_q = []
    for row in rows:
        scores = lex_mapper.score_all(row["tail_query"])
        ranked = rank_heads(scores, head_index)
        per_q.append(per_query_metrics(ranked, set(row["positive_heads"]), ks))
    return per_q


def eval_semantic_and_fusion(rows, head_index, tail_embs, head_embs, lex_mapper, alpha, ks=K_LIST):
    fusion = FusionMapper(alpha)
    per_q_sem, per_q_fus = [], []
    sim_matrix = tail_embs @ head_embs.T
    for i, row in enumerate(rows):
        sem_scores = sim_matrix[i]
        lex_scores = lex_mapper.score_all(row["tail_query"])
        ranked_sem = rank_heads(sem_scores, head_index)
        fused = fusion.fuse(lex_scores, sem_scores)
        ranked_fus = rank_heads(fused, head_index)
        pos = set(row["positive_heads"])
        per_q_sem.append(per_query_metrics(ranked_sem, pos, ks))
        per_q_fus.append(per_query_metrics(ranked_fus, pos, ks))
    return per_q_sem, per_q_fus, fusion


def eval_prior(rows, head_index, tail_embs, head_embs, lex_mapper, alpha, beta, train_pos_counts, ks=K_LIST):
    fusion = FusionMapper(alpha)
    prior = SupportPriorMapper(head_index, train_pos_counts, beta)
    sim_matrix = tail_embs @ head_embs.T
    per_q = []
    for i, row in enumerate(rows):
        sem_scores = sim_matrix[i]
        lex_scores = lex_mapper.score_all(row["tail_query"])
        fused = fusion.fuse(lex_scores, sem_scores)
        final = prior.add_prior(fused)
        ranked = rank_heads(final, head_index)
        per_q.append(per_query_metrics(ranked, set(row["positive_heads"]), ks))
    return per_q


def eval_rerank(rows, head_index, tail_embs, head_embs, lex_mapper, alpha, depth, cross_encoder_model, ks=K_LIST):
    fusion = FusionMapper(alpha)
    sim_matrix = tail_embs @ head_embs.T
    per_q = []
    ce_pairs, ce_meta = [], []
    ranked_lists = []
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
    global VAL_SAMPLE_N
    smoke = "--smoke" in sys.argv
    t_start = time.time()
    from sentence_transformers import SentenceTransformer, CrossEncoder

    train_rows = load_split("train")
    val_rows = load_split("val")
    test_rows = load_split("test")
    head_index = load_head_index()

    if smoke:
        train_rows, val_rows, test_rows = train_rows[:200], val_rows[:60], test_rows[:60]
        head_index = head_index[:300]
        VAL_SAMPLE_N = 40

    train_pos_counts = Counter()
    for r in train_rows:
        for h in r["positive_heads"]:
            train_pos_counts[h] += 1

    lex_mapper = LexicalMapper(head_index)

    # ---------- LEVEL 0/1: lexical baseline (frozen, no tuning) ----------
    print("[ladder] Level 0/1 lexical baseline on val...")
    val_lex = eval_lexical(val_rows, head_index, lex_mapper)

    # ---------- Phase A: VALIDATION-ONLY search over embedding models ----------
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(val_rows), size=min(VAL_SAMPLE_N, len(val_rows)), replace=False)
    val_sample = [val_rows[i] for i in sample_idx]
    print(f"[ladder] Phase A validation search on {len(val_sample)} sampled val queries...")

    search_log = []
    best = {"mrr": -1}
    head_embs_cache = {}
    sample_tail_embs_cache = {}
    for model_name in EMBED_MODEL_CANDIDATES:
        t0 = time.time()
        model = SentenceTransformer(model_name, device="cpu")
        head_embs = encode(model, head_index)
        tail_embs = encode(model, [r["tail_query"] for r in val_sample])
        head_embs_cache[model_name] = head_embs
        sample_tail_embs_cache[model_name] = tail_embs
        for alpha in ALPHA_CANDIDATES:
            per_q_sem, per_q_fus, _ = eval_semantic_and_fusion(
                val_sample, head_index, tail_embs, head_embs, lex_mapper, alpha
            )
            agg_sem, agg_fus = aggregate(per_q_sem), aggregate(per_q_fus)
            entry = {
                "model": model_name, "alpha": alpha,
                "semantic_mrr": agg_sem["mrr"], "fusion_mrr": agg_fus["mrr"],
                "fusion_recall@5": agg_fus["recall@5"],
                "elapsed_s": round(time.time() - t0, 1),
            }
            search_log.append(entry)
            if agg_fus["mrr"] > best["mrr"]:
                best = {"mrr": agg_fus["mrr"], "model": model_name, "alpha": alpha}
        del model

    print(f"[ladder] Phase A best so far: {best}")
    best_model_name, best_alpha = best["model"], best["alpha"]

    # tune beta (support prior) and rerank depth using the SAME val sample + best model/alpha
    head_embs = head_embs_cache[best_model_name]
    tail_embs = sample_tail_embs_cache[best_model_name]
    best_beta, best_beta_mrr = 0.0, -1
    for beta in BETA_CANDIDATES:
        per_q = eval_prior(val_sample, head_index, tail_embs, head_embs, lex_mapper, best_alpha, beta, train_pos_counts)
        m = aggregate(per_q)["mrr"]
        search_log.append({"stage": "beta_search", "beta": beta, "mrr": m})
        if m > best_beta_mrr:
            best_beta_mrr, best_beta = m, beta

    print(f"[ladder] Phase A best beta: {best_beta} (mrr={best_beta_mrr:.4f})")

    print("[ladder] Phase A: cross-encoder rerank depth search...")
    cross_encoder_model = CrossEncoder(CROSS_ENCODER, device="cpu")
    best_depth, best_depth_mrr = RERANK_DEPTH_CANDIDATES[0], -1
    for depth in RERANK_DEPTH_CANDIDATES:
        per_q = eval_rerank(val_sample, head_index, tail_embs, head_embs, lex_mapper, best_alpha, depth, cross_encoder_model)
        m = aggregate(per_q)["mrr"]
        search_log.append({"stage": "rerank_depth_search", "depth": depth, "mrr": m})
        if m > best_depth_mrr:
            best_depth_mrr, best_depth = m, depth

    frozen_config = {
        "embed_model": best_model_name,
        "alpha_fusion": best_alpha,
        "beta_support_prior": best_beta,
        "rerank_depth": best_depth,
        "cross_encoder": CROSS_ENCODER,
        "primary_k": PRIMARY_K,
        "primary_metric": "MRR",
        "seed": SEED,
        "val_sample_n": len(val_sample),
    }
    cfg_path = ART_DIR / ("frozen_config_smoke.json" if smoke else "frozen_config.json")
    log_path = ART_DIR / ("optimization_search_log_smoke.json" if smoke else "optimization_search_log.json")
    cfg_path.write_text(json.dumps(frozen_config, indent=2))
    log_path.write_text(json.dumps(search_log, indent=2))
    print("[ladder] FROZEN CONFIG:", frozen_config)

    # ---------- Phase B: frozen config, FULL val + FULL test, single pass ----------
    print("[ladder] Phase B: full val + full test evaluation with frozen config...")
    final_model = SentenceTransformer(best_model_name, device="cpu")
    head_embs_full = encode(final_model, head_index)

    results = {"frozen_config": frozen_config, "levels": {}}

    for split_name, rows in [("val", val_rows), ("test", test_rows)]:
        tail_embs_full = encode(final_model, [r["tail_query"] for r in rows])

        lvl0 = eval_lexical(rows, head_index, lex_mapper)
        lvl2, lvl3, _ = eval_semantic_and_fusion(rows, head_index, tail_embs_full, head_embs_full, lex_mapper, best_alpha)
        lvl4 = eval_rerank(rows, head_index, tail_embs_full, head_embs_full, lex_mapper, best_alpha, best_depth, cross_encoder_model)
        lvl5 = eval_prior(rows, head_index, tail_embs_full, head_embs_full, lex_mapper, best_alpha, best_beta, train_pos_counts)

        results["levels"].setdefault(split_name, {})
        results["levels"][split_name] = {
            "level0_1_lexical": {"per_query": lvl0, "agg": aggregate(lvl0)},
            "level2_semantic": {"per_query": lvl2, "agg": aggregate(lvl2)},
            "level3_fusion": {"per_query": lvl3, "agg": aggregate(lvl3)},
            "level4_rerank": {"per_query": lvl4, "agg": aggregate(lvl4)},
            "level5_support_prior": {"per_query": lvl5, "agg": aggregate(lvl5)},
        }
        print(f"[ladder] {split_name} lexical MRR={aggregate(lvl0)['mrr']:.4f} "
              f"semantic MRR={aggregate(lvl2)['mrr']:.4f} fusion MRR={aggregate(lvl3)['mrr']:.4f} "
              f"rerank MRR={aggregate(lvl4)['mrr']:.4f} prior MRR={aggregate(lvl5)['mrr']:.4f}")

        # per-query tail-query-id order preserved for paired bootstrap later
        results["levels"][split_name]["tail_query_order"] = [r["tail_query"] for r in rows]

    # ---------- head-query self-retrieval ceiling (reference for gap analysis) ----------
    print("[ladder] Computing head-query self-retrieval ceiling...")
    head_self_embs = head_embs_full  # heads embedded against themselves
    ceiling_per_q = []
    for i, h in enumerate(head_index):
        # ceiling task: does the head query (as its own query) retrieve ITSELF
        # at the top when self-identity IS allowed (diagonal restored) -- this
        # is a best-case reference, not a production measurement. See docs.
        scores = head_self_embs[i] @ head_embs_full.T
        ranked = rank_heads(scores, head_index)
        ceiling_per_q.append(per_query_metrics(ranked, {h}, K_LIST))
    results["head_self_retrieval_ceiling"] = {
        "per_query": ceiling_per_q, "agg": aggregate(ceiling_per_q),
        "n": len(head_index),
        "definition": (
            "Reference ceiling only: a head query embedded and matched against "
            "the SAME head-query index it was drawn from (self-identity "
            "retrieval). This is NOT an independent measurement of production "
            "head-query search quality (no raw head-only click logs are "
            "available in this repository -- see docs/DATA_PROVENANCE.md). It "
            "represents the best case a query with full historical evidence "
            "could achieve under this retrieval architecture, and is used only "
            "as the reference point for baseline_gap / final_gap in "
            "docs/BENCHMARK_REPORT.md."
        ),
    }

    results["elapsed_seconds"] = round(time.time() - t_start, 1)
    res_path = ART_DIR / ("ladder_results_smoke.json" if smoke else "ladder_results.json")
    res_path.write_text(json.dumps(results, indent=2))
    print(f"[ladder] Done in {results['elapsed_seconds']}s. Saved {res_path}")


if __name__ == "__main__":
    main()
