"""
Correctness-critical tests (Section 44). Run with: pytest tests/ -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mapping import LexicalMapper, FusionMapper, SupportPriorMapper, jaccard, tokenize  # noqa: E402
from metrics import (  # noqa: E402
    rank_of_first_relevant,
    per_query_metrics,
    aggregate,
    paired_bootstrap_ci,
)


# ---------------- tokenize / jaccard ----------------

def test_tokenize_lowercases_and_splits():
    assert tokenize("AOL Update!") == ["aol", "update"]


def test_jaccard_identical_sets():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_empty_both():
    assert jaccard(set(), set()) == 0.0


# ---------------- LexicalMapper ----------------

def test_lexical_mapper_ranks_exact_match_top():
    head_index = ["aol update", "cnn news", "yahoo mail"]
    mapper = LexicalMapper(head_index)
    scores = mapper.score_all("aol update")
    assert head_index[int(np.argmax(scores))] == "aol update"


def test_lexical_mapper_empty_query_no_crash():
    head_index = ["aol update", "cnn news"]
    mapper = LexicalMapper(head_index)
    scores = mapper.score_all("")
    assert len(scores) == len(head_index)
    assert np.all(scores == 0.0)


# ---------------- FusionMapper ----------------

def test_fusion_alpha_zero_is_pure_lexical_ordering():
    fusion = FusionMapper(alpha=0.0)
    lex = np.array([0.9, 0.1, 0.5])
    sem = np.array([0.0, 1.0, 0.0])  # would reorder if used
    fused = fusion.fuse(lex, sem)
    assert np.argmax(fused) == 0  # lexical top choice preserved


def test_fusion_alpha_one_is_pure_semantic_ordering():
    fusion = FusionMapper(alpha=1.0)
    lex = np.array([0.9, 0.1, 0.5])
    sem = np.array([0.0, 1.0, 0.0])
    fused = fusion.fuse(lex, sem)
    assert np.argmax(fused) == 1


def test_fusion_constant_scores_no_div_by_zero():
    fusion = FusionMapper(alpha=0.5)
    lex = np.array([0.5, 0.5, 0.5])
    sem = np.array([0.2, 0.2, 0.2])
    fused = fusion.fuse(lex, sem)
    assert np.all(np.isfinite(fused))


# ---------------- SupportPriorMapper ----------------

def test_support_prior_uses_only_provided_counts():
    head_index = ["a", "b", "c"]
    prior = SupportPriorMapper(head_index, {"a": 100}, beta=1.0)
    fused = np.array([0.0, 0.0, 0.0])
    out = prior.add_prior(fused)
    assert np.argmax(out) == 0  # "a" has the only nonzero prior


def test_support_prior_beta_zero_ignores_prior():
    head_index = ["a", "b"]
    prior = SupportPriorMapper(head_index, {"a": 1000}, beta=0.0)
    fused = np.array([0.1, 0.9])
    out = prior.add_prior(fused)
    np.testing.assert_allclose(out, fused)


def test_no_future_leakage_support_prior_only_from_train():
    """The support prior must be built only from a counts dict the caller
    passes in -- verifies the API cannot silently see val/test labels."""
    train_only_counts = {"headA": 5}
    head_index = ["headA", "headB"]
    prior = SupportPriorMapper(head_index, train_only_counts, beta=0.5)
    # headB, which might be popular in val/test, gets zero prior since it's
    # absent from the counts dict built from train only.
    assert prior.log_prior[1] == 0.0


# ---------------- metrics ----------------

def test_rank_of_first_relevant_found():
    assert rank_of_first_relevant(["a", "b", "c"], {"b"}) == 2


def test_rank_of_first_relevant_not_found():
    assert rank_of_first_relevant(["a", "b"], {"z"}) is None


def test_per_query_metrics_hit_and_recall():
    m = per_query_metrics(["a", "b", "c", "d"], {"c"}, ks=(1, 5))
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["hit@1"] == 0.0
    assert m["hit@5"] == 1.0
    assert m["recall@5"] == 1.0


def test_per_query_metrics_no_relevant_in_list():
    m = per_query_metrics(["a", "b"], {"z"}, ks=(1,))
    assert m["mrr"] == 0.0
    assert m["hit@1"] == 0.0


def test_per_query_metrics_empty_relevant_set_no_crash():
    m = per_query_metrics(["a", "b"], set(), ks=(1,))
    assert m["recall@1"] == 0.0


def test_aggregate_averages_correctly():
    dicts = [{"mrr": 1.0}, {"mrr": 0.0}]
    assert aggregate(dicts)["mrr"] == 0.5


def test_aggregate_empty_list():
    assert aggregate([]) == {}


def test_paired_bootstrap_ci_zero_delta_when_identical():
    a = [0.5] * 50
    b = [0.5] * 50
    obs, lo, hi, n, resamples, seed = paired_bootstrap_ci(a, b, n_resamples=200)
    assert obs == 0.0
    assert lo <= 0.0 <= hi


def test_paired_bootstrap_ci_detects_positive_delta():
    rng = np.random.default_rng(0)
    a = list(rng.normal(0.3, 0.05, 200))
    b = list(rng.normal(0.5, 0.05, 200))
    obs, lo, hi, n, resamples, seed = paired_bootstrap_ci(a, b, n_resamples=1000)
    assert obs > 0
    assert lo > 0  # CI should exclude zero given the large true gap


def test_paired_bootstrap_ci_requires_equal_length():
    with pytest.raises(AssertionError):
        paired_bootstrap_ci([1, 2], [1, 2, 3])


# ---------------- split integrity (Section 28/44: no leakage tests) ----------------

SPLIT_DIR = ROOT / "artifacts" / "splits"


@pytest.mark.skipif(not SPLIT_DIR.exists(), reason="splits not built yet")
def test_splits_are_disjoint():
    train = {r["tail_query"] for r in json.loads((SPLIT_DIR / "train.json").read_text())}
    val = {r["tail_query"] for r in json.loads((SPLIT_DIR / "val.json").read_text())}
    test = {r["tail_query"] for r in json.loads((SPLIT_DIR / "test.json").read_text())}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


@pytest.mark.skipif(not SPLIT_DIR.exists(), reason="splits not built yet")
def test_every_split_row_has_at_least_one_positive():
    for name in ("train", "val", "test"):
        rows = json.loads((SPLIT_DIR / f"{name}.json").read_text())
        assert all(len(r["positive_heads"]) >= 1 for r in rows), (
            f"{name} split contains an ineligible (no ground truth) tail query"
        )


@pytest.mark.skipif(not SPLIT_DIR.exists(), reason="splits not built yet")
def test_head_index_matches_head150_size():
    head_index = json.loads((SPLIT_DIR / "head_index.json").read_text())
    assert len(head_index) == 12261
