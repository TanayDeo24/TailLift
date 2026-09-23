"""
Regression test for the Level 4 cross-encoder NaN bug found while building
the serving layer (see docs/CHANGE_REPORT.md "Level 4 cross-encoder bug
fix"). Without src.mapping.materialize_model_, this cross-encoder model
produces NaN scores for every input on this platform -- this test fails
loudly if that regresses.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapping import materialize_model_  # noqa: E402


@pytest.fixture(scope="module")
def cross_encoder():
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    materialize_model_(ce.model)
    return ce


def test_cross_encoder_outputs_are_finite(cross_encoder):
    scores = cross_encoder.predict(
        [("aol update", "aol update"), ("aol update", "aol upgrade"),
         ("nfldraftcountdown", "nfl draft"), ("nfldraftcountdown", "california lottery")],
        show_progress_bar=False,
    )
    assert np.isfinite(scores).all(), f"non-finite cross-encoder scores: {scores}"


def test_cross_encoder_ranks_relevant_pair_higher(cross_encoder):
    scores = cross_encoder.predict(
        [("nfldraftcountdown", "nfl draft"), ("nfldraftcountdown", "california lottery")],
        show_progress_bar=False,
    )
    assert scores[0] > scores[1]


def test_cross_encoder_exact_match_scores_highest(cross_encoder):
    scores = cross_encoder.predict(
        [("aol update", "aol update"), ("aol update", "aol upgrade")],
        show_progress_bar=False,
    )
    assert scores[0] > scores[1]


def test_materialize_model_is_idempotent(cross_encoder):
    """Calling it twice must not change outputs or break anything."""
    materialize_model_(cross_encoder.model)
    scores = cross_encoder.predict([("a query", "a head query")], show_progress_bar=False)
    assert np.isfinite(scores).all()


def test_materialize_model_preserves_parameter_values():
    """Cloning must not change the numeric content of the weights -- only
    their storage. Regression guard against accidentally reinitializing."""
    from sentence_transformers import CrossEncoder
    import torch

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    before = {n: p.detach().clone() for n, p in ce.model.named_parameters()}
    materialize_model_(ce.model)
    after = dict(ce.model.named_parameters())
    for name, val in before.items():
        assert torch.equal(val, after[name].detach()), f"parameter {name} changed value after materialize_model_"
