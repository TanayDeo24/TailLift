"""
Query-mapping scorers for the head-query candidate index.

Each scorer implements: score_all(tail_query, head_index, head_vectors) -> np.ndarray
of length len(head_index), higher = more relevant. Ranking + metrics live in
scripts/evaluate.py; this module only implements mapping logic (Section 27 ladder).
"""
import re
from typing import List

import numpy as np


def tokenize(q: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", q.lower())


def jaccard(a_tokens: set, b_tokens: set) -> float:
    if not a_tokens and not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


class LexicalMapper:
    """Level 0/1: original repository's Jaccard token-overlap similarity
    (faithfully reproduced from separator.py's calcJaccard), applied as a
    retrieval ranker over the full head-query index instead of the original
    pairwise binary classifier."""

    name = "lexical_jaccard"

    def __init__(self, head_index: List[str]):
        self.head_index = head_index
        self.head_token_sets = [set(tokenize(h)) for h in head_index]

    def score_all(self, tail_query: str) -> np.ndarray:
        tail_tokens = set(tokenize(tail_query))
        return np.array([jaccard(tail_tokens, hts) for hts in self.head_token_sets])


class SemanticMapper:
    """Level 2: sentence-transformer bi-encoder cosine similarity retrieval."""

    name = "semantic_bi_encoder"

    def __init__(self, head_index: List[str], model_name: str, head_embeddings: np.ndarray):
        self.head_index = head_index
        self.model_name = model_name
        self.head_embeddings = head_embeddings  # pre-normalized, shape (N, D)

    def score_all_batch(self, tail_embeddings: np.ndarray) -> np.ndarray:
        # tail_embeddings: (M, D) pre-normalized -> cosine sim via dot product
        return tail_embeddings @ self.head_embeddings.T


class FusionMapper:
    """Level 3: convex combination of lexical and semantic scores.
    Both score vectors are min-max normalized per-query before fusion so
    neither scale dominates."""

    name = "lexical_semantic_fusion"

    def __init__(self, alpha: float):
        self.alpha = alpha  # weight on semantic score

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def fuse(self, lexical_scores: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
        return self.alpha * self._minmax(semantic_scores) + (1 - self.alpha) * self._minmax(lexical_scores)


class SupportPriorMapper:
    """Level 5: fused score + log-scaled prior for how often a head query was
    the correct target in the TRAINING split only (no leakage from val/test)."""

    name = "fusion_plus_support_prior"

    def __init__(self, head_index: List[str], train_positive_counts: dict, beta: float):
        self.beta = beta
        counts = np.array([train_positive_counts.get(h, 0) for h in head_index], dtype=float)
        self.log_prior = np.log1p(counts)
        m = self.log_prior.max()
        self.log_prior = self.log_prior / m if m > 0 else self.log_prior

    def add_prior(self, fused_scores: np.ndarray) -> np.ndarray:
        return (1 - self.beta) * fused_scores + self.beta * self.log_prior
