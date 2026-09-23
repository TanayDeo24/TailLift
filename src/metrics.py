"""Ranking metrics for the tail->head query mapping task. All metrics are
computed per-query then averaged, so paired bootstrap CIs are valid."""
import numpy as np


def rank_of_first_relevant(ranked_heads, relevant_set):
    """1-indexed rank of the first relevant head in the ranked list, or None."""
    for i, h in enumerate(ranked_heads):
        if h in relevant_set:
            return i + 1
    return None


def per_query_metrics(ranked_heads, relevant_set, ks=(1, 5, 10)):
    rank = rank_of_first_relevant(ranked_heads, relevant_set)
    out = {"mrr": 1.0 / rank if rank else 0.0}
    for k in ks:
        out[f"hit@{k}"] = 1.0 if (rank is not None and rank <= k) else 0.0
        out[f"recall@{k}"] = (
            len(set(ranked_heads[:k]) & relevant_set) / len(relevant_set) if relevant_set else 0.0
        )
    return out


def aggregate(list_of_metric_dicts):
    if not list_of_metric_dicts:
        return {}
    keys = list_of_metric_dicts[0].keys()
    return {k: float(np.mean([d[k] for d in list_of_metric_dicts])) for k in keys}


def paired_bootstrap_ci(values_a, values_b, n_resamples=5000, seed=20260922, alpha=0.05):
    """values_b - values_a paired bootstrap CI on the mean delta (b minus a),
    same query order for both. Returns (observed_delta, ci_low, ci_high)."""
    assert len(values_a) == len(values_b)
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    n = len(a)
    observed = float(np.mean(b) - np.mean(a))
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = np.mean(b[idx]) - np.mean(a[idx])
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed, float(lo), float(hi), n, n_resamples, seed
