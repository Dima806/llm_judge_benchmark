from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score as _sklearn_kappa


def bucketize(
    scores: np.ndarray,
    low: float = 0.4,
    high: float = 0.7,
) -> np.ndarray:
    """Map continuous scores to 0 (low), 1 (mid), 2 (high)."""
    buckets = np.zeros(len(scores), dtype=np.int64)
    buckets[scores >= low] = 1
    buckets[scores >= high] = 2
    return buckets


def cohen_kappa(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    low: float = 0.4,
    high: float = 0.7,
) -> float:
    """Cohen's kappa between two raters on bucketed scores."""
    a = bucketize(scores_a, low, high)
    b = bucketize(scores_b, low, high)
    return float(_sklearn_kappa(a, b))


def fleiss_kappa(
    ratings_matrix: np.ndarray,
    low: float = 0.4,
    high: float = 0.7,
) -> float:
    """Fleiss' kappa for multiple raters.

    ratings_matrix: shape (n_items, n_raters), values are continuous scores.
    """
    n_items, n_raters = ratings_matrix.shape
    n_categories = 3

    counts = np.zeros((n_items, n_categories), dtype=np.float64)
    for r in range(n_raters):
        buckets = bucketize(ratings_matrix[:, r], low, high)
        for i in range(n_items):
            counts[i, buckets[i]] += 1

    n = float(n_raters)
    p_j = counts.sum(axis=0) / (n_items * n)
    p_e = float(np.sum(p_j**2))

    p_i = (np.sum(counts**2, axis=1) - n) / (n * (n - 1))
    p_bar = float(np.mean(p_i))

    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)
