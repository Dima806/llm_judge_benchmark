from __future__ import annotations

import numpy as np
from scipy import stats


def spearman(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """Returns (correlation, p_value)."""
    result = stats.spearmanr(scores_a, scores_b)
    return float(result.statistic), float(result.pvalue)


def pearson(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """Returns (correlation, p_value)."""
    result = stats.pearsonr(scores_a, scores_b)
    return float(result.statistic), float(result.pvalue)


def kendall_tau(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """Returns (tau, p_value)."""
    result = stats.kendalltau(scores_a, scores_b)
    return float(result.statistic), float(result.pvalue)
