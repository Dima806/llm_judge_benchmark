from __future__ import annotations

import numpy as np


def systematic_bias(judge_scores: np.ndarray, human_scores: np.ndarray) -> float:
    """Mean signed difference. Positive means judge inflates relative to human."""
    return float(np.mean(judge_scores - human_scores))


def bias_table(
    scores: dict[str, dict[str, np.ndarray]],
    human_key: str = "human",
) -> dict[tuple[str, str], float]:
    """Compute bias for each (judge, metric) pair relative to human.

    scores: {model_name: {metric_name: array_of_scores}}
    Returns: {(model, metric): bias}
    """
    human = scores[human_key]
    result: dict[tuple[str, str], float] = {}
    for model, metrics in scores.items():
        if model == human_key:
            continue
        for metric, judge_arr in metrics.items():
            result[(model, metric)] = systematic_bias(judge_arr, human[metric])
    return result
