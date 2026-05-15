from __future__ import annotations

import numpy as np


def ensemble_scores(
    scores: dict[str, dict[str, np.ndarray]],
    exclude_key: str = "ensemble",
) -> dict[str, np.ndarray]:
    """Per-metric mean across all judge arrays, used as ensemble consensus."""
    all_metrics: set[str] = set()
    for model, metrics in scores.items():
        if model != exclude_key:
            all_metrics.update(metrics.keys())
    return {
        metric: np.mean(
            np.stack(
                [scores[m][metric] for m in scores if m != exclude_key and metric in scores[m]]
            ),
            axis=0,
        )
        for metric in all_metrics
    }


def systematic_bias(judge_scores: np.ndarray, reference_scores: np.ndarray) -> float:
    """Mean signed difference. Positive means judge inflates relative to reference."""
    return float(np.mean(judge_scores - reference_scores))


def bias_table(
    scores: dict[str, dict[str, np.ndarray]],
    reference_key: str = "ensemble",
) -> dict[tuple[str, str], float]:
    """Compute bias for each (judge, metric) pair relative to reference.

    scores: {model_name: {metric_name: array_of_scores}}
    Returns: {(model, metric): bias}
    """
    reference = scores[reference_key]
    result: dict[tuple[str, str], float] = {}
    for model, metrics in scores.items():
        if model == reference_key:
            continue
        for metric, judge_arr in metrics.items():
            result[(model, metric)] = systematic_bias(judge_arr, reference[metric])
    return result
