from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class LinearCorrector:
    """Fits corrected = a * raw + b per (judge, metric) against human scores."""

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, raw_scores: np.ndarray, human_scores: np.ndarray) -> LinearCorrector:
        model = LinearRegression()
        model.fit(raw_scores.reshape(-1, 1), human_scores)
        self.a = float(model.coef_[0])
        self.b = float(model.intercept_)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return np.clip(self.a * raw_scores + self.b, 0.0, 1.0)

    def fit_transform(self, raw_scores: np.ndarray, human_scores: np.ndarray) -> np.ndarray:
        self.fit(raw_scores, human_scores)
        return self.transform(raw_scores)


def correction_table(
    scores: dict[str, dict[str, np.ndarray]],
    human_key: str = "human",
) -> dict[tuple[str, str], LinearCorrector]:
    """Fit one LinearCorrector per (judge, metric) pair.

    scores: {model_name: {metric_name: array_of_scores}}
    """
    human = scores[human_key]
    correctors: dict[tuple[str, str], LinearCorrector] = {}
    for model, metrics in scores.items():
        if model == human_key:
            continue
        for metric, raw_arr in metrics.items():
            correctors[(model, metric)] = LinearCorrector().fit(raw_arr, human[metric])
    return correctors
