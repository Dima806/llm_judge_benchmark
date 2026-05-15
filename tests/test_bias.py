import numpy as np
import pytest

from src.agreement.bias import bias_table, ensemble_scores, systematic_bias


def test_positive_bias() -> None:
    judge = np.array([0.8, 0.7, 0.9])
    reference = np.array([0.6, 0.5, 0.7])
    assert systematic_bias(judge, reference) == pytest.approx(0.2)


def test_negative_bias() -> None:
    judge = np.array([0.3, 0.4, 0.5])
    reference = np.array([0.5, 0.6, 0.7])
    assert systematic_bias(judge, reference) == pytest.approx(-0.2)


def test_zero_bias() -> None:
    a = np.array([0.5, 0.6, 0.7])
    assert systematic_bias(a, a) == pytest.approx(0.0)


def test_bias_table_single_model() -> None:
    scores = {
        "ensemble": {"context_relevance": np.array([0.5, 0.6])},
        "model_a": {"context_relevance": np.array([0.7, 0.8])},
    }
    table = bias_table(scores)
    assert ("model_a", "context_relevance") in table
    assert table[("model_a", "context_relevance")] == pytest.approx(0.2)


def test_bias_table_excludes_ensemble() -> None:
    scores = {
        "ensemble": {"m": np.array([0.5])},
        "model_a": {"m": np.array([0.6])},
    }
    table = bias_table(scores)
    assert ("ensemble", "m") not in table


def test_bias_table_multiple_models() -> None:
    scores = {
        "ensemble": {"m": np.array([0.5, 0.5])},
        "small": {"m": np.array([0.6, 0.6])},
        "large": {"m": np.array([0.4, 0.4])},
    }
    table = bias_table(scores)
    assert table[("small", "m")] == pytest.approx(0.1)
    assert table[("large", "m")] == pytest.approx(-0.1)


def test_ensemble_scores_mean() -> None:
    scores = {
        "model_a": {"m": np.array([0.4, 0.8])},
        "model_b": {"m": np.array([0.6, 0.4])},
    }
    ens = ensemble_scores(scores)
    assert ens["m"] == pytest.approx(np.array([0.5, 0.6]))


def test_ensemble_scores_excludes_existing_ensemble_key() -> None:
    scores = {
        "model_a": {"m": np.array([0.4])},
        "model_b": {"m": np.array([0.6])},
        "ensemble": {"m": np.array([0.99])},  # stale — should be excluded
    }
    ens = ensemble_scores(scores)
    assert ens["m"] == pytest.approx(np.array([0.5]))
