import numpy as np
import pytest

from src.agreement.bias import bias_table, systematic_bias


def test_positive_bias() -> None:
    judge = np.array([0.8, 0.7, 0.9])
    human = np.array([0.6, 0.5, 0.7])
    assert systematic_bias(judge, human) == pytest.approx(0.2)


def test_negative_bias() -> None:
    judge = np.array([0.3, 0.4, 0.5])
    human = np.array([0.5, 0.6, 0.7])
    assert systematic_bias(judge, human) == pytest.approx(-0.2)


def test_zero_bias() -> None:
    a = np.array([0.5, 0.6, 0.7])
    assert systematic_bias(a, a) == pytest.approx(0.0)


def test_bias_table_single_model() -> None:
    scores = {
        "human": {"context_relevance": np.array([0.5, 0.6])},
        "model_a": {"context_relevance": np.array([0.7, 0.8])},
    }
    table = bias_table(scores)
    assert ("model_a", "context_relevance") in table
    assert table[("model_a", "context_relevance")] == pytest.approx(0.2)


def test_bias_table_excludes_human() -> None:
    scores = {
        "human": {"m": np.array([0.5])},
        "model_a": {"m": np.array([0.6])},
    }
    table = bias_table(scores)
    assert ("human", "m") not in table


def test_bias_table_multiple_models() -> None:
    scores = {
        "human": {"m": np.array([0.5, 0.5])},
        "small": {"m": np.array([0.6, 0.6])},
        "large": {"m": np.array([0.4, 0.4])},
    }
    table = bias_table(scores)
    assert table[("small", "m")] == pytest.approx(0.1)
    assert table[("large", "m")] == pytest.approx(-0.1)
