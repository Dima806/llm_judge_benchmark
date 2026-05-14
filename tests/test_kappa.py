import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from src.agreement.kappa import bucketize, cohen_kappa, fleiss_kappa


def test_bucketize_three_regions() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    buckets = bucketize(scores)
    assert list(buckets) == [0, 1, 2]


def test_bucketize_boundary_low() -> None:
    assert bucketize(np.array([0.4]))[0] == 1


def test_bucketize_boundary_high() -> None:
    assert bucketize(np.array([0.7]))[0] == 2


def test_cohen_kappa_perfect_agreement() -> None:
    a = np.array([0.2, 0.5, 0.8, 0.1, 0.6])
    assert cohen_kappa(a, a) == pytest.approx(1.0)


def test_cohen_kappa_matches_sklearn() -> None:
    a = np.array([0.2, 0.5, 0.8, 0.2, 0.5, 0.8])
    b = np.array([0.3, 0.6, 0.9, 0.1, 0.45, 0.75])
    expected = float(cohen_kappa_score(bucketize(a), bucketize(b)))
    assert cohen_kappa(a, b) == pytest.approx(expected)


def test_cohen_kappa_complete_disagreement() -> None:
    a = np.array([0.1, 0.1, 0.9, 0.9])
    b = np.array([0.9, 0.9, 0.1, 0.1])
    assert cohen_kappa(a, b) < 0


def test_fleiss_kappa_perfect_agreement() -> None:
    ratings = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.5, 0.5, 0.5]])
    kappa = fleiss_kappa(ratings)
    assert kappa == pytest.approx(1.0)


def test_fleiss_kappa_range() -> None:
    rng = np.random.default_rng(42)
    ratings = rng.uniform(0, 1, size=(20, 3))
    kappa = fleiss_kappa(ratings)
    assert -1.0 <= kappa <= 1.0
