import numpy as np
import pytest

from src.agreement.correlation import kendall_tau, pearson, spearman


def test_spearman_perfect_positive() -> None:
    a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    corr, _ = spearman(a, a)
    assert corr == pytest.approx(1.0)


def test_spearman_perfect_negative() -> None:
    a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    corr, _ = spearman(a, a[::-1])
    assert corr == pytest.approx(-1.0)


def test_pearson_perfect_positive() -> None:
    a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    corr, _ = pearson(a, a)
    assert corr == pytest.approx(1.0)


def test_pearson_linear_transform() -> None:
    a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    b = 2 * a + 0.1
    corr, _ = pearson(a, b)
    assert corr == pytest.approx(1.0)


def test_kendall_perfect_positive() -> None:
    a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    tau, _ = kendall_tau(a, a)
    assert tau == pytest.approx(1.0)


def test_all_return_pvalue() -> None:
    a = np.array([0.2, 0.4, 0.6, 0.8])
    for fn in [spearman, pearson, kendall_tau]:
        _, pval = fn(a, a)
        assert 0.0 <= pval <= 1.0
