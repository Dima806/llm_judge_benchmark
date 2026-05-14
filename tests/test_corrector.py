import numpy as np
import pytest

from src.calibration.corrector import LinearCorrector, correction_table


def test_fit_identity() -> None:
    raw = np.array([0.3, 0.5, 0.7, 0.9])
    corrector = LinearCorrector().fit(raw, raw)
    assert corrector.a == pytest.approx(1.0, abs=1e-6)
    assert corrector.b == pytest.approx(0.0, abs=1e-6)


def test_transform_applies_correction() -> None:
    corrector = LinearCorrector()
    corrector.a = 0.5
    corrector.b = 0.1
    raw = np.array([0.6])
    result = corrector.transform(raw)
    assert result[0] == pytest.approx(0.5 * 0.6 + 0.1)


def test_transform_clamps_above_one() -> None:
    corrector = LinearCorrector()
    corrector.a = 2.0
    corrector.b = 0.0
    result = corrector.transform(np.array([0.8]))
    assert result[0] == pytest.approx(1.0)


def test_transform_clamps_below_zero() -> None:
    corrector = LinearCorrector()
    corrector.a = 1.0
    corrector.b = -1.0
    result = corrector.transform(np.array([0.3]))
    assert result[0] == pytest.approx(0.0)


def test_fit_transform_roundtrip() -> None:
    raw = np.array([0.2, 0.4, 0.6, 0.8])
    human = np.array([0.3, 0.5, 0.7, 0.9])
    corrector = LinearCorrector()
    corrected = corrector.fit_transform(raw, human)
    assert corrected.shape == raw.shape
    assert all(0.0 <= v <= 1.0 for v in corrected)


def test_correction_table_keys() -> None:
    scores = {
        "human": {"metric_a": np.array([0.4, 0.6, 0.8])},
        "model_x": {"metric_a": np.array([0.5, 0.7, 0.9])},
    }
    table = correction_table(scores)
    assert ("model_x", "metric_a") in table
    assert isinstance(table[("model_x", "metric_a")], LinearCorrector)


def test_correction_table_excludes_human() -> None:
    scores = {
        "human": {"m": np.array([0.5, 0.6])},
        "model": {"m": np.array([0.6, 0.7])},
    }
    table = correction_table(scores)
    assert all(k[0] != "human" for k in table)
