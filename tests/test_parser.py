import pytest

from src.judging.parser import parse_score


def test_plain_decimal() -> None:
    assert parse_score("0.85") == pytest.approx(0.85)


def test_integer_response() -> None:
    assert parse_score("1") == pytest.approx(1.0)


def test_zero() -> None:
    assert parse_score("0.0") == pytest.approx(0.0)


def test_clamps_above_one() -> None:
    assert parse_score("1.5") == pytest.approx(1.0)


def test_clamps_high_integer() -> None:
    assert parse_score("5") == pytest.approx(1.0)


def test_with_surrounding_text() -> None:
    assert parse_score("The score is 0.75 out of 1.0") == pytest.approx(0.75)


def test_multiple_numbers_takes_first() -> None:
    assert parse_score("0.8 is my answer, not 0.5") == pytest.approx(0.8)


def test_no_number_raises() -> None:
    with pytest.raises(ValueError, match="No numeric score"):
        parse_score("no score here")


def test_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="No numeric score"):
        parse_score("")
