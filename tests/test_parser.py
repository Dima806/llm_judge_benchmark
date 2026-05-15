import pytest

from src.judging.parser import parse_multi_score, parse_score


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


# --- parse_multi_score ---


def test_multi_clean_json() -> None:
    text = '{"context_relevance": 0.8, "groundedness": 0.9, "answer_relevance": 0.7}'
    result = parse_multi_score(text)
    assert result == pytest.approx(
        {"context_relevance": 0.8, "groundedness": 0.9, "answer_relevance": 0.7}
    )


def test_multi_json_embedded_in_prose() -> None:
    text = 'Here are the scores: {"context_relevance": 0.6, "groundedness": 0.5, "answer_relevance": 0.4} done.'
    result = parse_multi_score(text)
    assert result["context_relevance"] == pytest.approx(0.6)
    assert result["groundedness"] == pytest.approx(0.5)
    assert result["answer_relevance"] == pytest.approx(0.4)


def test_multi_values_clamped() -> None:
    text = '{"context_relevance": 1.5, "groundedness": -0.2, "answer_relevance": 0.5}'
    result = parse_multi_score(text)
    assert result["context_relevance"] == pytest.approx(1.0)
    assert result["groundedness"] == pytest.approx(0.0)
    assert result["answer_relevance"] == pytest.approx(0.5)


def test_multi_missing_key_raises() -> None:
    text = '{"context_relevance": 0.8, "groundedness": 0.7}'
    with pytest.raises(ValueError, match="Missing key"):
        parse_multi_score(text)


def test_multi_no_json_raises() -> None:
    with pytest.raises(ValueError, match="Could not parse"):
        parse_multi_score("I give it a 0.8 overall.")
