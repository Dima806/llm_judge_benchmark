import httpx
import pytest

from src.judging.judge import OllamaJudge
from src.network_guard import validate_url

_MULTI_RESPONSE = '{"context_relevance": 0.8, "groundedness": 0.9, "answer_relevance": 0.7}'


class _MockTransport(httpx.BaseTransport):
    def __init__(self, score: str = "0.85") -> None:
        self.score = score
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return httpx.Response(200, json={"response": self.score})


def test_score_returns_float() -> None:
    transport = _MockTransport("0.85")
    judge = OllamaJudge(model="test-model", transport=transport)
    result = judge.score(
        metric="context_relevance",
        question="What is X?",
        context="X is Y.",
        answer="X is Y.",
    )
    assert result == pytest.approx(0.85)


def test_context_as_list_joined() -> None:
    transport = _MockTransport("0.7")
    judge = OllamaJudge(model="test-model", transport=transport)
    result = judge.score(
        metric="groundedness",
        question="What is X?",
        context=["X is Y.", "Y is Z."],
        answer="X is Y and Z.",
    )
    assert result == pytest.approx(0.7)


def test_all_three_metrics() -> None:
    for metric in ["context_relevance", "groundedness", "answer_relevance"]:
        transport = _MockTransport("0.6")
        judge = OllamaJudge(model="test-model", transport=transport)
        result = judge.score(metric=metric, question="Q?", context="C.", answer="A.")
        assert 0.0 <= result <= 1.0


def test_invalid_metric_raises() -> None:
    transport = _MockTransport("0.5")
    judge = OllamaJudge(model="test-model", transport=transport)
    with pytest.raises(ValueError, match="Unknown metric"):
        judge.score(metric="made_up", question="Q", context="C", answer="A")


def test_prompt_sent_to_ollama() -> None:
    transport = _MockTransport("0.9")
    judge = OllamaJudge(model="test-model", transport=transport)
    judge.score(metric="context_relevance", question="Q?", context="C.", answer="A.")
    assert transport.last_request is not None
    body = transport.last_request.content.decode()
    assert "test-model" in body
    assert "Q?" in body


def test_score_all_metrics_returns_three_keys() -> None:
    transport = _MockTransport(_MULTI_RESPONSE)
    judge = OllamaJudge(model="test-model", transport=transport)
    result = judge.score_all_metrics(question="What is X?", context="X is Y.", answer="X is Y.")
    assert set(result.keys()) == {"context_relevance", "groundedness", "answer_relevance"}


def test_score_all_metrics_values_in_range() -> None:
    transport = _MockTransport(_MULTI_RESPONSE)
    judge = OllamaJudge(model="test-model", transport=transport)
    result = judge.score_all_metrics(question="Q?", context="C.", answer="A.")
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_score_all_metrics_single_call() -> None:
    transport = _MockTransport(_MULTI_RESPONSE)
    judge = OllamaJudge(model="test-model", transport=transport)
    judge.score_all_metrics(question="Q?", context="C.", answer="A.")
    assert transport.last_request is not None
    body = transport.last_request.content.decode()
    assert "context_relevance" in body
    assert "groundedness" in body
    assert "answer_relevance" in body


def test_score_all_metrics_context_as_list() -> None:
    transport = _MockTransport(_MULTI_RESPONSE)
    judge = OllamaJudge(model="test-model", transport=transport)
    result = judge.score_all_metrics(question="Q?", context=["Part 1.", "Part 2."], answer="A.")
    assert result["groundedness"] == pytest.approx(0.9)


def test_network_guard_rejects_external_url() -> None:
    with pytest.raises(ValueError, match="NetworkGuard"):
        validate_url("https://api.openai.com/v1")


def test_network_guard_accepts_localhost() -> None:
    assert validate_url("http://localhost:11434") == "http://localhost:11434"
    assert validate_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
