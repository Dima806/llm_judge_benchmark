from __future__ import annotations

import httpx
from loguru import logger

from src.judging.parser import parse_score
from src.judging.prompts import PROMPTS
from src.network_guard import validate_url


class OllamaJudge:
    def __init__(
        self,
        model: str,
        ollama_url: str = "http://localhost:11434",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        validate_url(ollama_url)
        self.model = model
        self.ollama_url = ollama_url
        self._transport = transport

    def score(
        self,
        metric: str,
        question: str,
        context: str | list[str],
        answer: str,
    ) -> float:
        if metric not in PROMPTS:
            raise ValueError(f"Unknown metric: {metric!r}. Choose from {list(PROMPTS)}")
        if isinstance(context, list):
            context = "\n".join(context)
        prompt = PROMPTS[metric].format(question=question, context=context, answer=answer)
        raw = self._call_ollama(prompt)
        result = parse_score(raw)
        logger.debug(f"{self.model} | {metric} | raw={raw!r} | score={result:.3f}")
        return result

    def _call_ollama(self, prompt: str) -> str:
        if self._transport is not None:
            client = httpx.Client(
                base_url=self.ollama_url, timeout=120.0, transport=self._transport
            )
        else:
            client = httpx.Client(base_url=self.ollama_url, timeout=120.0)
        with client:
            response = client.post(
                "/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return str(response.json()["response"])
