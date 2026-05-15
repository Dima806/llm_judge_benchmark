from __future__ import annotations

import time

import httpx
from loguru import logger

from src.judging.parser import parse_multi_score, parse_score
from src.judging.prompts import COMBINED_PROMPT, PROMPTS
from src.network_guard import validate_url

_OLLAMA_HEALTH_POLL_INTERVAL = 5.0  # seconds between /api/tags polls
_OLLAMA_HEALTH_MAX_WAIT = 90.0  # maximum seconds to wait for Ollama to recover


class OllamaJudge:
    def __init__(
        self,
        model: str,
        ollama_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        validate_url(ollama_url)
        self.model = model
        self.ollama_url = ollama_url
        self.num_ctx = num_ctx
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

    def score_all_metrics(
        self,
        question: str,
        context: str | list[str],
        answer: str,
    ) -> dict[str, float]:
        """Score all three RAG Triad dimensions in a single Ollama call.

        Falls back to three individual calls if the combined call fails after all retries
        (individual prompts are smaller and survive marginal memory situations better).
        """
        if isinstance(context, list):
            context = "\n".join(context)
        prompt = COMBINED_PROMPT.format(question=question, context=context, answer=answer)
        try:
            raw = self._call_ollama(prompt)
            result = parse_multi_score(raw)
            logger.debug(f"{self.model} | combined | raw={raw!r} | scores={result}")
            return result
        except Exception as exc:
            logger.warning(
                f"{self.model} | combined call failed ({exc}), falling back to 3 individual calls"
            )
            return {
                metric: self.score(
                    metric=metric, question=question, context=context, answer=answer
                )
                for metric in PROMPTS
            }

    def warm_up(self, timeout: float = 120.0) -> bool:
        """Load the model into memory with a minimal prompt. Returns True if ready.

        Separates model loading from real scoring so the first instance isn't penalised
        by a 65-second load time and failures are detected before the loop starts.
        """
        if self._transport is not None:
            return True  # mock transport — always ready
        logger.info(f"{self.model} | warming up (model load may take up to {timeout:.0f}s)...")
        try:
            with httpx.Client(base_url=self.ollama_url, timeout=timeout) as client:
                response = client.post(
                    "/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "hi",
                        "stream": False,
                        "keep_alive": "1h",
                        "options": {"num_ctx": 128, "num_predict": 1},
                    },
                )
                if response.is_success:
                    logger.info(f"{self.model} | warm-up complete, model is ready")
                    return True
                logger.warning(
                    f"{self.model} | warm-up HTTP {response.status_code}: {response.text[:300]!r}"
                )
                return False
        except Exception as exc:
            logger.warning(f"{self.model} | warm-up failed: {exc}")
            return False

    def _wait_for_ollama(self) -> None:
        """Poll /api/tags until Ollama responds or the timeout is reached."""
        if self._transport is not None:
            return  # mock transport — nothing to poll
        deadline = time.monotonic() + _OLLAMA_HEALTH_MAX_WAIT
        while time.monotonic() < deadline:
            try:
                with httpx.Client(base_url=self.ollama_url, timeout=5.0) as client:
                    client.get("/api/tags")
                return
            except Exception:
                time.sleep(_OLLAMA_HEALTH_POLL_INTERVAL)
        logger.warning(
            f"{self.model} | Ollama did not recover within {_OLLAMA_HEALTH_MAX_WAIT:.0f}s"
        )

    def _call_ollama(self, prompt: str) -> str:
        kwargs: dict = {"base_url": self.ollama_url, "timeout": 300.0}
        if self._transport is not None:
            kwargs["transport"] = self._transport

        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(3):
            if attempt > 0:
                logger.warning(f"{self.model} | retry {attempt}/2: {last_exc}")
                self._wait_for_ollama()  # wait exactly until Ollama is responsive again
            try:
                with httpx.Client(**kwargs) as client:
                    response = client.post(
                        "/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "keep_alive": "1h",
                            "options": {"num_ctx": self.num_ctx},
                        },
                    )
                    if not response.is_success:
                        logger.error(
                            f"{self.model} | HTTP {response.status_code}: {response.text[:300]!r}"
                        )
                    response.raise_for_status()
                    return str(response.json()["response"])
            except Exception as exc:
                last_exc = exc

        raise last_exc
