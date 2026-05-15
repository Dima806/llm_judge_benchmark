from __future__ import annotations

import json
import re

_MULTI_KEYS: tuple[str, ...] = ("context_relevance", "groundedness", "answer_relevance")


def parse_score(text: str) -> float:
    """Extract the first numeric value from model output and clamp to [0.0, 1.0]."""
    matches = re.findall(r"\d+(?:\.\d+)?", text.strip())
    if not matches:
        raise ValueError(f"No numeric score found in: {text!r}")
    return max(0.0, min(1.0, float(matches[0])))


def parse_multi_score(text: str) -> dict[str, float]:
    """Extract all three RAG Triad scores from a JSON response, clamped to [0.0, 1.0].

    Searches for the first JSON object in the text, so surrounding prose is tolerated.
    """
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            result: dict[str, float] = {}
            for key in _MULTI_KEYS:
                if key not in data:
                    raise ValueError(f"Missing key {key!r} in response: {text!r}")
                result[key] = max(0.0, min(1.0, float(data[key])))
            return result
        except (json.JSONDecodeError, TypeError):
            pass
    raise ValueError(f"Could not parse multi-score JSON from: {text!r}")
