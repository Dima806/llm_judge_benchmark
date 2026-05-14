from __future__ import annotations

import re


def parse_score(text: str) -> float:
    """Extract the first numeric value from model output and clamp to [0.0, 1.0]."""
    matches = re.findall(r"\d+(?:\.\d+)?", text.strip())
    if not matches:
        raise ValueError(f"No numeric score found in: {text!r}")
    return max(0.0, min(1.0, float(matches[0])))
