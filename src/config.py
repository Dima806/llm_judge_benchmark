from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class Settings(BaseModel):
    ollama_url: str = "http://localhost:11434"
    models: list[str] = ["qwen2.5:1.5b", "qwen2.5:3b", "gemma3:4b"]
    score_bucket_low: float = 0.4
    score_bucket_high: float = 0.7
    num_ctx: int = 4096


def load_settings(path: Path = Path("config/settings.yaml")) -> Settings:
    if path.exists():
        data = yaml.safe_load(path.read_text())
        return Settings(**(data or {}))
    return Settings()
