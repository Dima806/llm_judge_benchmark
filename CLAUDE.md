# CLAUDE.md — llm_judge_benchmark

## Project

Systematic comparison of three local LLM judges (via Ollama) on 50 RAG answers. Measures
inter-judge agreement (Cohen's kappa), divergence from ensemble consensus, and calibration
correction factors. Runs entirely inside a 2-CPU / 8 GB GitHub Codespace — no GPU, no cloud APIs.

The **ensemble consensus** (mean of all three judges per instance and metric) serves as the
reference for bias and calibration analysis. No external ground truth or human annotation is used.

## Key constraint

**No framework abstractions.** No LangChain, LlamaIndex, RAGAS, or OpenAI SDK. All Ollama calls
go through `httpx` directly. This is deliberate — the project demonstrates the self-evaluation
problem with these frameworks, so it must not depend on them.

## Stack

- **Python 3.11+**, managed with `uv` (frozen lockfile, never use pip directly)
- **Linting:** `ruff` (line length 99, rules E/F/W/I/UP/N/B/A/SIM/PTH)
- **Type checking:** `ty` (not mypy)
- **Tests:** `pytest` — unit tests only, no Ollama required (mock all HTTP)
- **Notebooks:** JupyterLab via `uv run jupyter lab`

## Common commands

```bash
make setup       # first-time: install uv, Ollama, pull models, sync deps
make dev         # lint + test (fast offline loop)
make ci          # sync + lint + test (CI pipeline)
make test        # pytest only
make lint        # ruff format + check + ty typecheck
make score-all   # score all answers with all 3 judges (~37 min)
make run         # Streamlit app on :8501
make lab         # JupyterLab on :8888
```

## Repository layout

```
src/
  config.py              # Pydantic Settings — load from config/settings.yaml
  network_guard.py       # Rejects non-localhost URLs (security boundary)
  judging/
    judge.py             # OllamaJudge: score one (question, context, answer) tuple
    prompts.py           # Prompt templates for all three RAG Triad dimensions
    parser.py            # parse_score() / parse_multi_score(): extract and clamp floats
    runner.py            # Batch scorer: all answers × all judges; CLI entry point
  agreement/
    kappa.py             # Cohen's kappa, Fleiss' kappa (buckets: low<0.4, mid 0.4-0.7, high>0.7)
    correlation.py       # Spearman, Pearson, Kendall-tau on raw scores
    bias.py              # ensemble_scores(), mean(judge − ensemble) per (judge, metric)
  calibration/
    corrector.py         # Linear fit: corrected = a * raw + b, per (judge, metric)
  visualisation.py       # Scatter matrices, bias heatmaps, calibration charts
app/
  streamlit_app.py       # Interactive score explorer
data/
  answers/               # Cached RAG answers (JSON)
  eval/                  # Judge outputs (generated, gitignored)
notebooks/               # 01–05, run sequentially
tests/                   # Unit tests, all mocked
config/settings.yaml     # Model names, Ollama URL, scoring thresholds
```

## Judge models

| Model | Ollama tag | RAM |
|---|---|---|
| Qwen 2.5 7B | `qwen2.5:7b` | ~4.4 GB |
| Gemma 3 4B | `gemma3:4b` | ~3.3 GB |
| Llama 3.1 8B | `llama3.1:8b` | ~4.9 GB |

Models range from 3.3 to 4.9 GB. Load one at a time; the notebook restarts Ollama between
models automatically.

## Scoring protocol

- Each judge receives the same prompt template for each RAG Triad dimension: context relevance,
  groundedness, answer relevance.
- `parse_score()` in `src/judging/parser.py` extracts the float and clamps to [0.0, 1.0]. Use
  it everywhere — never parse scores inline.
- Bucket scores before computing kappa: low < 0.4, mid 0.4–0.7, high > 0.7.

## Ensemble consensus

The **ensemble** is computed at runtime as the per-metric mean of all three loaded judge score
arrays. It is injected into the `scores` dict under key `"ensemble"` before bias and calibration
analysis. No CSV file or external reference is required.

```python
from src.agreement.bias import ensemble_scores
scores["ensemble"] = ensemble_scores(scores)  # {metric: np.ndarray}
```

## NetworkGuard

`src/network_guard.py` must be applied to every URL before any `httpx` call. It rejects anything
that is not `http://localhost` or `http://127.0.0.1`. This prevents accidental external calls.
Do not bypass it.

## Data contracts

Answer instance schema:
```json
{
  "id": "q01_vector",
  "question": "...",
  "context": ["..."],
  "answer": "...",
  "pipeline": "vector|graph|handcrafted",
  "question_type": "single_hop|multi_hop|absence_reasoning"
}
```

Judge scores output (`data/eval/scores_<safe_model>.json`) — one record per (id, model, metric):
```json
{"id": "q01_vector", "model": "qwen2.5:7b", "metric": "context_relevance", "score": 0.82}
```

## Testing

- Mock all Ollama HTTP with `httpx` transport overrides or `pytest` fixtures — never spin up real
  Ollama in tests.
- `test_parser.py`: edge cases — no number found, multiple numbers, out-of-range values, integer
  responses.
- `test_kappa.py`: verify against known Cohen's kappa examples (sklearn reference values).
- `test_judge.py`: verify prompt formatting is correct for each triad dimension.

## What to avoid

- Do not add `torch`, `tensorflow`, `openai`, `tiktoken`, or any GPU package.
- Do not add LangChain, LlamaIndex, RAGAS, or any RAG framework.
- Do not hardcode model names outside `config/settings.yaml` and `src/config.py`.
- Do not write scores to `data/eval/` in tests — use tmp paths.
- Notebook 02 (scoring) takes ~37 min. Do not add `time.sleep` or retry loops that would extend
  this.
- Do not add a `data/human/` directory or synthetic baseline CSV. The ensemble is the reference.
