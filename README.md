# LLM Judge Benchmark

Systematic comparison of three local LLM judges against a **synthetic baseline** on 50 RAG
answers. Measures inter-judge agreement (Cohen's kappa), systematic bias per metric, and
calibration correction factors. Runs entirely inside a 2-CPU / 8 GB GitHub Codespace — no
GPU, no cloud APIs.

> **No real human annotators.** `data/human/human_scores.csv` is a synthetic baseline
> generated programmatically. It stands in for human annotation to demonstrate the evaluation
> methodology; it is not real human labels.

## Judge models

| Model | Ollama tag | RAM | Mean divergence vs baseline |
|---|---|---|---|
| Qwen 2.5 1.5B | `qwen2.5:1.5b` | ~1.1 GB | 0.217 |
| Gemma 3 1B | `gemma3:1b` | ~830 MB | 0.205 |
| Llama 3.2 1B | `llama3.2:1b` | ~1.3 GB | 0.284 |

Each judge scores the same 50 RAG answers across three RAG Triad dimensions:
**context relevance**, **groundedness**, and **answer relevance**.

## Quickstart

```bash
make setup     # install uv + Ollama, pull models, sync deps (~5 min)
make test      # unit tests (no Ollama required)
make score-all # run all 3 judges (~37 min)
make lab       # open JupyterLab on :8888
make run       # Streamlit explorer on :8501
```

## Notebooks

Run in order:

| # | Notebook | Description |
|---|---|---|
| 01 | `01_answer_corpus.ipynb` | Build answer corpus and synthetic baseline |
| 02 | `02_judge_scoring.ipynb` | Score all instances with each judge (~37 min) |
| 03 | `03_inter_judge_agreement.ipynb` | Cohen's kappa and correlation across judges |
| 04 | `04_human_vs_model.ipynb` | Bias analysis: judge scores vs synthetic baseline |
| 05 | `05_calibration_and_correction.ipynb` | Linear calibration correction per judge/metric |

## Results

### Mean scores per judge (50 instances)

| Model | context\_relevance | groundedness | answer\_relevance |
|---|---|---|---|
| `qwen2.5:1.5b` | 0.824 | 0.833 | 0.881 |
| `gemma3:1b` | 0.771 | 0.880 | 0.866 |
| `llama3.2:1b` | 0.468 | 0.595 | 0.645 |
| Synthetic baseline | 0.741 | 0.773 | 0.819 |

### Inter-judge agreement (Fleiss' kappa, all 3 judges + baseline)

| Metric | Fleiss κ | Interpretation |
|---|---|---|
| context\_relevance | 0.054 | slight |
| groundedness | 0.215 | fair |
| answer\_relevance | 0.107 | slight |

All values fall well below the 0.60 substantial-agreement threshold, confirming that 1B-class
judges diverge significantly — both from each other and from the synthetic baseline.

### Bias summary

| Model | Inflation rate (judge > baseline) | Large disagreement rate (div > 0.2) |
|---|---|---|
| `qwen2.5:1.5b` | 78% | 33% |
| `gemma3:1b` | 75% | 34% |
| `llama3.2:1b` | 25% | 55% |

`qwen2.5:1.5b` and `gemma3:1b` systematically over-score relative to the baseline.
`llama3.2:1b` deflates scores and has the highest per-instance variance.

### Calibration (linear correction per judge/metric)

Linear correction (`corrected = a * raw + b`) does not reliably close the gap — post-calibration
kappa remains below 0.25 across all (model, metric) pairs and never reaches the 0.60 target.
The low slopes (`a` ≈ 0.07–0.19) indicate that raw judge scores have very low variance relative
to the baseline, making linear rescaling insufficient.

## Key design decisions

- **No framework abstractions** — no LangChain, LlamaIndex, RAGAS, or OpenAI SDK. All Ollama
  calls go through `httpx` directly.
- **Combined scoring** — each judge returns all three metric scores in a single Ollama call
  (one JSON response), reducing API calls by 3×.
- **Dynamic model discovery** — notebooks 03–05 scan `data/eval/scores_*.json` at runtime;
  no hardcoded model lists.
- **NetworkGuard** — rejects any non-localhost URL before an `httpx` call is made.

## Project structure

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
    kappa.py             # Cohen's kappa, Fleiss' kappa
    correlation.py       # Spearman, Pearson, Kendall-tau
    bias.py              # mean(judge_score − baseline_score) per (judge, metric)
  calibration/
    corrector.py         # Linear fit: corrected = a * raw + b
  visualisation.py       # Scatter matrices, bias heatmaps, agreement charts
app/
  streamlit_app.py       # Interactive score explorer
data/
  answers/               # Cached RAG answers (JSON)
  human/human_scores.csv # Synthetic baseline (not real human annotation)
  eval/                  # Judge outputs (generated, gitignored)
config/settings.yaml     # Model names, Ollama URL, scoring thresholds
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (managed automatically by `make setup`)
- [Ollama](https://ollama.com) (installed automatically by `make setup`)
- 8 GB RAM (GitHub Codespace default)
