# LLM Judge Benchmark

Systematic comparison of three local LLM judges on 50 RAG answers. Measures inter-judge
agreement (Cohen's kappa), divergence from ensemble consensus, and calibration correction
factors. Runs entirely inside a 2-CPU / 8 GB GitHub Codespace — no GPU, no cloud APIs.

The **ensemble consensus** (mean of all three judges per instance and metric) serves as
the reference for divergence and calibration. No external ground truth is used.

## Judge models

| Model | Ollama tag | RAM | Mean divergence from ensemble |
|---|---|---|---|
| Qwen 2.5 7B | `qwen2.5:7b` | ~4.4 GB | 0.099 |
| Gemma 3 4B | `gemma3:4b` | ~3.3 GB | 0.086 |
| Llama 3.1 8B | `llama3.1:8b` | ~4.9 GB | 0.120 |

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
| 01 | `01_answer_corpus.ipynb` | Build answer corpus |
| 02 | `02_judge_scoring.ipynb` | Score all instances with each judge (~37 min) |
| 03 | `03_inter_judge_agreement.ipynb` | Cohen's kappa and correlation across judges |
| 04 | `04_consensus_vs_individual.ipynb` | Consensus vs Individual: divergence from ensemble mean |
| 05 | `05_calibration_and_correction.ipynb` | Linear calibration toward ensemble consensus |

## Results

### Mean scores per judge (50 instances)

| Model | context\_relevance | groundedness | answer\_relevance |
|---|---|---|---|
| `qwen2.5:7b` | 0.688 | 0.762 | 0.852 |
| `gemma3:4b` | 0.878 | 0.824 | 0.872 |
| `llama3.1:8b` | 0.868 | 0.786 | 0.810 |
| Ensemble mean | 0.811 | 0.791 | 0.845 |

### Inter-judge agreement (Fleiss' kappa, 3 judges)

| Metric | Fleiss κ | Interpretation |
|---|---|---|
| context\_relevance | 0.094 | slight |
| groundedness | 0.633 | substantial |
| answer\_relevance | 0.336 | fair |

Groundedness reaches substantial agreement (κ = 0.63) among the three judges. Context relevance
shows only slight agreement (κ = 0.09), indicating the judges diverge most on retrieval quality.

### Divergence from ensemble consensus

| Model | Inflation rate (judge > ensemble) | Large disagreement rate (div > 0.2) |
|---|---|---|
| `qwen2.5:7b` | 23% | 15% |
| `gemma3:4b` | 59% | 13% |
| `llama3.1:8b` | 38% | 19% |

`gemma3:4b` over-scores relative to the ensemble most often (59%). `llama3.1:8b` has the
highest mean divergence (0.120) and the highest large-disagreement rate (19%).

### Calibration (linear correction toward ensemble)

Linear correction (`corrected = a * raw + b`) toward the ensemble mean effectively reduces
per-judge idiosyncrasies for `gemma3:4b` and `qwen2.5:7b`, which reach κ ≥ 0.60 across all
metrics after calibration. `llama3.1:8b` remains below the threshold — it diverges more
systematically from the consensus and requires a larger correction.

| Model | Metrics reaching κ ≥ 0.60 after calibration |
|---|---|
| `gemma3:4b` | 3/3 |
| `qwen2.5:7b` | 3/3 |
| `llama3.1:8b` | 0/3 |

## Key design decisions

- **No framework abstractions** — no LangChain, LlamaIndex, RAGAS, or OpenAI SDK. All Ollama
  calls go through `httpx` directly.
- **Ensemble as reference** — the mean of all three judges replaces any synthetic baseline.
  Bias and calibration are measured against this consensus, not an external ground truth.
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
    bias.py              # ensemble_scores(), mean(judge − ensemble) per (judge, metric)
  calibration/
    corrector.py         # Linear fit: corrected = a * raw + b
  visualisation.py       # Scatter matrices, bias heatmaps, calibration charts
app/
  streamlit_app.py       # Interactive score explorer
data/
  answers/               # Cached RAG answers (JSON)
  eval/                  # Judge outputs (generated, gitignored)
config/settings.yaml     # Model names, Ollama URL, scoring thresholds
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (managed automatically by `make setup`)
- [Ollama](https://ollama.com) (installed automatically by `make setup`)
- 8 GB RAM (GitHub Codespace default)
