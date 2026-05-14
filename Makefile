.PHONY: help setup sync lint format check typecheck test \
        score-all score-1.5b score-3b score-4b run lab clean reset ci dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup: install uv, Ollama, pull models, sync deps
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh
	uv sync --all-extras
	@pgrep -x ollama >/dev/null 2>&1 || (ollama serve &>/dev/null & echo "Started ollama serve")
	@echo "Waiting for Ollama to be ready..."; \
	for i in $$(seq 1 30); do \
		curl -sf http://localhost:11434 >/dev/null 2>&1 && echo "Ollama is ready." && break; \
		[ $$i -eq 30 ] && echo "Ollama did not start in time." && exit 1; \
		sleep 1; \
	done
	ollama pull qwen2.5:1.5b
	ollama pull qwen2.5:3b
	ollama pull gemma3:4b
	uv run python -m ipykernel install --user --name llm-judge
	@echo "\n✅ Ready. Run 'make test' to verify."

sync: ## Sync deps from lockfile
	uv sync --all-extras

lint: format check typecheck ## Run all linters

format: ## Auto-format with ruff
	uv run ruff format src/ tests/ app/

check: ## Lint and auto-fix with ruff
	uv run ruff check --fix src/ tests/ app/

typecheck: ## Type-check with ty
	uv run ty check src/

test: ## Run pytest (unit tests, no Ollama required)
	uv run pytest

score-1.5b: ## Score all answers with qwen2.5:1.5b (~15 min)
	uv run python -m src.judging.runner --model qwen2.5:1.5b

score-3b: ## Score all answers with qwen2.5:3b (~20 min)
	uv run python -m src.judging.runner --model qwen2.5:3b

score-4b: ## Score all answers with gemma3:4b (~30 min)
	uv run python -m src.judging.runner --model gemma3:4b

score-all: score-1.5b score-3b score-4b ## Score with all 3 judges sequentially (~65 min)

run: ## Launch Streamlit app
	uv run streamlit run app/streamlit_app.py --server.port 8501

lab: ## Launch JupyterLab
	uv run jupyter lab --no-browser --port 8888

clean: ## Remove caches and eval outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache htmlcov .coverage data/eval/
	@echo "🧹 Cleaned."

reset: clean ## Full reset: also remove virtualenv
	rm -rf .venv
	@echo "🔄 Reset."

ci: sync lint test ## CI pipeline (no Ollama needed)

dev: lint test ## Fast offline loop
