from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from src.config import load_settings
from src.judging.judge import OllamaJudge

METRICS: list[str] = ["context_relevance", "groundedness", "answer_relevance"]


def score_all(model: str, answers_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    settings = load_settings()
    judge = OllamaJudge(model=model, ollama_url=settings.ollama_url, num_ctx=settings.num_ctx)
    results: list[dict[str, object]] = []

    for path in sorted(answers_dir.glob("*.json")):
        instances = json.loads(path.read_text())
        if isinstance(instances, dict):
            instances = [instances]
        for instance in instances:
            scores = judge.score_all_metrics(
                question=str(instance["question"]),
                context=instance["context"],
                answer=str(instance["answer"]),
            )
            for metric, score in scores.items():
                record: dict[str, object] = {
                    "id": instance["id"],
                    "model": model,
                    "metric": metric,
                    "score": score,
                }
                results.append(record)
                logger.info(f"{instance['id']} | {model} | {metric} = {score:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "_").replace(".", "_")
    out_path = output_dir / f"scores_{safe_model}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved {len(results)} records to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch LLM judge scoring")
    parser.add_argument("--model", required=True, help="Ollama model tag")
    parser.add_argument("--answers-dir", default="data/answers", help="Answer JSON directory")
    parser.add_argument("--output-dir", default="data/eval", help="Score output directory")
    args = parser.parse_args()
    score_all(
        model=args.model,
        answers_dir=Path(args.answers_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
