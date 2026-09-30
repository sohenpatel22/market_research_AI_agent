"""Combine saved eval runs into one comparison table (quality vs cost vs latency).

Usage:
    uv run python -m market_research_agent.eval.compare
"""

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("eval/results")

COLUMNS = {
    "ragas_faithfulness": "faithfulness",
    "ragas_answer_relevancy": "answer_rel",
    "ragas_context_precision": "ctx_precision",
    "ragas_context_recall": "ctx_recall",
    "refusal_accuracy": "refusal_acc",
    "data_fact_rate": "data_fact",
    "citation_rate": "cited",
    "quality_pass_rate": "grader_pass",
    "cost_usd_per_question": "$/question",
    "tokens_per_question": "tokens/q",
    "latency_p50_s": "p50_s",
    "latency_p95_s": "p95_s",
}


def load_runs(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    rows = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" not in data or "records" not in data:
            continue  # e.g. retrieval_ablation.json
        cfg, summary = data["config"], data["summary"]
        label = f"{data['name']} ({cfg['provider']}/{cfg['model']}, n={cfg['n_items']})"
        rows[label] = {new: summary.get(old) for old, new in COLUMNS.items()}
    return pd.DataFrame(rows).T


def main() -> None:
    table = load_runs()
    if table.empty:
        raise SystemExit("No eval runs found in eval/results/. Run eval.run_eval first.")
    md = table.round(3).to_markdown()
    (RESULTS_DIR / "provider_comparison.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
