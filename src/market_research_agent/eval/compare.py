"""Combine saved eval runs into comparison tables (quality vs cost vs latency)"""

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
    "multi_source_rate": "multi_src",
    "abstain_rate": "abstain",
    "quality_pass_rate": "grader_pass",
    "cost_usd_per_question": "$/question",
    "tokens_per_question": "tokens/q",
    "latency_p50_s": "p50_s",
    "latency_p95_s": "p95_s",
}

# Categories where declining to answer is the expected behaviour.
REFUSAL_OK = {"out_of_scope", "adversarial", "unanswerable"}


def false_refusals(records: list[dict]) -> int:
    """Questions the agent declined although it should have answered them (over-refusal)"""
    return sum(1 for r in records if r.get("refused") and r["category"] not in REFUSAL_OK)


def load_runs(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    rows = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" not in data or "records" not in data:
            continue  # e.g. retrieval_ablation.json
        cfg, summary = data["config"], data["summary"]
        label = f"{data['name']} ({cfg['provider']}/{cfg['model']})"
        row = {new: summary.get(old) for old, new in COLUMNS.items()}
        row["false_refusals"] = false_refusals(data["records"])
        row["n"] = cfg["n_items"]
        rows[label] = row
    return pd.DataFrame(rows).T


def to_markdown(table: pd.DataFrame) -> str:
    """One markdown table per question-set size, largest first"""
    sections = []
    for n in sorted(table["n"].unique(), reverse=True):
        part = table[table["n"] == n].drop(columns="n")
        sections.append(f"### {int(n)} questions\n\n{part.round(3).to_markdown()}\n")
    return "\n".join(sections)


def main() -> None:
    table = load_runs()
    if table.empty:
        raise SystemExit("No eval runs found in eval/results/. Run eval.run_eval first.")
    md = to_markdown(table)
    (RESULTS_DIR / "provider_comparison.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
