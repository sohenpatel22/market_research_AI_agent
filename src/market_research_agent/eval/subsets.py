"""Smaller, comparable evaluation subsets, and re-scoring a saved run on one without LLM calls"""

import json
from pathlib import Path

from market_research_agent.eval.golden import GoldenItem

RESULTS_DIR = Path("eval/results")


def stratified_subset(items: list[GoldenItem], filing_sample: int | None) -> list[GoldenItem]:
    """All non-filing items plus `filing_sample` evenly spaced filing items (deterministic)"""
    if filing_sample is None:
        return items
    filings = [i for i in items if i.category == "filings"]
    if filing_sample >= len(filings):
        return items
    step = len(filings) / filing_sample
    keep = {filings[int(k * step)].id for k in range(filing_sample)}
    return [i for i in items if i.category != "filings" or i.id in keep]


def derive_run(
    source_name: str, ids: set[str], new_name: str, results_dir: Path = RESULTS_DIR
) -> dict:
    """Write `new_name` = the records of a saved run restricted to `ids`, re-summarised"""
    from market_research_agent.eval.run_eval import summarize

    data = json.loads((results_dir / f"{source_name}.json").read_text(encoding="utf-8"))
    records = [r for r in data["records"] if r["id"] in ids]
    if len(records) != len(ids):
        missing = sorted(ids - {r["id"] for r in records})
        raise ValueError(f"{source_name} has no records for: {missing[:5]}")
    summary = summarize(records)
    summary["judge_cost_usd"] = None
    out = {
        "name": new_name,
        "config": {**data["config"], "n_items": len(records), "derived_from": source_name},
        "summary": summary,
        "records": records,
    }
    (results_dir / f"{new_name}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return out
