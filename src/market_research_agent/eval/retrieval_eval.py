"""Retrieval-only evaluation and ablation (no LLM calls, so it is free to run)"""

import argparse
import json
from pathlib import Path

import pandas as pd

from market_research_agent.agent.retriever import retrieve
from market_research_agent.eval.golden import GoldenItem, load_golden
from market_research_agent.eval.retrieval_metrics import ndcg_at_k

RESULTS_DIR = Path("eval/results")

CONFIGS = {
    "dense": dict(mode="dense", use_rerank=False),
    "sparse": dict(mode="sparse", use_rerank=False),
    "hybrid": dict(mode="hybrid", use_rerank=False),
    "hybrid+rerank": dict(mode="hybrid", use_rerank=True),
}


def relevance_map(item: GoldenItem) -> dict[tuple[str, int], float]:
    src = item.source
    rel = {(src.accession_number, src.chunk_index): 2.0}
    for neighbour in (src.chunk_index - 1, src.chunk_index + 1):
        rel[(src.accession_number, neighbour)] = 1.0
    return rel


def evaluate_config(items: list[GoldenItem], k: int, **kwargs) -> dict:
    ndcgs, hits, rrs, per_item = [], [], [], {}
    for item in items:
        relevance = relevance_map(item)
        chunks = retrieve(item.question, item.expected_tickers or None, None, k, **kwargs)
        keys = [(c.accession_number, c.chunk_index) for c in chunks]
        ndcg = ndcg_at_k(keys, relevance, k)
        exact = (item.source.accession_number, item.source.chunk_index)
        rank = keys.index(exact) + 1 if exact in keys else None
        ndcgs.append(ndcg)
        hits.append(float(rank is not None))
        rrs.append(1 / rank if rank else 0.0)
        per_item[item.id] = {"ndcg": round(ndcg, 4), "rank": rank}
    n = len(items)
    return {
        f"ndcg@{k}": sum(ndcgs) / n,
        f"recall@{k}": sum(hits) / n,
        "mrr": sum(rrs) / n,
        "per_item": per_item,
    }


def run(k: int = 6, rerank: bool = True) -> pd.DataFrame:
    items = [i for i in load_golden(categories={"filings"}) if i.source]
    rows, detail = {}, {}
    for name, kwargs in CONFIGS.items():
        if kwargs["use_rerank"] and not rerank:
            continue
        result = evaluate_config(items, k, **kwargs)
        detail[name] = result.pop("per_item")
        rows[name] = result
    table = pd.DataFrame(rows).T
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_ablation.json").write_text(
        json.dumps(
            {
                "k": k,
                "n_questions": len(items),
                "summary": table.to_dict("index"),
                "per_item": detail,
            },
            indent=2,
        )
    )
    (RESULTS_DIR / "retrieval_ablation.md").write_text(
        f"Retrieval ablation on {len(items)} filing questions (k={k})\n\n"
        + table.round(3).to_markdown()
        + "\n"
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    print(run(args.k, rerank=not args.no_rerank).round(3).to_string())


if __name__ == "__main__":
    main()
