"""Run the agent over the golden set and score it. This is the dev-loop eval (costs LLM tokens).

Scores per item:
  * deterministic checks (refusal, tool use, numeric facts, citations) - free
  * RAGAS faithfulness / answer_relevancy / context_precision / context_recall (filing questions)
plus tokens, estimated cost and latency for the agent under test. Results go to
eval/results/<name>.json and, when Langfuse is configured, to a Langfuse dataset run.

Usage:
    uv run python -m market_research_agent.eval.run_eval --name deepseek-flash
    uv run python -m market_research_agent.eval.run_eval --name quick --limit 6 --no-ragas
    uv run python -m market_research_agent.eval.run_eval --name gpt --provider openai \
        --model gpt-4o-mini

For fair cost/latency numbers keep LLM_CACHE=false; enable it to re-score cheaply.
"""

import argparse
import functools
import json
import time
from contextlib import suppress
from pathlib import Path

import pandas as pd
from langchain_core.callbacks import UsageMetadataCallbackHandler

from market_research_agent.agent.graph import Dependencies
from market_research_agent.agent.retriever import retrieve
from market_research_agent.agent.service import ask
from market_research_agent.agent.tools import forecast_tool, sql_tool
from market_research_agent.config import settings
from market_research_agent.eval.checks import deterministic_checks, load_thresholds, rate
from market_research_agent.eval.golden import GoldenItem, load_golden
from market_research_agent.llm.factory import get_chat_model, get_judge_model
from market_research_agent.llm.pricing import estimate_cost, total_tokens
from market_research_agent.observability import langfuse as lf
from market_research_agent.observability.langfuse import git_sha

RESULTS_DIR = Path("eval/results")
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_deps(
    provider: str | None, model: str | None, max_retries: int, rerank: bool, top_k: int = 6
) -> Dependencies:
    provider = provider or settings.llm_provider
    return Dependencies(
        llm=get_chat_model(provider, model),
        judge=get_judge_model(),
        retrieve=functools.partial(retrieve, use_rerank=rerank),
        forecast=forecast_tool,
        run_sql=sql_tool,
        provider=provider,
        judge_provider=settings.judge_provider or settings.llm_provider,
        max_retries=max_retries,
        top_k=top_k,
    )


def run_items(items: list[GoldenItem], deps: Dependencies, run_name: str) -> list[dict]:
    records = []
    for n, item in enumerate(items, start=1):
        usage = UsageMetadataCallbackHandler()
        start = time.perf_counter()
        # The usage handler is attached to the models so it only counts the agent, not RAGAS.
        deps.llm.callbacks = [usage]
        deps.judge.callbacks = [usage]
        traced = ask(
            item.question,
            session_id=f"eval-{run_name}",
            deps=deps,
            tags=["eval", run_name, item.category],
            flush_now=False,
        )
        latency = time.perf_counter() - start
        a = traced.answer
        records.append(
            {
                "id": item.id,
                "category": item.category,
                "question": item.question,
                "answer": a.answer,
                "reference": item.reference_answer,
                "contexts": [c.text for c in a.retrieved_context],
                "refused": a.refused,
                "quality_passed": a.quality_passed,
                "grade_score": a.grade_score,
                "retries": a.retries,
                "n_sources": len(a.sources),
                "forecast_horizons": [f.horizon for f in a.forecasts],
                "n_data": len(a.data),
                "tool_errors": a.tool_errors,
                "tokens": total_tokens(usage.usage_metadata),
                "cost_usd": estimate_cost(usage.usage_metadata),
                "latency_s": round(latency, 2),
                "trace_id": traced.trace_id,
                "checks": deterministic_checks(item, a),
            }
        )
        print(
            f"[{n}/{len(items)}] {item.id}: refused={a.refused} passed={a.quality_passed} "
            f"retries={a.retries} tokens={records[-1]['tokens']} {latency:.1f}s",
            flush=True,
        )
    return records


def score_ragas(records: list[dict], judge_usage: UsageMetadataCallbackHandler) -> None:
    """Attach RAGAS scores (in place) to filing answers that have retrieved context."""
    from market_research_agent.eval.judges import ragas_judge

    llm, embeddings = ragas_judge(callbacks=[judge_usage])
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    targets = [
        r for r in records if r["category"] == "filings" and r["contexts"] and not r["refused"]
    ]
    if not targets:
        return
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["reference"],
            )
            for r in targets
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(strictness=1),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=4, timeout=180),
        raise_exceptions=False,
        show_progress=False,
    )
    frame = result.to_pandas()
    for record, (_, row) in zip(targets, frame.iterrows(), strict=True):
        record["ragas"] = {
            m: (None if pd.isna(row.get(m)) else float(row[m])) for m in RAGAS_METRICS if m in row
        }


def summarize(records: list[dict]) -> dict:
    checks = [r["checks"] for r in records]
    summary: dict = {
        "refusal_accuracy": rate(checks, "refusal_correct"),
        "forecast_tool_rate": rate(checks, "forecast_tool"),
        "data_tool_rate": rate(checks, "data_tool"),
        "data_fact_rate": rate(checks, "data_fact"),
        "citation_rate": rate(checks, "cited"),
        "multi_source_rate": rate(checks, "multi_source"),
        "abstain_rate": rate(checks, "abstained"),
        "quality_pass_rate": sum(r["quality_passed"] for r in records) / len(records),
        "mean_retries": sum(r["retries"] for r in records) / len(records),
    }
    for metric in RAGAS_METRICS:
        vals = [r["ragas"][metric] for r in records if r.get("ragas", {}).get(metric) is not None]
        summary[f"ragas_{metric}"] = sum(vals) / len(vals) if vals else None
    summary["cost_usd_total"] = sum(r["cost_usd"] for r in records)
    summary["cost_usd_per_question"] = summary["cost_usd_total"] / len(records)
    summary["tokens_per_question"] = sum(r["tokens"] for r in records) / len(records)
    lat = sorted(r["latency_s"] for r in records)
    summary["latency_p50_s"] = lat[len(lat) // 2]
    summary["latency_p95_s"] = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
    return summary


def compare_to_thresholds(summary: dict, thresholds: dict) -> list[str]:
    """Human-readable list of failed thresholds (empty = all good)."""
    failures = []
    for metric, minimum in thresholds["ragas"].items():
        got = summary.get(f"ragas_{metric}")
        if got is not None and got < minimum:
            failures.append(f"ragas {metric}: {got:.3f} < {minimum}")
    for metric, minimum in thresholds["deterministic"].items():
        got = summary.get(metric)
        if got is not None and got < minimum:
            failures.append(f"{metric}: {got:.3f} < {minimum}")
    return failures


def log_to_langfuse(run_name: str, items: list[GoldenItem], records: list[dict]) -> None:
    client = lf.get_client()
    if client is None:
        return
    dataset = "market-research-agent-golden"
    with suppress(Exception):  # already exists
        client.create_dataset(name=dataset, description="Golden Q&A set for the research agent")
    by_id = {i.id: i for i in items}
    for r in records:
        item = by_id[r["id"]]
        try:
            client.create_dataset_item(
                id=item.id,
                dataset_name=dataset,
                input={"question": item.question},
                expected_output={"reference": item.reference_answer, "key_facts": item.key_facts},
                metadata={"category": item.category},
            )
            if r["trace_id"]:
                client.api.dataset_run_items.create(
                    run_name=run_name, dataset_item_id=item.id, trace_id=r["trace_id"]
                )
        except Exception:  # noqa: BLE001
            pass
        scores = {f"eval.{k}": v for k, v in r["checks"].items() if v is not None}
        scores.update({f"ragas.{k}": v for k, v in r.get("ragas", {}).items()})
        scores["eval.cost_usd"] = r["cost_usd"]
        scores["eval.latency_s"] = r["latency_s"]
        lf.post_scores(r["trace_id"], scores)
    lf.flush()


def run(args: argparse.Namespace) -> dict:
    items = load_golden(categories=set(args.categories) if args.categories else None)
    if args.limit:
        # Take a spread across categories rather than only the first category.
        items = sorted(items, key=lambda i: (items.index(i) % max(args.limit, 1), i.id))[
            : args.limit
        ]
    deps = build_deps(args.provider, args.model, args.max_retries, not args.no_rerank)

    records = run_items(items, deps, args.name)
    judge_usage = UsageMetadataCallbackHandler()
    if not args.no_ragas:
        score_ragas(records, judge_usage)
    summary = summarize(records)
    summary["judge_cost_usd"] = estimate_cost(judge_usage.usage_metadata)

    payload = {
        "name": args.name,
        "config": {
            "provider": deps.provider,
            "model": getattr(deps.llm, "model_name", None) or getattr(deps.llm, "model", None),
            "judge_provider": deps.judge_provider,
            "judge_model": getattr(deps.judge, "model_name", None),
            "max_retries": args.max_retries,
            "rerank": not args.no_rerank,
            "n_items": len(items),
            "git_sha": git_sha(),
            "llm_cache": settings.llm_cache,
        },
        "summary": summary,
        "records": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log_to_langfuse(args.name, items, records)

    failures = compare_to_thresholds(summary, load_thresholds())
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v if v is None else round(v, 4)}")
    print("\nThresholds:", "all met" if not failures else "\n  " + "\n  ".join(failures))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="run name; results saved as <name>.json")
    parser.add_argument("--provider", choices=["deepseek", "openai", "anthropic"])
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--max-retries", type=int, default=settings.agent_max_retries)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
