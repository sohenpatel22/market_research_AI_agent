import json

import pytest

from market_research_agent.eval.golden import GoldenItem, load_golden
from market_research_agent.eval.subsets import derive_run, stratified_subset


def items(n_filings=10):
    out = [
        GoldenItem(
            id=f"filings-{i:02d}", category="filings", question="q", expected_tickers=["AAPL"]
        )
        for i in range(n_filings)
    ]
    out += [GoldenItem(id="data-01", category="data", question="q")]
    out += [GoldenItem(id="forecast-01", category="forecast", question="q")]
    return out


def test_subset_keeps_every_non_filing_item_and_spaces_the_filing_sample():
    subset = stratified_subset(items(10), 5)
    ids = [i.id for i in subset]
    assert "data-01" in ids and "forecast-01" in ids
    assert [i for i in ids if i.startswith("filings")] == [
        f"filings-{k:02d}" for k in (0, 2, 4, 6, 8)
    ]
    assert stratified_subset(items(10), None) == items(10)
    assert len(stratified_subset(items(3), 10)) == len(items(3))  # asking for more keeps all


def test_the_real_golden_subset_spans_every_company():
    subset = stratified_subset(load_golden(), 20)
    filings = [i for i in subset if i.category == "filings"]
    assert len(filings) == 20
    assert {i.expected_tickers[0] for i in filings} == {"AAPL", "MSFT", "NVDA", "JPM", "XOM"}
    assert len(subset) == len(load_golden()) - 30


def _record(i, category, passed=True, ragas=None):
    return {
        "id": i,
        "category": category,
        "question": "q",
        "answer": "a",
        "refused": False,
        "quality_passed": passed,
        "grade_score": 0.9,
        "retries": 0,
        "n_sources": 1,
        "contexts": [],
        "tokens": 1000,
        "cost_usd": 0.01,
        "latency_s": 2.0,
        "checks": {"cited": True} if category == "filings" else {"refusal_correct": True},
        **({"ragas": ragas} if ragas else {}),
    }


def test_derive_run_restricts_and_resummarises_without_llm_calls(tmp_path):
    records = [
        _record("filings-00", "filings", ragas={"faithfulness": 1.0}),
        _record("filings-01", "filings", ragas={"faithfulness": 0.0}),
        _record("data-01", "data"),
    ]
    source = {
        "name": "src",
        "config": {"provider": "deepseek", "model": "m", "n_items": 3},
        "summary": {},
        "records": records,
    }
    (tmp_path / "src.json").write_text(json.dumps(source))
    out = derive_run("src", {"filings-00", "data-01"}, "sub", tmp_path)
    assert out["config"]["n_items"] == 2 and out["config"]["derived_from"] == "src"
    assert out["summary"]["ragas_faithfulness"] == 1.0  # only the kept filing item counts
    assert out["summary"]["tokens_per_question"] == 1000
    assert out["summary"]["judge_cost_usd"] is None
    assert (tmp_path / "sub.json").exists()
    with pytest.raises(ValueError, match="no records"):
        derive_run("src", {"nope-01"}, "x", tmp_path)
