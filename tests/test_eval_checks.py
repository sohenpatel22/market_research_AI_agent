import datetime as dt
import json

from market_research_agent.agent.schemas import AgentAnswer, Citation, DataResult
from market_research_agent.eval.checks import (
    deterministic_checks,
    fact_numbers_present,
    load_thresholds,
    numbers_in,
    rate,
)
from market_research_agent.eval.compare import load_runs
from market_research_agent.eval.golden import GoldenItem, load_golden
from market_research_agent.eval.run_eval import compare_to_thresholds, summarize
from market_research_agent.llm.pricing import estimate_cost
from market_research_agent.models.forecast import ForecastResult


def answer(**kw) -> AgentAnswer:
    return AgentAnswer(question="q", answer=kw.pop("text", "a"), quality_passed=True, **kw)


def test_numbers_are_matched_in_different_formats():
    fact = "Total Revenue for period 2026-06-30 was 90,007,000,000"
    assert fact_numbers_present(fact, "Revenue was $90.0 billion")
    assert fact_numbers_present(fact, "revenue of 90,007 million")
    assert fact_numbers_present(fact, "about $90.007 billion")
    assert not fact_numbers_present(fact, "Revenue was $85 billion")
    assert fact_numbers_present("MSFT closed at 509.22 on 2026-09-28.", "It closed at $509.22.")
    assert not fact_numbers_present("AAPL moved +6.8% from 2026-08-31", "It rose 9.1%")
    assert numbers_in("up 6.8% and 1,200") == [6.8, 1200.0]


def test_deterministic_checks_by_category():
    forecast = ForecastResult(
        ticker="AAPL", horizon="1w", as_of=dt.date(2026, 1, 1), model_version="t", predicted_vol=0.2
    )
    f_item = GoldenItem(
        id="f",
        category="forecast",
        question="q",
        expected_forecasts=["1w"],
        expected_tickers=["AAPL"],
    )
    assert deterministic_checks(f_item, answer(forecasts=[forecast]))["forecast_tool"] is True
    assert deterministic_checks(f_item, answer())["forecast_tool"] is False

    d_item = GoldenItem(id="d", category="data", question="q", key_facts=["closed at 509.22"])
    data = DataResult(kind="latest_price", ticker="MSFT", summary="s")
    ok = deterministic_checks(d_item, answer(text="It closed at 509.22", data=[data]))
    assert ok["data_tool"] and ok["data_fact"]
    assert deterministic_checks(d_item, answer(text="no idea"))["data_tool"] is False

    r_item = GoldenItem(id="r", category="out_of_scope", question="q", should_refuse=True)
    assert deterministic_checks(r_item, answer(refused=True))["refusal_correct"] is True
    assert deterministic_checks(r_item, answer())["refusal_correct"] is False

    cite = Citation(
        source_id=1, ticker="AAPL", filing_type="10-K", filed_date=dt.date(2025, 1, 1), snippet="s"
    )
    q_item = GoldenItem(id="q", category="filings", question="q")
    assert deterministic_checks(q_item, answer(sources=[cite]))["cited"] is True
    assert deterministic_checks(q_item, answer())["cited"] is False


def test_rate_ignores_not_applicable():
    rows = [{"x": True}, {"x": False}, {"x": None}]
    assert rate(rows, "x") == 0.5 and rate(rows, "missing") is None


def test_golden_dataset_is_well_formed():
    items = load_golden()
    assert 30 <= len(items) <= 50
    assert len({i.id for i in items}) == len(items)
    for i in items:
        if i.category == "filings":
            assert i.source and i.key_facts and i.expected_tickers
        if i.category in ("out_of_scope", "adversarial"):
            assert i.should_refuse
    cats = {i.category for i in items}
    assert cats == {"filings", "forecast", "data", "out_of_scope", "adversarial"}
    ci_ids = set(load_thresholds()["deepeval"]["ci_items"])
    assert ci_ids <= {i.id for i in items}


def test_summary_and_threshold_failures():
    records = [
        {
            "checks": {"refusal_correct": True, "cited": True},
            "quality_passed": True,
            "retries": 0,
            "cost_usd": 0.01,
            "tokens": 100,
            "latency_s": 2.0,
            "ragas": {"faithfulness": 0.4, "answer_relevancy": 0.9},
        },
        {
            "checks": {"refusal_correct": False},
            "quality_passed": False,
            "retries": 2,
            "cost_usd": 0.03,
            "tokens": 300,
            "latency_s": 4.0,
        },
    ]
    s = summarize(records)
    assert s["refusal_accuracy"] == 0.5 and s["ragas_faithfulness"] == 0.4
    assert s["cost_usd_per_question"] == 0.02 and s["tokens_per_question"] == 200
    failures = compare_to_thresholds(s, load_thresholds())
    assert any("faithfulness" in f for f in failures) and any(
        "refusal_accuracy" in f for f in failures
    )


def test_cost_estimate():
    usage = {
        "deepseek-flash": {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "input_token_details": {"cache_read": 0},
        },
        "unknown-model": {"input_tokens": 5, "output_tokens": 5},
    }
    assert abs(estimate_cost(usage) - 1.5) < 1e-9


def test_compare_skips_non_run_files(tmp_path):
    (tmp_path / "retrieval_ablation.json").write_text(json.dumps({"summary": {}}))
    run = {
        "name": "r1",
        "config": {"provider": "deepseek", "model": "m", "n_items": 3},
        "summary": {"ragas_faithfulness": 0.9, "cost_usd_per_question": 0.01},
        "records": [],
    }
    (tmp_path / "r1.json").write_text(json.dumps(run))
    table = load_runs(tmp_path)
    assert list(table.index) == ["r1 (deepseek/m, n=3)"]
    assert table.iloc[0]["faithfulness"] == 0.9


def _cite(ticker: str) -> Citation:
    return Citation(
        source_id=1,
        ticker=ticker,
        filing_type="10-K",
        filed_date=dt.date(2025, 1, 1),
        snippet="s",
    )


def test_multi_source_requires_every_expected_company_to_be_cited():
    item = GoldenItem(
        id="m", category="multi_source", question="q", expected_tickers=["AAPL", "NVDA"]
    )
    both = deterministic_checks(item, answer(sources=[_cite("AAPL"), _cite("NVDA")]))
    assert both["multi_source"] is True and both["cited"] is True
    one = deterministic_checks(item, answer(sources=[_cite("AAPL")]))
    assert one["multi_source"] is False and one["cited"] is True


def test_mixed_questions_check_each_tool_they_expect():
    item = GoldenItem(
        id="x",
        category="mixed",
        question="q",
        expected_tickers=["NVDA"],
        expected_forecasts=["1w"],
        expects_data=True,
    )
    forecast = ForecastResult(
        ticker="NVDA", horizon="1w", as_of=dt.date(2026, 1, 1), model_version="t", predicted_vol=0.2
    )
    data = DataResult(kind="price_change", ticker="NVDA", summary="s")
    full = deterministic_checks(
        item, answer(forecasts=[forecast], data=[data], sources=[_cite("NVDA")])
    )
    assert full["forecast_tool"] and full["data_tool"] and full["cited"]
    missing = deterministic_checks(item, answer(sources=[_cite("NVDA")]))
    assert missing["forecast_tool"] is False and missing["data_tool"] is False


def test_forecast_check_needs_the_right_tickers_not_just_any_forecast():
    item = GoldenItem(
        id="f",
        category="forecast",
        question="q",
        expected_tickers=["AAPL", "MSFT"],
        expected_forecasts=["1w"],
    )

    def vol(ticker):
        return ForecastResult(
            ticker=ticker,
            horizon="1w",
            as_of=dt.date(2026, 1, 1),
            model_version="t",
            predicted_vol=0.2,
        )

    assert deterministic_checks(item, answer(forecasts=[vol("AAPL"), vol("MSFT")]))["forecast_tool"]
    assert not deterministic_checks(item, answer(forecasts=[vol("AAPL")]))["forecast_tool"]


def test_unanswerable_accepts_abstention_or_refusal_but_not_a_confident_answer():
    item = GoldenItem(id="u", category="unanswerable", question="q")
    ok = deterministic_checks(item, answer(text="The filings do not contain that information."))
    assert ok["abstained"] is True and ok["refusal_correct"] is None
    assert deterministic_checks(item, answer(refused=True))["abstained"] is True
    confident = deterministic_checks(item, answer(text="Apple's 1999 revenue was $6.1 billion."))
    assert confident["abstained"] is False
