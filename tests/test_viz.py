import datetime as dt

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from market_research_agent.agent.schemas import AgentAnswer, Citation, DataResult
from market_research_agent.api.streaming import iter_in_thread
from market_research_agent.models.forecast import ForecastResult
from market_research_agent.viz import charts, formatting


def make_prices(n=300, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
        }
    )


def vol_result(**kw) -> ForecastResult:
    base = dict(
        ticker="AAPL",
        horizon="1w",
        as_of=dt.date(2026, 1, 1),
        model_version="t",
        predicted_vol=0.25,
        har_baseline_vol=0.22,
        current_realized_vol=0.2,
    )
    return ForecastResult(**{**base, **kw})


def test_forecast_chart_marks_the_forecast():
    fig = charts.forecast_chart(make_prices(), vol_result())
    names = {t.name for t in fig.data}
    assert {"Adj. close", "Realized vol", "LSTM forecast", "HAR baseline"} <= names
    lstm = next(t for t in fig.data if t.name == "LSTM forecast")
    assert lstm.y[0] == 25.0  # shown in percent
    # without a result it still draws the history
    assert {t.name for t in charts.forecast_chart(make_prices()).data} == {
        "Adj. close",
        "Realized vol",
    }


def test_direction_gauge():
    fig = charts.direction_gauge(0.62)
    assert fig.data[0].value == 62.0


def test_matplotlib_figures_are_built_without_pyplot():
    idx = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))) for t in ["AAA", "BBB", "CCC"]},
        index=idx,
    )
    for fig in (
        charts.price_performance_figure(wide),
        charts.returns_distribution_figure(wide, "AAA"),
        charts.correlation_figure(wide),
    ):
        assert isinstance(fig, Figure)
        assert fig.axes


def test_charts_degrade_gracefully_on_missing_data():
    empty = pd.DataFrame()
    assert isinstance(charts.price_performance_figure(empty), Figure)
    assert isinstance(charts.returns_distribution_figure(empty, "AAA"), Figure)
    one = pd.DataFrame({"AAA": [1.0, 2.0]})
    assert isinstance(charts.correlation_figure(one), Figure)


def answer(**kw) -> AgentAnswer:
    return AgentAnswer(
        question="q", answer=kw.pop("text", "Because [S1]."), quality_passed=True, **kw
    )


def test_formatting_sources_and_status():
    cite = Citation(
        source_id=1,
        ticker="AAPL",
        filing_type="10-K",
        filed_date=dt.date(2025, 10, 31),
        section="Item 1A",
        snippet="line one\nline two",
    )
    a = answer(sources=[cite], grade_score=0.9, retries=1)
    rows = formatting.source_rows(a)
    assert rows == [[1, "AAPL", "10-K", "2025-10-31", "Item 1A", "line one line two"]]
    assert (
        formatting.status_line(a, 3.21)
        == "passed quality check | grader score 0.90 | 1 retry | 3.2s"
    )
    assert formatting.status_line(answer(refused=True)) == "declined"


def test_formatting_warns_when_quality_check_failed():
    bad = AgentAnswer(question="q", answer="Maybe.", quality_passed=False)
    assert "could not fully verify" in formatting.answer_markdown(bad)
    refused = AgentAnswer(question="q", answer="No.", quality_passed=True, refused=True)
    assert formatting.answer_markdown(refused).startswith(">")


def test_formatting_forecasts_and_data():
    md = formatting.forecast_markdown(vol_result())
    assert "25.0%" in md and "22.0%" in md and "20.0%" in md
    up = ForecastResult(
        ticker="MSFT", horizon="1m", as_of=dt.date(2026, 1, 1), model_version="t", prob_up=0.61
    )
    assert "61%" in formatting.forecast_markdown(
        up
    ) and "Weak model" in formatting.forecast_markdown(up)
    facts = formatting.facts_markdown(
        answer(
            forecasts=[vol_result()],
            data=[DataResult(kind="k", ticker="AAPL", summary="S")],
            tool_errors=["boom"],
        )
    )
    assert "Data:" in facts and "boom" in facts
    assert "No forecasts" in formatting.facts_markdown(answer())
    assert formatting.step_markdown(["route", "gather"]).splitlines() == [
        "- Understanding the question",
        "- Retrieving filings, data and forecasts",
    ]


def test_iter_in_thread_runs_in_one_worker_thread_and_reraises():
    import threading

    seen = set()

    def gen():
        for i in range(3):
            seen.add(threading.get_ident())
            yield {"i": i}

    assert [e["i"] for e in iter_in_thread(gen)] == [0, 1, 2]
    assert len(seen) == 1 and threading.get_ident() not in seen

    def broken():
        yield {"i": 0}
        raise RuntimeError("boom")

    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        list(iter_in_thread(broken))


def test_gradio_app_builds_and_mounts():
    from fastapi.testclient import TestClient

    from market_research_agent.agent.schemas import DraftAnswer, GradeResult
    from market_research_agent.api.main import create_app
    from market_research_agent.api.ratelimit import SlidingWindowLimiter
    from market_research_agent.api.runtime import Runtime
    from tests.test_agent import FILINGS_ROUTE, GOOD, FakeLLM, make_deps

    llm = FakeLLM(RouteDecision=[FILINGS_ROUTE], DraftAnswer=[DraftAnswer(answer="a")])
    deps, _ = make_deps(llm, FakeLLM(GradeResult=[GOOD]))
    app = create_app(Runtime(deps), SlidingWindowLimiter(0), with_ui=True)
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200 and "Market Research Agent" in page.text
    assert client.get("/health").status_code == 200  # API routes still win over the UI mount
    assert GradeResult  # imported for clarity of the fake wiring
