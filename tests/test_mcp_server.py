import asyncio
import datetime as dt

import pytest

pytest.importorskip("mcp")

from market_research_agent import mcp_server  # noqa: E402
from market_research_agent.agent.schemas import DataResult, RetrievedChunk  # noqa: E402
from market_research_agent.models.forecast import ForecastResult  # noqa: E402


def test_server_exposes_the_four_tools_with_schemas():
    tools = asyncio.run(mcp_server.server.list_tools())
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {
        "search_filings",
        "forecast_ticker",
        "lookup_market_data",
        "ask_research_agent",
    }
    assert "question" in by_name["ask_research_agent"].input_schema["properties"]
    horizon = by_name["forecast_ticker"].input_schema["properties"]["horizon"]
    assert set(horizon.get("enum", [])) == {"1w", "1m"}  # constrained, not free text


def test_search_filings_clamps_k_and_returns_plain_json(monkeypatch):
    seen = {}

    def fake_search(args):
        seen["args"] = args
        return [
            RetrievedChunk(
                id=1,
                ticker="AAPL",
                filing_type="10-K",
                filed_date=dt.date(2025, 10, 31),
                section="Item 1A",
                text="Supply chain risk.",
                score=0.5,
            )
        ]

    monkeypatch.setattr(mcp_server, "retriever_tool", fake_search)
    out = mcp_server.search_filings("supply chain", tickers=["AAPL"], k=999)
    assert seen["args"].k == 20  # clamped to the schema limit rather than failing
    assert out[0]["filed_date"] == "2025-10-31" and out[0]["section"] == "Item 1A"


def test_forecast_and_lookup_tools(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "forecast_tool",
        lambda ticker, horizon: ForecastResult(
            ticker=ticker,
            horizon=horizon,
            as_of=dt.date(2026, 1, 1),
            model_version="t",
            predicted_vol=0.2,
        ),
    )
    f = mcp_server.forecast_ticker("aapl", "1w")
    assert f["ticker"] == "AAPL" and f["predicted_vol"] == 0.2

    monkeypatch.setattr(
        mcp_server,
        "sql_tool",
        lambda req: DataResult(kind=req.kind, ticker=req.ticker, summary=f"{req.kind} ok"),
    )
    assert mcp_server.lookup_market_data("latest_price", "MSFT")["summary"] == "latest_price ok"


def test_ask_agent_tool_omits_bulky_context(monkeypatch):
    from market_research_agent.agent.schemas import AgentAnswer
    from market_research_agent.agent.service import TracedAnswer

    answer = AgentAnswer(question="q", answer="a", quality_passed=True)
    monkeypatch.setattr(
        "market_research_agent.agent.service.ask",
        lambda q, **kw: TracedAnswer(answer=answer, trace_url="https://trace"),
    )
    out = mcp_server.ask_research_agent("q")
    assert out["answer"] == "a" and out["trace_url"] == "https://trace"
    assert "retrieved_context" not in out
