"""Model Context Protocol server: expose the research tools to any MCP client (Claude Desktop,
IDE agents, ...).

    uv run python -m market_research_agent.mcp_server                # stdio (default)
    uv run python -m market_research_agent.mcp_server --http         # streamable HTTP

Four tools: `search_filings`, `forecast_ticker`, `lookup_market_data` need only the database and the
trained models; `ask_research_agent` runs the full agent and needs an LLM key.
Requires the `mcp` extra:  uv sync --extra mcp
"""

import argparse
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from market_research_agent.agent.schemas import DataRequest, RetrieverInput
from market_research_agent.agent.tools import forecast_tool, retriever_tool, sql_tool

INSTRUCTIONS = (
    "Research tools over SEC filings (10-K/10-Q), daily prices and fundamentals, and trained "
    "volatility/direction forecasts for AAPL, MSFT, NVDA, JPM and XOM. Outputs are for "
    "research and education only, not investment advice."
)

server = MCPServer("market-research-agent", instructions=INSTRUCTIONS)


@server.tool()
def search_filings(
    query: str, tickers: list[str] | None = None, k: int = 5
) -> list[dict[str, Any]]:
    """Hybrid (vector + keyword) search over SEC filing excerpts, reranked by a cross-encoder.

    Returns excerpts with company, form type, filing date, section and text."""
    args = RetrieverInput(query=query, tickers=tickers, k=min(max(k, 1), 20))
    return [c.model_dump(mode="json") for c in retriever_tool(args)]


@server.tool()
def forecast_ticker(ticker: str, horizon: Literal["1w", "1m"] = "1w") -> dict[str, Any]:
    """Model forecast: '1w' = annualized volatility over the next 5 trading days (LSTM, with a
    HAR-RV baseline); '1m' = probability of a positive return over the next month."""
    return forecast_tool(ticker.upper(), horizon).model_dump(mode="json")


@server.tool()
def lookup_market_data(
    kind: Literal["latest_price", "price_change", "fundamental"],
    ticker: str,
    days: int = 30,
    metric: str | None = None,
) -> dict[str, Any]:
    """Whitelisted read-only lookups: the latest close, the price change over `days`, or a
    fundamental metric by name fragment (e.g. 'Total Revenue')."""
    request = DataRequest(kind=kind, ticker=ticker, days=days, metric=metric)
    return sql_tool(request).model_dump(mode="json")


@server.tool()
def ask_research_agent(question: str) -> dict[str, Any]:
    """Ask the full agent: it routes the question, retrieves evidence, runs forecasts and lookups,
    and returns a cited answer that a judge model has checked against its sources."""
    from market_research_agent.agent.service import ask

    traced = ask(question, tags=["mcp"], flush_now=True)
    answer = traced.answer.model_dump(mode="json", exclude={"retrieved_context"})
    return {**answer, "trace_url": traced.trace_url}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP, not stdio")
    args = parser.parse_args()
    server.run("streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
