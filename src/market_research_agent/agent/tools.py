"""The agent's three tools, each with Pydantic inputs/outputs: filing search, forecast, SQL lookup.

The SQL tool never accepts SQL. The model picks one of a few whitelisted, parameterized lookups
and the values are validated; queries run in a READ ONLY transaction.
"""

import re

from sqlalchemy import text

from market_research_agent.agent.retriever import retrieve
from market_research_agent.agent.schemas import (
    DataRequest,
    DataResult,
    RetrievedChunk,
    RetrieverInput,
)
from market_research_agent.data.db import session_scope
from market_research_agent.models.forecast import ForecastResult, forecast

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_METRIC_RE = re.compile(r"^[A-Za-z0-9 &,'()/\-]{1,64}$")


class ToolInputError(ValueError):
    pass


def retriever_tool(args: RetrieverInput, **kwargs) -> list[RetrievedChunk]:
    """Semantic + keyword search over filings."""
    return retrieve(args.query, args.tickers, args.filing_types, args.k, **kwargs)


def forecast_tool(ticker: str, horizon: str) -> ForecastResult:
    """Run the trained forecasting models (1w volatility / 1m direction)."""
    return forecast(ticker, horizon)  # type: ignore[arg-type]


def _validate(req: DataRequest) -> tuple[str, str | None]:
    ticker = req.ticker.upper().strip()
    if not _TICKER_RE.match(ticker):
        raise ToolInputError(f"Invalid ticker {req.ticker!r}")
    metric = None
    if req.kind == "fundamental":
        if not req.metric or not _METRIC_RE.match(req.metric):
            raise ToolInputError("A valid `metric` name is required for fundamental lookups")
        # Escape LIKE wildcards so the value can only ever be a literal substring.
        metric = req.metric.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return ticker, metric


def sql_tool(req: DataRequest) -> DataResult:
    """Whitelisted read-only lookups against prices / fundamentals."""
    ticker, metric = _validate(req)
    with session_scope() as s:
        s.execute(text("SET LOCAL transaction_read_only = on"))

        if req.kind == "latest_price":
            row = s.execute(
                text(
                    "SELECT date, close, adj_close, volume FROM prices "
                    "WHERE ticker = :t ORDER BY date DESC LIMIT 1"
                ),
                {"t": ticker},
            ).first()
            if row is None:
                return DataResult(
                    kind=req.kind, ticker=ticker, summary=f"No price data for {ticker}."
                )
            return DataResult(
                kind=req.kind,
                ticker=ticker,
                summary=f"{ticker} closed at {row.close:.2f} on {row.date}.",
                rows=[{"date": str(row.date), "close": row.close, "volume": row.volume}],
            )

        if req.kind == "price_change":
            rows = s.execute(
                text(
                    "SELECT date, adj_close FROM prices WHERE ticker = :t AND date >= "
                    "(SELECT max(date) FROM prices WHERE ticker = :t) - CAST(:d AS integer) "
                    "ORDER BY date"
                ),
                {"t": ticker, "d": req.days},
            ).all()
            if len(rows) < 2:
                return DataResult(
                    kind=req.kind, ticker=ticker, summary=f"Not enough data for {ticker}."
                )
            first, last = rows[0], rows[-1]
            pct = (last.adj_close / first.adj_close - 1) * 100
            return DataResult(
                kind=req.kind,
                ticker=ticker,
                summary=f"{ticker} moved {pct:+.1f}% from {first.date} to {last.date} (adjusted).",
                rows=[
                    {"start": str(first.date), "end": str(last.date), "pct_change": round(pct, 2)}
                ],
            )

        rows = s.execute(
            text(
                "SELECT period, metric, value FROM fundamentals "
                "WHERE ticker = :t AND metric ILIKE :m ESCAPE '\\' "
                "ORDER BY period DESC, metric LIMIT 12"
            ),
            {"t": ticker, "m": f"%{metric}%"},
        ).all()
        if not rows:
            return DataResult(
                kind=req.kind, ticker=ticker, summary=f"No fundamentals matching {req.metric!r}."
            )
        return DataResult(
            kind=req.kind,
            ticker=ticker,
            summary=f"{len(rows)} fundamental rows for {ticker} matching {req.metric!r}.",
            rows=[{"period": str(r.period), "metric": r.metric, "value": r.value} for r in rows],
        )
