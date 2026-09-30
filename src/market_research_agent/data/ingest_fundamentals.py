"""Ingest basic quarterly fundamentals for a ticker (yfinance -> pandas -> fundamentals table).

Stored long/tidy: one row per (ticker, period, metric). yfinance's set of
reported line items varies by company, so a wide table would be mostly NULLs.
"""

from collections.abc import Callable

import pandas as pd
import yfinance as yf
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from market_research_agent.data.cache import raw_path
from market_research_agent.data.models import Fundamental

STATEMENTS: dict[str, Callable[[yf.Ticker], pd.DataFrame]] = {
    "income": lambda t: t.quarterly_income_stmt,
    "balance": lambda t: t.quarterly_balance_sheet,
    "cashflow": lambda t: t.quarterly_cashflow,
}


def fetch_fundamentals(ticker: str) -> dict[str, pd.DataFrame]:
    """Download (or load from cache) quarterly income/balance/cashflow statements."""
    statements: dict[str, pd.DataFrame] = {}
    yf_ticker: yf.Ticker | None = None

    for name, getter in STATEMENTS.items():
        cache_file = raw_path("fundamentals", f"{ticker.upper()}_{name}.csv")
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0)
        else:
            yf_ticker = yf_ticker or yf.Ticker(ticker)
            df = getter(yf_ticker)
            df.index.name = "metric"
            if not df.empty:
                df.to_csv(cache_file)
        statements[name] = df

    return statements


def tidy_fundamentals(ticker: str, statements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Melt wide (metric x period) statements into a long tidy frame."""
    frames = []
    for df in statements.values():
        if df.empty:
            continue
        df = df.rename_axis("metric").reset_index()
        melted = df.melt(id_vars="metric", var_name="period", value_name="value")
        melted["ticker"] = ticker.upper()
        melted["period"] = pd.to_datetime(melted["period"]).dt.date
        frames.append(melted[["ticker", "period", "metric", "value"]])

    if not frames:
        return pd.DataFrame(columns=["ticker", "period", "metric", "value"])

    tidy = pd.concat(frames, ignore_index=True)
    return tidy.dropna(subset=["value"])


def ingest_fundamentals(session: Session, ticker: str) -> int:
    """Fetch, clean, and upsert fundamentals for a ticker. Returns rows inserted."""
    tidy = tidy_fundamentals(ticker, fetch_fundamentals(ticker))
    if tidy.empty:
        return 0

    stmt = insert(Fundamental).values(tidy.to_dict(orient="records"))
    stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "period", "metric"]).returning(
        Fundamental.id
    )
    # psycopg3 rowcount is -1 for multi-row INSERT ON CONFLICT; count via RETURNING.
    return len(session.execute(stmt).fetchall())
