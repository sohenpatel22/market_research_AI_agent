"""Read price data for charts (the UI reads through the same database the agent uses)"""

import pandas as pd
from sqlalchemy import text

from market_research_agent.data.db import get_engine


def load_prices(ticker: str, days: int | None = None) -> pd.DataFrame:
    """OHLCV rows for one ticker, oldest first (date, open, high, low, close, adj_close, volume)"""
    query = text(
        "SELECT date, open, high, low, close, adj_close, volume FROM prices "
        "WHERE ticker = :t ORDER BY date"
    )
    df = pd.read_sql(query, get_engine(), params={"t": ticker.upper()})
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(days) if days else df


def load_adj_close(tickers: list[str], days: int | None = None) -> pd.DataFrame:
    """Wide frame of adjusted closes: one column per ticker, indexed by date"""
    series = {}
    for t in tickers:
        df = load_prices(t)
        if not df.empty:
            series[t.upper()] = df.set_index("date")["adj_close"]
    wide = pd.DataFrame(series).dropna(how="all")
    return wide.tail(days) if days else wide
