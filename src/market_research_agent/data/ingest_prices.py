"""Ingest daily OHLCV price history for a ticker (yfinance -> pandas -> prices table)."""

import pandas as pd
import yfinance as yf
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from market_research_agent.data.cache import raw_path
from market_research_agent.data.models import Price

COLUMN_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def fetch_price_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Download (or load from cache) daily OHLCV history for a ticker."""
    cache_file = raw_path("prices", f"{ticker.upper()}_{period}.csv")
    if cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    if not df.empty:
        df.to_csv(cache_file)
    return df


def tidy_prices(ticker: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw yfinance OHLCV frame into the prices table's tidy shape."""
    if raw_df.empty:
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        )

    tidy = raw_df.rename(columns=COLUMN_RENAME).copy()
    tidy.index.name = "date"
    tidy = tidy.reset_index()
    # utc=True avoids errors from mixed UTC offsets across DST transitions in the index.
    tidy["date"] = pd.to_datetime(tidy["date"], utc=True).dt.date
    tidy["ticker"] = ticker.upper()
    if "adj_close" not in tidy.columns:
        tidy["adj_close"] = tidy["close"]

    columns = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    return tidy[columns].dropna().astype({"volume": "int64"})


def ingest_prices(session: Session, ticker: str, period: str = "10y") -> int:
    """Fetch, clean, and upsert price history for a ticker. Returns rows inserted."""
    tidy = tidy_prices(ticker, fetch_price_history(ticker, period=period))
    if tidy.empty:
        return 0

    stmt = insert(Price).values(tidy.to_dict(orient="records"))
    stmt = stmt.on_conflict_do_nothing(index_elements=["ticker", "date"]).returning(Price.id)
    # psycopg3 rowcount is -1 for multi-row INSERT ON CONFLICT; count via RETURNING.
    return len(session.execute(stmt).fetchall())
