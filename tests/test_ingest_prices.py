import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from market_copilot.data import ingest_prices
from market_copilot.data.models import Price


def _fake_price_history() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.5, 101.5],
            "Volume": [1_000_000, 1_200_000],
        },
        index=index,
    )


def test_tidy_prices_shapes_columns():
    tidy = ingest_prices.tidy_prices("AAPL", _fake_price_history())

    assert list(tidy.columns) == [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert len(tidy) == 2
    assert tidy.iloc[0]["ticker"] == "AAPL"
    assert tidy.iloc[0]["date"] == dt.date(2024, 1, 2)


def test_tidy_prices_handles_empty_frame():
    tidy = ingest_prices.tidy_prices("AAPL", pd.DataFrame())
    assert tidy.empty


# Not one of the real tickers we ingest, so it can never collide with live data
# sitting in the same dev database the tests run against.
TEST_TICKER = "ZZZTEST"


def test_ingest_prices_inserts_rows(db_session: Session, monkeypatch):
    monkeypatch.setattr(
        ingest_prices,
        "fetch_price_history",
        lambda ticker, period="1y": _fake_price_history(),
    )

    n_inserted = ingest_prices.ingest_prices(db_session, TEST_TICKER)

    assert n_inserted == 2
    rows = db_session.execute(select(Price).where(Price.ticker == TEST_TICKER)).scalars().all()
    assert len(rows) == 2


def test_ingest_prices_is_idempotent(db_session: Session, monkeypatch):
    monkeypatch.setattr(
        ingest_prices,
        "fetch_price_history",
        lambda ticker, period="1y": _fake_price_history(),
    )

    ingest_prices.ingest_prices(db_session, TEST_TICKER)
    n_second_run = ingest_prices.ingest_prices(db_session, TEST_TICKER)

    assert n_second_run == 0
    rows = db_session.execute(select(Price).where(Price.ticker == TEST_TICKER)).scalars().all()
    assert len(rows) == 2
