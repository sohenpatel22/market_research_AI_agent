import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from market_copilot.data import ingest_fundamentals
from market_copilot.data.models import Fundamental


def _fake_statements() -> dict[str, pd.DataFrame]:
    income = pd.DataFrame(
        {
            pd.Timestamp("2024-06-30"): [1000.0, 200.0],
            pd.Timestamp("2024-03-31"): [950.0, 180.0],
        },
        index=pd.Index(["Total Revenue", "Net Income"], name="metric"),
    )
    return {"income": income, "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}


def test_tidy_fundamentals_melts_to_long_format():
    tidy = ingest_fundamentals.tidy_fundamentals("AAPL", _fake_statements())

    assert list(tidy.columns) == ["ticker", "period", "metric", "value"]
    assert len(tidy) == 4
    assert set(tidy["metric"]) == {"Total Revenue", "Net Income"}


# Not one of the real tickers we ingest, so it can never collide with live data
# sitting in the same dev database the tests run against.
TEST_TICKER = "ZZZTEST"


def test_ingest_fundamentals_inserts_rows(db_session: Session, monkeypatch):
    monkeypatch.setattr(
        ingest_fundamentals, "fetch_fundamentals", lambda ticker: _fake_statements()
    )

    n_inserted = ingest_fundamentals.ingest_fundamentals(db_session, TEST_TICKER)

    assert n_inserted == 4
    rows = (
        db_session.execute(select(Fundamental).where(Fundamental.ticker == TEST_TICKER))
        .scalars()
        .all()
    )
    assert len(rows) == 4
