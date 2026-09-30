"""CLI entrypoint that ingests prices, fundamentals, and filings for a ticker list.

Usage:
    uv run python -m market_research_agent.data.ingest
    uv run python -m market_research_agent.data.ingest --tickers AAPL MSFT
"""

import argparse
import logging

from market_research_agent.data.db import init_db, session_scope
from market_research_agent.data.ingest_filings import ingest_filings
from market_research_agent.data.ingest_fundamentals import ingest_fundamentals
from market_research_agent.data.ingest_prices import ingest_prices

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]


def ingest_ticker(ticker: str) -> None:
    with session_scope() as session:
        n_prices = ingest_prices(session, ticker)
        n_fundamentals = ingest_fundamentals(session, ticker)
        n_filings = ingest_filings(session, ticker)
    logger.info(
        "%s: %d price rows, %d fundamental rows, %d filing chunks",
        ticker,
        n_prices,
        n_fundamentals,
        n_filings,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    init_db()
    for ticker in args.tickers:
        ingest_ticker(ticker)


if __name__ == "__main__":
    main()
