import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Date, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from market_research_agent.data.db import Base

# Output dimension of the default embedding model (BAAI/bge-small-en-v1.5).
EMBEDDING_DIM = 384


class Price(Base):
    """Daily OHLCV bar for a ticker."""

    __tablename__ = "prices"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_prices_ticker_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)


class Fundamental(Base):
    """A single reported metric (e.g. revenue) for a ticker and fiscal period.

    Stored long/tidy (one row per metric) rather than wide, since the set of
    metrics available per ticker/period from a given data source varies.
    """

    __tablename__ = "fundamentals"
    __table_args__ = (
        UniqueConstraint("ticker", "period", "metric", name="uq_fundamentals_ticker_period_metric"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    period: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    metric: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=True)


class Document(Base):
    """A chunk of filing text with its embedding."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("accession_number", "chunk_index", name="uq_documents_accession_chunk"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    filing_type: Mapped[str] = mapped_column(String(16), nullable=False)
    filed_date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    accession_number: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
