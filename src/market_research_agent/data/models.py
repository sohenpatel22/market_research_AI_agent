import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from market_research_agent.data.db import Base

# Output dimension of the default embedding model (BAAI/bge-small-en-v1.5).
EMBEDDING_DIM = 384


class Price(Base):
    """Daily OHLCV bar for a ticker"""

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
    """A single reported metric (e.g"""

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
    """A chunk of filing text with its embedding"""

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
    section: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_text: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)


class ForecastRecord(Base):
    """A published model forecast (one row per ticker, horizon, as-of date and model version)"""

    __tablename__ = "model_forecasts"
    __table_args__ = (
        UniqueConstraint("ticker", "horizon", "as_of", "model_version", name="uq_forecast_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    horizon: Mapped[str] = mapped_column(String(2), nullable=False)  # "1w" or "1m"
    as_of: Mapped[dt.date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    har_baseline_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_realized_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_up: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class VolBacktestRecord(Base):
    """Volatility forecasts vs what actually happened, per ticker and date (annualized, decimal)"""

    __tablename__ = "vol_backtest"
    __table_args__ = (
        UniqueConstraint("ticker", "date", "model_version", name="uq_vol_backtest_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    date: Mapped[dt.date] = mapped_column(Date, index=True, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    split: Mapped[str] = mapped_column(String(8), nullable=False)  # train | val | test | gap
    actual_vol: Mapped[float] = mapped_column(Float, nullable=False)
    lstm_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    har_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
    naive_vol: Mapped[float | None] = mapped_column(Float, nullable=True)
