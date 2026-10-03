"""Inference entrypoint used by the agent's forecast tool"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from market_research_agent.models import registry
from market_research_agent.models.classifier import predict_proba_up
from market_research_agent.models.features import DIR_FEATURES, dir_frame, vol_frame
from market_research_agent.models.lstm import denormalize, make_samples, predict_lstm

DISCLAIMER = (
    "Statistical model output for research and education only; not investment advice. "
    "Forecasts are uncertain and past performance does not predict future results."
)


class ForecastError(ValueError):
    pass


class ForecastResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ticker: str
    horizon: Literal["1w", "1m"]
    as_of: date = Field(description="Date of the last price bar used")
    model_version: str
    # horizon == "1w": annualized volatility over the next 5 trading days
    current_realized_vol: float | None = Field(None, description="Last 5-day range-based vol")
    predicted_vol: float | None = Field(None, description="LSTM forecast, annualized")
    har_baseline_vol: float | None = Field(None, description="HAR-RV baseline forecast")
    # horizon == "1m": probability the price is higher in ~21 trading days
    prob_up: float | None = Field(None, ge=0, le=1)
    disclaimer: str = DISCLAIMER


@lru_cache(maxsize=4)
def _bundle(version: str | None, root: str):
    return registry.load_bundle(version, Path(root))


def _load_prices(ticker: str) -> pd.DataFrame:
    from sqlalchemy import select

    from market_research_agent.data.db import session_scope
    from market_research_agent.data.models import Price

    with session_scope() as s:
        rows = s.execute(
            select(Price.date, Price.high, Price.low, Price.adj_close, Price.volume).where(
                Price.ticker == ticker
            )
        ).all()
    return pd.DataFrame(rows, columns=["date", "high", "low", "adj_close", "volume"])


def forecast(
    ticker: str,
    horizon: Literal["1w", "1m"] = "1w",
    prices: pd.DataFrame | None = None,
    model_dir: Path | str | None = None,
    version: str | None = None,
) -> ForecastResult:
    """Forecast for a ticker"""
    if horizon not in ("1w", "1m"):
        raise ForecastError(f"Unsupported horizon {horizon!r}; use '1w' or '1m'.")
    ticker = ticker.upper()
    bundle = _bundle(version, str(model_dir or registry.ARTIFACT_ROOT))
    meta = bundle["meta"]
    if prices is None:
        prices = _load_prices(ticker)
    if prices.empty:
        raise ForecastError(f"No price data for {ticker}.")

    common = dict(ticker=ticker, horizon=horizon, model_version=meta["version"])

    if horizon == "1w":
        stats = meta["stats"].get(ticker)
        if stats is None:
            raise ForecastError(
                f"{ticker} was not in the training set; supported: {sorted(meta['stats'])}."
            )
        frame = vol_frame(prices)
        x, _, t = make_samples(frame, stats, bundle["lstm_cfg"].window)
        if len(x) == 0 or t[-1] != len(frame) - 1:
            raise ForecastError(f"Not enough recent history for {ticker} to build features.")
        pred_log = denormalize(predict_lstm(bundle["lstm"], x[-1:]), stats)[0]
        har = meta["har_params"]
        last = frame.iloc[-1]
        har_log = har[0] + float(np.dot(last[["log_rv1", "log_rv5", "log_rv22"]], har[1:]))
        return ForecastResult(
            **common,
            as_of=frame.index[-1].date(),
            current_realized_vol=float(np.exp(last["log_rv5"])),
            predicted_vol=float(np.exp(pred_log)),
            har_baseline_vol=float(np.exp(har_log)),
        )

    frame = dir_frame(prices)
    last = frame.iloc[[-1]]
    if last[DIR_FEATURES].isna().any(axis=None):
        raise ForecastError(f"Not enough history for {ticker} to compute direction features.")
    return ForecastResult(
        **common,
        as_of=frame.index[-1].date(),
        prob_up=float(predict_proba_up(bundle["classifier"], last)[0]),
    )
