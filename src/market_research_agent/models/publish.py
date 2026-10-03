"""Publish model output to Postgres so BI tools (Power BI) can chart it"""

import argparse
import logging

import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from market_research_agent.data.models import ForecastRecord, VolBacktestRecord
from market_research_agent.models import registry
from market_research_agent.models.baselines import har_predict
from market_research_agent.models.features import VOL_HORIZON, time_split, vol_frame
from market_research_agent.models.forecast import ForecastError, forecast
from market_research_agent.models.lstm import denormalize, make_samples, predict_lstm

logger = logging.getLogger(__name__)

HORIZONS = ("1w", "1m")


def backtest_frame(prices: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Volatility forecasts vs realized, for every date where all of them are defined"""
    meta, window = bundle["meta"], bundle["lstm_cfg"].window
    frames = {t: vol_frame(g) for t, g in prices.groupby("ticker") if t in meta["stats"]}
    if not frames:
        return pd.DataFrame()
    all_dates = pd.Index(np.concatenate([f.index.values for f in frames.values()]))
    labels = time_split(all_dates, gap=VOL_HORIZON)

    parts = []
    for ticker, frame in frames.items():
        stats = meta["stats"][ticker]
        x, _, pos = make_samples(frame, stats, window)
        if not len(x):
            continue
        lstm_log = denormalize(predict_lstm(bundle["lstm"], x), stats)
        out = pd.DataFrame(index=frame.index[pos])
        out["lstm"] = np.exp(lstm_log)
        out["har"] = np.exp(har_predict(frame, meta["har_params"]).reindex(out.index))
        out["naive"] = np.exp(frame["log_rv5"].reindex(out.index))
        out["actual"] = np.exp(frame["y_vol"].reindex(out.index))
        out = out.dropna(subset=["actual"])
        out["split"] = labels.of(out.index)
        out["ticker"] = ticker
        out.index.name = "date"
        parts.append(out.reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def publish_backtest(session: Session, backtest: pd.DataFrame, model_version: str) -> int:
    if backtest.empty:
        return 0
    rows = [
        {
            "ticker": r.ticker,
            "date": r.date.date(),
            "model_version": model_version,
            "split": r.split,
            "actual_vol": float(r.actual),
            "lstm_vol": float(r.lstm),
            "har_vol": float(r.har),
            "naive_vol": float(r.naive),
        }
        for r in backtest.itertuples()
    ]
    inserted = 0
    for start in range(0, len(rows), 1000):
        stmt = insert(VolBacktestRecord).values(rows[start : start + 1000])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_vol_backtest_key",
            set_={
                "split": stmt.excluded.split,
                "actual_vol": stmt.excluded.actual_vol,
                "lstm_vol": stmt.excluded.lstm_vol,
                "har_vol": stmt.excluded.har_vol,
                "naive_vol": stmt.excluded.naive_vol,
            },
        ).returning(VolBacktestRecord.id)
        inserted += len(session.execute(stmt).fetchall())
    return inserted


def publish_forecasts(
    session: Session, prices: pd.DataFrame, tickers: list[str], model_dir=None
) -> int:
    """Run `forecast()` for each ticker and horizon and upsert the results"""
    rows = []
    for ticker in tickers:
        history = prices[prices["ticker"] == ticker]
        for horizon in HORIZONS:
            try:
                f = forecast(ticker, horizon, prices=history, model_dir=model_dir)
            except ForecastError as exc:
                logger.warning("skipping %s %s: %s", ticker, horizon, exc)
                continue
            rows.append(
                {
                    "ticker": f.ticker,
                    "horizon": f.horizon,
                    "as_of": f.as_of,
                    "model_version": f.model_version,
                    "predicted_vol": f.predicted_vol,
                    "har_baseline_vol": f.har_baseline_vol,
                    "current_realized_vol": f.current_realized_vol,
                    "prob_up": f.prob_up,
                }
            )
    if not rows:
        return 0
    stmt = insert(ForecastRecord).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_forecast_key",
        set_={
            "predicted_vol": stmt.excluded.predicted_vol,
            "har_baseline_vol": stmt.excluded.har_baseline_vol,
            "current_realized_vol": stmt.excluded.current_realized_vol,
            "prob_up": stmt.excluded.prob_up,
        },
    ).returning(ForecastRecord.id)
    return len(session.execute(stmt).fetchall())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()

    from market_research_agent.data.db import get_engine, init_db, session_scope

    init_db()
    prices = pd.read_sql(
        "SELECT ticker, date, high, low, adj_close, volume FROM prices", get_engine()
    )
    if prices.empty:
        raise SystemExit("No price data. Run the ingestion first.")
    bundle = registry.load_bundle()
    version = bundle["meta"]["version"]
    backtest = backtest_frame(prices, bundle)
    with session_scope() as session:
        n_back = publish_backtest(session, backtest, version)
        n_fc = publish_forecasts(session, prices, bundle["meta"]["tickers"])
    print(f"model {version}: {n_fc} forecasts and {n_back} backtest rows written")


if __name__ == "__main__":
    main()
