import numpy as np
import pytest
from sqlalchemy import func, select

from market_research_agent.data.models import ForecastRecord, VolBacktestRecord
from market_research_agent.models import registry
from market_research_agent.models.lstm import LSTMConfig
from market_research_agent.models.publish import backtest_frame, publish_backtest, publish_forecasts
from market_research_agent.models.train import train_all
from tests.test_models import synthetic_prices


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundle")
    prices = synthetic_prices(("PUBA", "PUBB"), n=700)
    train_all(
        prices, LSTMConfig(epochs=2, patience=1), use_mlflow=False, use_arima=False, root=root
    )
    return prices, root, registry.load_bundle(root=root)


def test_backtest_frame_is_labelled_and_realistic(trained):
    prices, _, bundle = trained
    bt = backtest_frame(prices, bundle)
    assert {"date", "ticker", "split", "actual", "lstm", "har", "naive"} <= set(bt.columns)
    assert set(bt["ticker"]) == {"PUBA", "PUBB"}
    assert not bt[["actual", "lstm", "har", "naive"]].isna().any().any()
    assert (bt[["actual", "lstm", "har", "naive"]] > 0).all().all()
    # volatilities are decimals (25% = 0.25), not percentages
    assert bt["actual"].median() < 5
    # chronological order of the labels across the date axis
    order = {"train": 0, "val": 1, "test": 2}
    seq = bt[bt["split"].isin(order)].sort_values("date")["split"].map(order)
    assert seq.is_monotonic_increasing
    # the last 5 days have no realized outcome yet and are excluded
    assert bt.groupby("ticker")["date"].max().max() < prices["date"].max()


def test_publish_is_idempotent(db_session, trained):
    prices, root, bundle = trained
    version = bundle["meta"]["version"]
    bt = backtest_frame(prices, bundle)
    first = publish_backtest(db_session, bt, version)
    assert first == len(bt)
    assert publish_backtest(db_session, bt, version) == len(bt)  # upsert, not duplicate
    count = db_session.execute(
        select(func.count())
        .select_from(VolBacktestRecord)
        .where(VolBacktestRecord.ticker.in_(["PUBA", "PUBB"]))
    ).scalar_one()
    assert count == len(bt)

    n = publish_forecasts(db_session, prices, ["PUBA", "PUBB"], model_dir=root)
    assert n == 4  # two tickers x two horizons
    assert publish_forecasts(db_session, prices, ["PUBA", "PUBB"], model_dir=root) == 4
    rows = (
        db_session.execute(select(ForecastRecord).where(ForecastRecord.ticker == "PUBA"))
        .scalars()
        .all()
    )
    assert {r.horizon for r in rows} == {"1w", "1m"}
    vol = next(r for r in rows if r.horizon == "1w")
    assert vol.predicted_vol > 0 and vol.prob_up is None
    direction = next(r for r in rows if r.horizon == "1m")
    assert 0 <= direction.prob_up <= 1 and direction.predicted_vol is None
    assert np.isfinite(vol.har_baseline_vol)
