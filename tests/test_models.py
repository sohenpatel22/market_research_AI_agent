import numpy as np
import pandas as pd
import pytest

from market_research_agent.models import registry
from market_research_agent.models.evaluate import dm_test, vol_metrics
from market_research_agent.models.features import (
    DIR_FEATURES,
    VOL_HORIZON,
    dir_frame,
    time_split,
    vol_frame,
)
from market_research_agent.models.forecast import ForecastError, ForecastResult, forecast
from market_research_agent.models.lstm import LSTMConfig
from market_research_agent.models.train import train_all


def synthetic_prices(tickers=("AAA", "BBB", "CCC"), n=900, seed=0) -> pd.DataFrame:
    """Random walks with volatility clustering (AR(1) log-vol)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    frames = []
    for t in tickers:
        log_vol = np.zeros(n)
        for i in range(1, n):
            log_vol[i] = 0.95 * log_vol[i - 1] + 0.2 * rng.standard_normal()
        sigma = 0.01 * np.exp(log_vol)
        ret = sigma * rng.standard_normal(n)
        close = 100 * np.exp(np.cumsum(ret))
        rng_range = sigma * np.abs(rng.standard_normal(n)) + 1e-4
        frames.append(
            pd.DataFrame(
                {
                    "ticker": t,
                    "date": dates,
                    "high": close * (1 + rng_range),
                    "low": close * (1 - rng_range),
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_vol_features_have_no_lookahead():
    prices = synthetic_prices(("AAA",))
    base = vol_frame(prices)
    perturbed = prices.copy()
    perturbed.loc[perturbed.index[-30:], ["high", "adj_close"]] *= 2  # change only the future
    alt = vol_frame(perturbed)
    cutoff = base.index[-31]
    feats = ["ret", "log_rv1", "log_rv5", "log_rv22"]
    pd.testing.assert_frame_equal(base.loc[:cutoff, feats], alt.loc[:cutoff, feats])


def test_targets_are_nan_at_the_end_only():
    frame = vol_frame(synthetic_prices(("AAA",)))
    assert frame["y_vol"].iloc[-VOL_HORIZON:].isna().all()
    assert frame["y_vol"].iloc[30:-VOL_HORIZON].notna().all()
    dfr = dir_frame(synthetic_prices(("AAA",)))
    assert dfr["y_dir"].iloc[-21:].isna().all()


def test_time_split_is_ordered_and_purged():
    dates = pd.bdate_range("2020-01-01", periods=100)
    labels = time_split(dates, gap=5).labels
    order = [x for x in labels.tolist() if x != "gap"]
    assert order == sorted(order, key=["train", "val", "test"].index)
    assert (labels == "gap").sum() == 10
    assert labels.iloc[-1] == "test" and labels.iloc[0] == "train"


def test_metrics_and_dm():
    y = np.log(np.full(50, 0.2))
    assert vol_metrics(y, y)["qlike"] == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    better, worse = rng.random(200) * 0.5, rng.random(200) + 0.5
    assert dm_test(better, worse, horizon=5)["statistic"] < -3


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    prices = synthetic_prices()
    result = train_all(
        prices,
        LSTMConfig(epochs=3, patience=2),
        use_mlflow=False,
        use_arima=False,
        root=root,
    )
    return prices, root, result


def test_train_all_writes_bundle_and_metrics(trained):
    _, root, result = trained
    assert (root / "LATEST").read_text() == result["version"]
    for f in ("lstm.pt", "classifier.joblib", "meta.json"):
        assert (result["dir"] / f).exists()
    assert {"naive", "har", "garch", "lstm"} <= set(result["table"].index)
    assert 0 <= result["clf"]["roc_auc"] <= 1


def test_forecast_volatility(trained):
    prices, root, result = trained
    out = forecast("AAA", "1w", prices=prices[prices.ticker == "AAA"], model_dir=root)
    assert isinstance(out, ForecastResult)
    assert out.model_version == result["version"]
    assert out.predicted_vol > 0 and out.har_baseline_vol > 0 and out.prob_up is None


def test_forecast_direction(trained):
    prices, root, _ = trained
    out = forecast("BBB", "1m", prices=prices[prices.ticker == "BBB"], model_dir=root)
    assert 0 <= out.prob_up <= 1 and out.predicted_vol is None
    assert set(DIR_FEATURES)  # feature list is non-empty


def test_forecast_errors(trained):
    prices, root, _ = trained
    with pytest.raises(ForecastError):
        forecast("ZZZ", "1w", prices=prices[prices.ticker == "AAA"], model_dir=root)
    with pytest.raises(ForecastError):
        forecast("AAA", "1y", prices=prices, model_dir=root)  # type: ignore[arg-type]


def test_missing_bundle(tmp_path):
    with pytest.raises(registry.ModelNotFoundError):
        forecast("AAA", "1w", prices=synthetic_prices(("AAA",)), model_dir=tmp_path)
