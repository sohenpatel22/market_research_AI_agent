"""Statistical volatility baselines the LSTM has to beat: persistence, HAR, GARCH, ARIMA.

All predict log annualized 5-day-ahead volatility, aligned to the row where the forecast is made.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

from market_research_agent.models.features import HAR_FEATURES, VOL_HORIZON, ann_log_vol


def naive_predict(frame: pd.DataFrame) -> pd.Series:
    """Persistence: the next 5 days' vol equals the last 5 days' vol."""
    return frame["log_rv5"]


def fit_har(train_rows: pd.DataFrame) -> list[float]:
    """Pooled HAR-RV regression (Corsi 2009): y ~ daily + weekly + monthly log-vol. Returns
    [intercept, b_daily, b_weekly, b_monthly]."""
    data = train_rows.dropna(subset=[*HAR_FEATURES, "y_vol"])
    res = sm.OLS(data["y_vol"], sm.add_constant(data[HAR_FEATURES])).fit()
    return [float(v) for v in res.params.to_numpy()]


def har_predict(frame: pd.DataFrame, params: list[float]) -> pd.Series:
    vals = params[0] + frame[HAR_FEATURES].to_numpy() @ np.array(params[1:])
    return pd.Series(vals, index=frame.index)


def garch_predict(frame: pd.DataFrame, train_mask: np.ndarray) -> pd.Series:
    """GARCH(1,1)-t fitted on train returns only; forecasts the mean variance over the next 5
    days from every origin. A level bias (Parkinson vs close-to-close variance) is corrected
    using train rows."""
    ret = (frame["ret"].dropna() * 100.0).astype(float)
    last_train = frame.index[train_mask].max()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(ret, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(
            last_obs=last_train, disp="off"
        )
        fc = res.forecast(horizon=VOL_HORIZON, start=ret.index[0], reindex=False)
    daily_var = fc.variance.mean(axis=1) / 1e4
    pred = pd.Series(ann_log_vol(daily_var.to_numpy()), index=daily_var.index).reindex(frame.index)
    bias = (frame["y_vol"] - pred)[train_mask].dropna().mean()
    return pred + bias


def arima_predict(
    frame: pd.DataFrame, train_mask: np.ndarray, origin_mask: np.ndarray, bias_mask: np.ndarray
) -> pd.Series:
    """ARIMA(1,0,1) on daily log-vol, fitted on train; parameters frozen and the state updated
    day by day (no look-ahead) to forecast the next 5 days from each origin in `origin_mask`.
    The exp/log round trip biases the level, corrected on origins in `bias_mask` (validation)."""
    values = frame["log_rv1"].to_numpy()
    finite = np.isfinite(values)
    if (train_mask & finite).sum() < 100:
        return pd.Series(np.nan, index=frame.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = ARIMA(values[train_mask & finite], order=(1, 0, 1)).fit()
        out = np.full(len(frame), np.nan)
        for i in np.flatnonzero(origin_mask & finite):
            hist = values[: i + 1]
            fc = base.apply(hist[np.isfinite(hist)], refit=False).forecast(VOL_HORIZON)
            out[i] = 0.5 * np.log(np.mean(np.exp(2 * fc)))
    pred = pd.Series(out, index=frame.index)
    bias = (frame["y_vol"] - pred)[bias_mask].dropna().mean()
    return pred + (0.0 if np.isnan(bias) else bias)
