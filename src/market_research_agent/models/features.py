"""Feature engineering and time-ordered splits for the forecasting models"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUALIZE = 252
VOL_HORIZON = 5
DIR_HORIZON = 21
EPS = 1e-12

VOL_FEATURES = ["log_rv1", "log_rv5", "log_rv22", "ret"]
HAR_FEATURES = ["log_rv1", "log_rv5", "log_rv22"]
DIR_FEATURES = [
    "mom_1",
    "mom_5",
    "mom_21",
    "mom_63",
    "vol_21",
    "vol_63",
    "drawdown_252",
    "volume_z",
    "rsi_14",
]


def ann_log_vol(daily_var):
    """log(annualized volatility) from a daily variance"""
    return 0.5 * np.log(np.maximum(daily_var, EPS) * ANNUALIZE)


def _single_ticker(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values("date").drop_duplicates("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def vol_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Volatility features + target for ONE ticker (columns: date, high, low, adj_close)"""
    df = _single_ticker(prices)
    pk_var = np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))
    out = pd.DataFrame(index=df.index)
    out["ret"] = np.log(df["adj_close"]).diff()
    out["log_rv1"] = ann_log_vol(pk_var)
    out["log_rv5"] = ann_log_vol(pk_var.rolling(VOL_HORIZON).mean())
    out["log_rv22"] = ann_log_vol(pk_var.rolling(22).mean())
    # mean of days t+1..t+5, placed at row t
    out["y_vol"] = ann_log_vol(pk_var.rolling(VOL_HORIZON).mean().shift(-VOL_HORIZON))
    return out


def dir_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Direction features + target for ONE ticker (columns: date, adj_close, volume)"""
    df = _single_ticker(prices)
    close = df["adj_close"]
    logc = np.log(close)
    ret = logc.diff()
    out = pd.DataFrame(index=df.index)
    for k in (1, 5, 21, 63):
        out[f"mom_{k}"] = logc.diff(k)
    out["vol_21"] = ret.rolling(21).std() * np.sqrt(ANNUALIZE)
    out["vol_63"] = ret.rolling(63).std() * np.sqrt(ANNUALIZE)
    out["drawdown_252"] = close / close.rolling(252, min_periods=63).max() - 1
    lv = np.log1p(df["volume"].astype(float))
    out["volume_z"] = (lv - lv.rolling(63).mean()) / lv.rolling(63).std().replace(0, np.nan)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    fwd = close.shift(-DIR_HORIZON) / close - 1
    out["y_dir"] = (fwd > 0).astype(float).where(fwd.notna())
    return out


def build_frames(prices: pd.DataFrame, fn) -> dict[str, pd.DataFrame]:
    return {t: fn(g) for t, g in prices.groupby("ticker")}


@dataclass(frozen=True)
class SplitLabels:
    """Maps each calendar date to 'train' | 'val' | 'test' | 'gap'"""

    labels: pd.Series

    def of(self, index: pd.Index) -> np.ndarray:
        return self.labels.reindex(index).fillna("gap").to_numpy()


def time_split(
    dates: pd.Index, train_frac: float = 0.70, val_frac: float = 0.15, gap: int = 0
) -> SplitLabels:
    """Chronological split shared by ALL tickers"""
    u = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(dates)).unique()))
    n = len(u)
    b1, b2 = int(n * train_frac), int(n * (train_frac + val_frac))
    lab = np.array(["train"] * b1 + ["val"] * (b2 - b1) + ["test"] * (n - b2), dtype=object)
    if gap:
        lab[max(b1 - gap, 0) : b1] = "gap"
        lab[max(b2 - gap, b1) : b2] = "gap"
    return SplitLabels(pd.Series(lab, index=u))
