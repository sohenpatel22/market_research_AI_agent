"""Forecast/classification metrics and the Diebold-Mariano test."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def vol_metrics(y_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    """Metrics for log-annualized-vol forecasts. RMSE/MAE are reported on the vol scale."""
    y, p = np.exp(y_log), np.exp(pred_log)
    ratio = (y / p) ** 2  # realized variance / forecast variance
    return {
        "rmse_vol": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae_vol": float(np.mean(np.abs(y - p))),
        "rmse_log": float(np.sqrt(np.mean((y_log - pred_log) ** 2))),
        "qlike": float(np.mean(ratio - np.log(ratio) - 1)),
    }


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> dict[str, float]:
    """Diebold-Mariano test (HAC / Newey-West, lag = horizon-1). Negative statistic means
    model A has the lower loss. Overlapping h-step targets make errors autocorrelated, so
    naive t-tests would overstate significance."""
    import statsmodels.api as sm  # training/eval extra; not needed for inference

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    fit = sm.OLS(d, np.ones_like(d)).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon - 1, 0)})
    return {"statistic": float(fit.tvalues[0]), "p_value": float(fit.pvalues[0])}


def classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "base_rate": float(np.mean(y)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def comparison_table(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Vol metrics per model on rows where every model has a prediction."""
    common = preds.dropna(subset=["y", *models])
    rows = {m: vol_metrics(common["y"].to_numpy(), common[m].to_numpy()) for m in models}
    return pd.DataFrame(rows).T.sort_values("qlike")
