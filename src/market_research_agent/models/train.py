"""Train, evaluate and track every forecasting model"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from market_research_agent.models import baselines, registry
from market_research_agent.models.classifier import predict_proba_up, select_and_fit
from market_research_agent.models.evaluate import (
    classification_metrics,
    comparison_table,
    dm_test,
)
from market_research_agent.models.features import (
    DIR_HORIZON,
    VOL_FEATURES,
    VOL_HORIZON,
    build_frames,
    dir_frame,
    time_split,
    vol_frame,
)
from market_research_agent.models.lstm import (
    LSTMConfig,
    denormalize,
    fit_stats,
    make_samples,
    predict_lstm,
    train_lstm,
)

logger = logging.getLogger(__name__)

VOL_MODELS = ["naive", "har", "garch", "arima", "lstm"]
EXPERIMENT = "market-research-agent"


def _vol_predictions(prices: pd.DataFrame, cfg: LSTMConfig, use_arima: bool):
    frames = build_frames(prices, vol_frame)
    labels = time_split(
        pd.Index(np.concatenate([f.index.values for f in frames.values()])), gap=VOL_HORIZON
    )
    for f in frames.values():
        f["split"] = labels.of(f.index)

    har_params = baselines.fit_har(pd.concat([f[f["split"] == "train"] for f in frames.values()]))

    # LSTM samples per split, pooled across tickers
    stats = {t: fit_stats(f, (f["split"] == "train").to_numpy()) for t, f in frames.items()}
    per_ticker, buckets = {}, {s: ([], []) for s in ("train", "val")}
    for t, f in frames.items():
        x, y, pos = make_samples(f, stats[t], cfg.window)
        per_ticker[t] = (x, y, pos)
        split = f["split"].to_numpy()[pos]
        for s in ("train", "val"):
            keep = (split == s) & np.isfinite(y)
            buckets[s][0].append(x[keep])
            buckets[s][1].append(y[keep])
    data = {s: (np.concatenate(v[0]), np.concatenate(v[1])) for s, v in buckets.items()}
    model, history = train_lstm(data["train"], data["val"], cfg)

    rows = []
    for t, f in frames.items():
        train_mask = (f["split"] == "train").to_numpy()
        out = pd.DataFrame({"ticker": t, "y": f["y_vol"], "split": f["split"]})
        out["naive"] = baselines.naive_predict(f)
        out["har"] = baselines.har_predict(f, har_params)
        out["garch"] = baselines.garch_predict(f, train_mask)
        out["arima"] = np.nan
        if use_arima:
            origin = f["split"].isin(["val", "test"]).to_numpy()
            out["arima"] = baselines.arima_predict(
                f, train_mask, origin, (f["split"] == "val").to_numpy()
            )
        x, _, pos = per_ticker[t]
        lstm = np.full(len(f), np.nan)
        if len(x):
            lstm[pos] = denormalize(predict_lstm(model, x), stats[t])
        out["lstm"] = lstm
        rows.append(out)
    preds = pd.concat(rows)
    preds.index.name = "date"
    return preds, model, history, har_params, stats


def _direction(prices: pd.DataFrame, seed: int):
    frames = build_frames(prices, dir_frame)
    labels = time_split(
        pd.Index(np.concatenate([f.index.values for f in frames.values()])), gap=DIR_HORIZON
    )
    parts = []
    for t, f in frames.items():
        f = f.copy()
        f["ticker"], f["split"] = t, labels.of(f.index)
        parts.append(f.dropna())
    data = pd.concat(parts)
    tr, va, te = (data[data["split"] == s] for s in ("train", "val", "test"))
    name, clf, val_auc = select_and_fit(tr, va, seed)
    metrics = classification_metrics(te["y_dir"].to_numpy(), predict_proba_up(clf, te))
    return name, clf, val_auc, metrics


def train_all(
    prices: pd.DataFrame,
    cfg: LSTMConfig | None = None,
    use_mlflow: bool = True,
    use_arima: bool = True,
    root: Path = registry.ARTIFACT_ROOT,
) -> dict:
    """Train everything on `prices` (columns: ticker, date, high, low, adj_close, volume)"""
    cfg = cfg or LSTMConfig()
    preds, model, history, har_params, stats = _vol_predictions(prices, cfg, use_arima)
    models = [m for m in VOL_MODELS if preds[m].notna().any()]

    test = preds[preds["split"] == "test"]
    table = comparison_table(test, models)
    common = test.dropna(subset=["y", *models])
    dm = {}
    for m in models:
        if m in ("har", "naive"):
            continue
        loss_m = (common["y"] - common[m]) ** 2
        loss_har = (common["y"] - common["har"]) ** 2
        dm[m] = dm_test(loss_m.to_numpy(), loss_har.to_numpy(), VOL_HORIZON)

    clf_name, clf, val_auc, clf_metrics = _direction(prices, cfg.seed)

    version = registry.new_version()
    meta = {
        "tickers": sorted(stats),
        "stats": stats,
        "har_params": har_params,
        "vol_horizon_days": VOL_HORIZON,
        "direction_horizon_days": DIR_HORIZON,
        "classifier": clf_name,
        "vol_test_metrics": table.to_dict(orient="index"),
        "dm_vs_har": dm,
        "classifier_val_auc": val_auc,
        "classifier_test_metrics": clf_metrics,
        "test_rows": int(len(common)),
        "data_through": str(pd.to_datetime(prices["date"]).max().date()),
    }
    bundle_dir = registry.save_bundle(version, model, cfg, clf, meta, root)
    (bundle_dir / "vol_comparison.csv").write_text(table.to_csv())

    if use_mlflow:
        _log_mlflow(cfg, history, table, dm, clf_name, val_auc, clf_metrics, bundle_dir, model, clf)

    from market_research_agent.observability.experiments import log_training_run

    trace_url = log_training_run({**meta, "version": version})

    return {
        "trace_url": trace_url,
        "version": version,
        "table": table,
        "dm": dm,
        "clf": clf_metrics,
        "dir": bundle_dir,
    }


def _log_mlflow(cfg, history, table, dm, clf_name, val_auc, clf_metrics, bundle_dir, model, clf):
    import mlflow

    from market_research_agent.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="vol-forecasters"):
        mlflow.log_params({f"lstm_{k}": v for k, v in asdict(cfg).items()})
        for i, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"], strict=True)):
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=i)
        for name, row in table.iterrows():
            mlflow.log_metrics({f"test_{name}_{k}": float(v) for k, v in row.items()})
        for name, res in dm.items():
            mlflow.log_metrics({f"dm_{name}_vs_har_stat": res["statistic"]})
            mlflow.log_metrics({f"dm_{name}_vs_har_p": res["p_value"]})
        mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
        example = np.zeros((1, cfg.window, len(VOL_FEATURES)), dtype=np.float32)
        mlflow.pytorch.log_model(
            model, name="lstm", input_example=example, registered_model_name="vol-lstm"
        )

    with mlflow.start_run(run_name="direction-classifier"):
        mlflow.log_param("selected_model", clf_name)
        mlflow.log_metrics({f"val_auc_{k}": v for k, v in val_auc.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in clf_metrics.items() if isinstance(v, float)})
        mlflow.log_dict({"confusion_matrix": clf_metrics["confusion_matrix"]}, "confusion.json")
        # We trained this model ourselves, so its HistGradientBoosting tree type is safe to trust.
        mlflow.sklearn.log_model(
            clf,
            name="classifier",
            registered_model_name="return-classifier",
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=LSTMConfig.epochs)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-arima", action="store_true", help="skip the slow rolling ARIMA")
    args = parser.parse_args()

    from market_research_agent.data.db import get_engine

    prices = pd.read_sql(
        "SELECT ticker, date, high, low, adj_close, volume FROM prices", get_engine()
    )
    if prices.empty:
        raise SystemExit("No price data. Run the ingestion CLI first.")

    result = train_all(
        prices,
        LSTMConfig(epochs=args.epochs),
        use_mlflow=not args.no_mlflow,
        use_arima=not args.no_arima,
    )
    print(f"\nModel version: {result['version']}  ->  {result['dir']}")
    print("\nVolatility forecasts, test set (lower is better; sorted by QLIKE):")
    print(result["table"].round(4).to_string())
    print("\nDiebold-Mariano vs HAR (negative stat = better than HAR):")
    print(json.dumps(result["dm"], indent=2))
    print("\nDirection classifier, test set:")
    print(json.dumps(result["clf"], indent=2))


if __name__ == "__main__":
    main()
