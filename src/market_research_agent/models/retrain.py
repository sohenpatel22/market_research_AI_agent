"""Scheduled walk-forward retraining.

Hold out the latest trading days, train a challenger on the data before them and compare it with the
current model on those days. If it is clearly better, train on all the data and save a new bundle.
The current model's holdout error is also compared with its original test error as a drift check.

    uv run python -m market_research_agent.models.retrain --report retrain_report.md
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from market_research_agent.models import registry
from market_research_agent.models.evaluate import vol_metrics
from market_research_agent.models.features import VOL_HORIZON
from market_research_agent.models.lstm import LSTMConfig
from market_research_agent.models.publish import backtest_frame
from market_research_agent.models.train import train_all

DRIFT_RATIO = 1.5


def holdout_start(prices: pd.DataFrame, holdout_days: int) -> pd.Timestamp:
    dates = sorted(pd.to_datetime(prices["date"]).unique())
    return pd.Timestamp(dates[-(holdout_days + VOL_HORIZON)])


def _version_date(version: str) -> str:
    """Bundles made before `data_through` was recorded: use the date in the version name."""
    return f"{version[:4]}-{version[4:6]}-{version[6:8]}"


def holdout_metrics(prices: pd.DataFrame, bundle: dict, start: pd.Timestamp) -> dict:
    """LSTM volatility forecast errors for one bundle on days from `start` that have an outcome."""
    frame = backtest_frame(prices, bundle)
    frame = frame[pd.to_datetime(frame["date"]) >= start]
    metrics = vol_metrics(np.log(frame["actual"].to_numpy()), np.log(frame["lstm"].to_numpy()))
    return {**metrics, "n_rows": int(len(frame))}


def run_retrain(
    prices: pd.DataFrame,
    root: Path = registry.ARTIFACT_ROOT,
    holdout_days: int = 60,
    min_gain: float = 0.02,
    dry_run: bool = False,
    cfg: LSTMConfig | None = None,
) -> dict:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    start = holdout_start(prices, holdout_days)
    champion = registry.load_bundle(root=root)

    with tempfile.TemporaryDirectory() as tmp:
        trained = train_all(
            prices[prices["date"] < start], cfg, use_mlflow=False, use_arima=False, root=Path(tmp)
        )
        challenger = registry.load_bundle(trained["version"], Path(tmp))
        champion_scores = holdout_metrics(prices, champion, start)
        challenger_scores = holdout_metrics(prices, challenger, start)

    champion_qlike, challenger_qlike = champion_scores["qlike"], challenger_scores["qlike"]
    gain = 1 - challenger_qlike / champion_qlike
    original_mae = champion["meta"]["vol_test_metrics"]["lstm"]["mae_vol"]
    drift = champion_scores["mae_vol"] > DRIFT_RATIO * original_mae
    champion_through = champion["meta"].get("data_through") or _version_date(
        champion["meta"]["version"]
    )

    report = {
        "holdout_start": str(start.date()),
        "data_through": str(prices["date"].max().date()),
        "champion_version": champion["meta"]["version"],
        "champion_saw_holdout": pd.Timestamp(champion_through) >= start,
        "champion": champion_scores,
        "challenger": challenger_scores,
        "qlike_gain": gain,
        "min_gain": min_gain,
        "drift_detected": bool(drift),
        "promote": bool(gain > min_gain),
        "new_version": None,
    }
    if report["promote"] and not dry_run:
        final = train_all(prices, cfg, use_mlflow=False, use_arima=False, root=root)
        report["new_version"] = final["version"]
    return report


def to_markdown(report: dict) -> str:
    verdict = "promote the retrained model" if report["promote"] else "keep the current model"
    drift = "yes" if report["drift_detected"] else "no"
    rows = "\n".join(
        f"| {name} | {s['qlike']:.4f} | {s['rmse_vol']:.4f} | {s['mae_vol']:.4f} |"
        for name, s in (("champion", report["champion"]), ("challenger", report["challenger"]))
    )
    seen = (
        "Note: the current model was trained on part of the holdout, so the comparison "
        "favours it.\n\n"
        if report["champion_saw_holdout"]
        else ""
    )
    new = f"\n\nNew bundle: `{report['new_version']}`." if report["new_version"] else ""
    return (
        "## Walk-forward retraining\n\n"
        f"Data through {report['data_through']}; holdout from {report['holdout_start']} "
        f"({report['champion']['n_rows']} rows). Champion: `{report['champion_version']}`.\n\n"
        "| Model | QLIKE | RMSE (vol) | MAE (vol) |\n|---|---|---|---|\n"
        f"{rows}\n\n"
        f"QLIKE improvement {report['qlike_gain']:+.1%} "
        f"(required: more than {report['min_gain']:.0%}). Drift detected: {drift}.\n\n"
        f"{seen}Decision: {verdict}.{new}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-days", type=int, default=60)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true", help="evaluate only, write no bundle")
    parser.add_argument("--report", type=Path, help="write a markdown report here")
    args = parser.parse_args()

    from market_research_agent.data.db import get_engine

    prices = pd.read_sql(
        "SELECT ticker, date, high, low, adj_close, volume FROM prices", get_engine()
    )
    if prices.empty:
        raise SystemExit("No price data. Run the ingestion first.")

    report = run_retrain(
        prices, holdout_days=args.holdout_days, min_gain=args.min_gain, dry_run=args.dry_run
    )
    text = to_markdown(report)
    print(text)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"promoted={str(report['new_version'] is not None).lower()}\n")
            f.write(
                f"details={json.dumps({k: report[k] for k in ('qlike_gain', 'drift_detected')})}\n"
            )


if __name__ == "__main__":
    main()
