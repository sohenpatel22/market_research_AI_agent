import pandas as pd
import pytest

from market_research_agent.models import registry
from market_research_agent.models.lstm import LSTMConfig
from market_research_agent.models.retrain import holdout_start, run_retrain, to_markdown
from market_research_agent.models.train import train_all
from tests.test_models import synthetic_prices

CFG = LSTMConfig(epochs=2, patience=1)


@pytest.fixture
def setup(tmp_path):
    prices = synthetic_prices(("RTA", "RTB"), n=700)
    train_all(prices, CFG, use_mlflow=False, use_arima=False, root=tmp_path)
    return prices, tmp_path


def test_holdout_starts_after_the_requested_number_of_days_with_outcomes():
    prices = synthetic_prices(("RTA",), n=300)
    start = holdout_start(prices, 40)
    dates = sorted(pd.to_datetime(prices["date"]))
    assert dates.index(start) == len(dates) - 45


def test_a_better_challenger_is_promoted_and_becomes_latest(setup):
    prices, root = setup
    before = (root / "LATEST").read_text()
    report = run_retrain(prices, root, holdout_days=60, min_gain=-10.0, cfg=CFG)
    assert report["promote"] and report["new_version"]
    assert (root / "LATEST").read_text() == report["new_version"] != before
    assert registry.load_bundle(root=root)["meta"]["version"] == report["new_version"]


def test_the_champion_is_kept_when_the_gain_is_too_small(setup):
    prices, root = setup
    before = (root / "LATEST").read_text()
    report = run_retrain(prices, root, holdout_days=60, min_gain=10.0, cfg=CFG)
    assert not report["promote"] and report["new_version"] is None
    assert (root / "LATEST").read_text() == before


def test_dry_run_writes_nothing_even_when_promotion_is_warranted(setup):
    prices, root = setup
    before = sorted(p.name for p in root.iterdir())
    report = run_retrain(prices, root, min_gain=-10.0, dry_run=True, cfg=CFG)
    assert report["promote"] and report["new_version"] is None
    assert sorted(p.name for p in root.iterdir()) == before


def test_report_compares_both_models_on_the_same_rows(setup):
    prices, root = setup
    report = run_retrain(prices, root, holdout_days=60, min_gain=10.0, cfg=CFG)
    assert report["champion"]["n_rows"] == report["challenger"]["n_rows"] > 0
    text = to_markdown(report)
    assert "| champion |" in text and "| challenger |" in text
    assert "keep the current model" in text
    assert report["champion_saw_holdout"] and "favours it" in text


def test_bundles_record_how_far_their_training_data_goes(setup):
    prices, root = setup
    meta = registry.load_bundle(root=root)["meta"]
    assert meta["data_through"] == str(pd.to_datetime(prices["date"]).max().date())
