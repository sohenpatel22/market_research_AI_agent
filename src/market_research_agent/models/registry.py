"""Versioned on-disk storage for trained model bundles.

Layout: artifacts/models/<version>/{lstm.pt, classifier.joblib, meta.json}, plus a LATEST pointer.
The bundles are tiny (KBs), so they are committed to git and ship with the deployed image.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from market_research_agent.models.lstm import LSTMConfig, VolLSTM, load_lstm, save_lstm

ARTIFACT_ROOT = Path("artifacts/models")


class ModelNotFoundError(FileNotFoundError):
    pass


def new_version() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def save_bundle(
    version: str,
    lstm: VolLSTM,
    lstm_cfg: LSTMConfig,
    classifier,
    meta: dict,
    root: Path = ARTIFACT_ROOT,
) -> Path:
    out = root / version
    out.mkdir(parents=True, exist_ok=True)
    save_lstm(lstm, lstm_cfg, out / "lstm.pt")
    joblib.dump(classifier, out / "classifier.joblib")
    (out / "meta.json").write_text(json.dumps({**meta, "version": version}, indent=2))
    (root / "LATEST").write_text(version)
    return out


def load_bundle(version: str | None = None, root: Path = ARTIFACT_ROOT) -> dict:
    if version is None:
        latest = root / "LATEST"
        if not latest.exists():
            raise ModelNotFoundError(f"No trained models in {root}. Run the training CLI first.")
        version = latest.read_text().strip()
    path = root / version
    if not (path / "meta.json").exists():
        raise ModelNotFoundError(f"Model bundle {version!r} not found in {root}.")
    lstm, lstm_cfg = load_lstm(path / "lstm.pt")
    return {
        "lstm": lstm,
        "lstm_cfg": lstm_cfg,
        "classifier": joblib.load(path / "classifier.joblib"),
        "meta": json.loads((path / "meta.json").read_text()),
        "path": path,
    }
