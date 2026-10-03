"""PyTorch LSTM that forecasts next-5-day log annualized volatility"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from market_research_agent.models.features import VOL_FEATURES


@dataclass
class LSTMConfig:
    window: int = 22
    hidden: int = 32
    layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 60
    patience: int = 8
    seed: int = 42


class VolLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def fit_stats(frame: pd.DataFrame, train_mask: np.ndarray) -> dict:
    """Per-ticker standardization stats from training rows only"""
    tr = frame[train_mask]
    return {
        "x_mean": tr[VOL_FEATURES].mean().tolist(),
        "x_std": (tr[VOL_FEATURES].std() + 1e-8).tolist(),
        "y_mean": float(tr["y_vol"].mean()),
        "y_std": float(tr["y_vol"].std() + 1e-8),
    }


def make_samples(
    frame: pd.DataFrame, stats: dict, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows over the normalized features"""
    feats = (frame[VOL_FEATURES].to_numpy() - np.array(stats["x_mean"])) / np.array(stats["x_std"])
    if len(feats) < window:
        return np.empty((0, window, len(VOL_FEATURES))), np.empty(0), np.empty(0, dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(feats, (window, feats.shape[1]))[:, 0]
    t = np.arange(window - 1, len(feats))
    ok = np.isfinite(windows).all(axis=(1, 2))
    y = (frame["y_vol"].to_numpy() - stats["y_mean"]) / stats["y_std"]
    return windows[ok].astype(np.float32), y[t][ok].astype(np.float32), t[ok]


def train_lstm(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    cfg: LSTMConfig,
) -> tuple[VolLSTM, dict[str, list[float]]]:
    """Train with Adam + MSE and early stopping on validation loss (best weights restored)"""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x_tr, y_tr = (torch.from_numpy(a) for a in train)
    x_va, y_va = (torch.from_numpy(a) for a in val)

    model = VolLSTM(x_tr.shape[-1], cfg.hidden, cfg.layers, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    # Shuffling within the training set is fine: windows are already built in time order.
    loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best, best_state, stale = float("inf"), None, 0
    for _ in range(cfg.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_va), y_va).item()
        history["train_loss"].append(total / len(x_tr))
        history["val_loss"].append(val_loss)
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_lstm(model: VolLSTM, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x.astype(np.float32))).numpy()


def denormalize(pred_norm: np.ndarray, stats: dict) -> np.ndarray:
    return pred_norm * stats["y_std"] + stats["y_mean"]


def save_lstm(model: VolLSTM, cfg: LSTMConfig, path: Path) -> None:
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, path)


def load_lstm(path: Path) -> tuple[VolLSTM, LSTMConfig]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    cfg = LSTMConfig(**blob["config"])
    model = VolLSTM(len(VOL_FEATURES), cfg.hidden, cfg.layers, cfg.dropout)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg
