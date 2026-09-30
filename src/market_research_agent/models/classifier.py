"""scikit-learn classifier: will the price be higher ~1 month (21 trading days) from now?"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_research_agent.models.features import DIR_FEATURES


def candidates(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1, max_iter=1000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        ),
        "hist_gb": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=150,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                )
            ]
        ),
    }


def select_and_fit(
    train: pd.DataFrame, val: pd.DataFrame, seed: int = 42
) -> tuple[str, Pipeline, dict[str, float]]:
    """Fit every candidate on train, pick the best by validation ROC-AUC."""
    val_auc: dict[str, float] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipe in candidates(seed).items():
        pipe.fit(train[DIR_FEATURES], train["y_dir"])
        val_auc[name] = float(
            roc_auc_score(val["y_dir"], pipe.predict_proba(val[DIR_FEATURES])[:, 1])
        )
        fitted[name] = pipe
    best = max(val_auc, key=val_auc.get)
    return best, fitted[best], val_auc


def predict_proba_up(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features[DIR_FEATURES])[:, 1]
