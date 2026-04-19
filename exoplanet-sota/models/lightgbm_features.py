"""LightGBM baseline training on tsfresh features."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def train_lightgbm_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> tuple[LGBMClassifier, dict[str, float]]:
    """Train LightGBM classifier and compute test metrics.

    Args:
        x_train: Train feature matrix.
        y_train: Train labels.
        x_test: Test feature matrix.
        y_test: Test labels.
        random_state: RNG seed.

    Returns:
        Tuple of trained model and metric dictionary.
    """
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) == 2 else float("nan"),
        "confusion_matrix": cm,
    }
    return model, metrics
