"""Reusable evaluation helpers for binary classification.

This module centralizes metric computation so baseline and CNN scripts report
comparable outputs and JSON structures.

Main public functions:
- ``compute_classification_metrics``: metric dictionary from probabilities.
- ``positive_class_scores_from_model``: score extraction adapter for sklearn.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probability-like scores for class 1.
        threshold: Decision threshold for deriving hard predictions.

    Returns:
        Dictionary containing accuracy, precision, recall, F1, ROC-AUC, and
        confusion matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        # ROC-AUC requires both classes in ``y_true``; handle tiny-set edge case.
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def positive_class_scores_from_model(model: Any, x: np.ndarray) -> np.ndarray:
    """Extract positive-class scores from sklearn estimators.

    Args:
        model: Fitted sklearn estimator.
        x: Feature matrix.

    Returns:
        Probability-like scores for positive class.

    Raises:
        ValueError: If the estimator lacks supported scoring interfaces.

    Notes:
        For estimators without ``predict_proba``, decision scores are mapped via
        logistic transform to provide a bounded probability-like quantity.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model must implement predict_proba or decision_function")
