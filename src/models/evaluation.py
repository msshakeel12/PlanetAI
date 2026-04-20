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
    average_precision_score,
    balanced_accuracy_score,
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

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    support_negative = int(tn + fp)
    support_positive = int(tp + fn)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    has_both_classes = np.unique(y_true).size == 2

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if has_both_classes else float("nan"),
        "specificity": specificity,
        "support_negative": support_negative,
        "support_positive": support_positive,
    }

    # Backward-compatible aliases used in historical reports.
    metrics["f1score"] = metrics["f1_score"]
    metrics["rocauc"] = float("nan")
    metrics["confusionmatrix"] = metrics["confusion_matrix"]

    try:
        # ROC-AUC requires both classes in ``y_true``; skip when undefined.
        if has_both_classes:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["rocauc"] = metrics["roc_auc"]
        else:
            metrics["roc_auc"] = float("nan")
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["average_precision"] = (
            float(average_precision_score(y_true, y_prob)) if support_positive > 0 else float("nan")
        )
    except ValueError:
        metrics["average_precision"] = float("nan")

    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> dict[str, Any]:
    """Find the best probability threshold on validation predictions.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probability-like scores for class 1.
        metric: Optimization metric; one of ``f1`` or ``balanced_accuracy``.

    Returns:
        Dictionary with best threshold, best metric value, optimization metric,
        and a per-threshold summary table.

    Raises:
        ValueError: If ``metric`` is not supported.
    """
    metric_key = metric.strip().lower()
    if metric_key == "f1":
        metric_name = "f1_score"
    elif metric_key == "balanced_accuracy":
        metric_name = "balanced_accuracy"
    else:
        raise ValueError("metric must be one of: f1, balanced_accuracy")

    thresholds = np.arange(0.05, 0.951, 0.05)
    summary: list[dict[str, float]] = []
    best_threshold = 0.5
    best_metric_value = float("-inf")
    found_finite = False

    for threshold in thresholds:
        metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob, threshold=float(threshold))
        metric_value = float(metrics[metric_name])
        summary.append(
            {
                "threshold": float(threshold),
                "f1_score": float(metrics["f1_score"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "metric_value": metric_value,
            }
        )
        if np.isfinite(metric_value) and metric_value > best_metric_value:
            found_finite = True
            best_metric_value = metric_value
            best_threshold = float(threshold)

    if not found_finite:
        best_metric_value = float("nan")
        best_threshold = 0.5

    return {
        "best_threshold": best_threshold,
        "best_metric_value": float(best_metric_value),
        "metric": metric_key,
        "threshold_summary": summary,
    }


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
