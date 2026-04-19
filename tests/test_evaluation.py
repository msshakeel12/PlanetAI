"""Unit tests for classification metric helpers."""

import numpy as np

from src.models.evaluation import compute_classification_metrics


def test_compute_classification_metrics_keys() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2])
    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob)

    expected = {"accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"}
    assert expected.issubset(metrics.keys())


def test_confusion_matrix_shape() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.1, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob)
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (2, 2)
