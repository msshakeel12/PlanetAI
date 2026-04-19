"""Unit tests for classification metric helpers."""

import numpy as np

from src.models.evaluation import compute_classification_metrics, find_best_threshold


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


def test_find_best_threshold_returns_expected_range() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.05, 0.2, 0.6, 0.85, 0.9, 0.1])
    result = find_best_threshold(y_true=y_true, y_prob=y_prob, metric="balanced_accuracy")

    assert 0.05 <= float(result["best_threshold"]) <= 0.95
    assert result["metric"] == "balanced_accuracy"
    assert len(result["threshold_summary"]) == 19


def test_compute_classification_metrics_one_class_safe() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob)

    assert metrics["confusion_matrix"] == [[4, 0], [0, 0]]
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["average_precision"])
