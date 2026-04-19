"""Plotting utilities for classification evaluation.

This module generates standardized figures used across baseline and CNN runs so
comparisons remain visually consistent.

Main public functions:
- ``plot_confusion_matrix``
- ``plot_roc_curve``
- ``plot_precision_recall_curve``
- ``plot_model_comparison``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, threshold: float = 0.5) -> Path:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for positive class.
        output_path: Destination image path.
        threshold: Probability threshold for hard predictions.

    Returns:
        Path to saved figure.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Save into caller-provided report directories for reproducible artifact layout.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> Path:
    """Plot and save ROC curve.

    Args:
        y_true: Ground-truth labels.
        y_prob: Positive-class probabilities.
        output_path: Destination image path.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> Path:
    """Plot and save precision-recall curve.

    Args:
        y_true: Ground-truth labels.
        y_prob: Positive-class probabilities.
        output_path: Destination image path.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Precision-Recall Curve")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison(metrics_by_model: dict[str, dict[str, float]], output_path: Path) -> Path:
    """Plot comparison bars for F1 and ROC-AUC across models.

    Args:
        metrics_by_model: Mapping from model name to metric dictionary.
        output_path: Destination image path.

    Returns:
        Path to saved figure.

    Notes:
        F1 and ROC-AUC are shown together to balance thresholded and
        threshold-independent views of model quality.
    """
    models = list(metrics_by_model.keys())
    f1_values = [metrics_by_model[m].get("f1_score", np.nan) for m in models]
    auc_values = [metrics_by_model[m].get("roc_auc", np.nan) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, f1_values, width, label="F1")
    ax.bar(x + width / 2, auc_values, width, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
