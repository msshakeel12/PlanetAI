"""Train a 1D CNN on fixed-length light-curve sequences.

This module handles sequence split logic, training/validation monitoring,
checkpointing, optional hyperparameter search, evaluation, plotting, and
experiment logging.

Main public functions:
- ``run_cnn_training``: programmatic training interface.
- ``main``: command-line entry point.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.cnn_model import TransitCNN
from src.models.evaluation import compute_classification_metrics, find_best_threshold
from src.models.plotting import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.utils.io import save_json
from src.utils.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_project_directories
from src.utils.experiment_tracking import ExperimentTracker


class SequenceDataset(Dataset):
    """Torch dataset wrapper for sequence classification.

    Args:
        sequences: Array shaped ``(n_samples, sequence_length)``.
        labels: Binary label array shaped ``(n_samples,)``.
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def _safe_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays with adaptive stratification for tiny datasets.

    Args:
        x: Feature/sequence array.
        y: Label array.
        test_size: Desired split fraction.
        random_seed: RNG seed.

    Returns:
        Tuple ``x_train, x_test, y_train, y_test``.

    Notes:
        This helper preserves training continuity in low-sample real-ingestion
        runs by relaxing strict stratification when required.
    """
    y_arr = np.asarray(y)
    n_samples = y_arr.shape[0]
    class_values, class_counts = np.unique(y_arr, return_counts=True)
    n_classes = class_values.shape[0]

    raw_test_n = max(1, int(round(test_size * n_samples)))
    min_required = max(1, n_classes)
    test_n = max(raw_test_n, min_required)
    test_n = min(test_n, n_samples - 1)
    eff_test_size = test_n / n_samples

    use_stratify = np.all(class_counts >= 2) and test_n >= n_classes
    stratify_arg = y_arr if use_stratify else None

    return train_test_split(
        x,
        y_arr,
        test_size=eff_test_size,
        random_state=random_seed,
        stratify=stratify_arg,
    )


def _grouped_split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices with group isolation and best-effort stratification.

    Args:
        y: Binary labels.
        groups: Group identifiers (for example ``kepid``).
        test_size: Desired held-out fraction.
        random_seed: RNG seed.

    Returns:
        Tuple ``(train_indices, test_indices)``.

    Notes:
        If stratified grouped splitting is impossible (common on tiny datasets),
        this helper degrades to grouped shuffle split and finally row-level
        splitting while keeping execution stable.
    """
    y_arr = np.asarray(y).astype(int)
    groups_arr = np.asarray(groups)
    n_samples = y_arr.shape[0]
    if n_samples < 2:
        idx = np.arange(n_samples)
        return idx, np.array([], dtype=int)

    unique_groups = np.unique(groups_arr)
    if unique_groups.size < 2:
        warnings.warn(
            "Strong warning: grouped split fallback to row-level split because fewer than 2 unique groups are available.",
            RuntimeWarning,
            stacklevel=2,
        )
        idx = np.arange(n_samples)
        train_idx, test_idx, _, _ = _safe_train_test_split(idx, y_arr, test_size=test_size, random_seed=random_seed)
        return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)

    desired_splits = int(round(1.0 / max(test_size, 1e-6)))
    candidate_splits = [v for v in [5, 4, 3, 2, desired_splits] if 2 <= v <= unique_groups.size]
    candidate_splits = list(dict.fromkeys(candidate_splits))

    x_dummy = np.zeros((n_samples, 1), dtype=float)
    best_pair: tuple[np.ndarray, np.ndarray] | None = None
    best_error = float("inf")

    for n_splits in candidate_splits:
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            for train_idx, test_idx in sgkf.split(x_dummy, y_arr, groups_arr):
                holdout_fraction = len(test_idx) / n_samples
                size_error = abs(holdout_fraction - test_size)
                if size_error < best_error:
                    best_error = size_error
                    best_pair = (train_idx, test_idx)
        except ValueError:
            continue

    if best_pair is not None:
        return best_pair

    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(gss.split(x_dummy, y_arr, groups_arr))
        return train_idx, test_idx
    except ValueError:
        warnings.warn(
            "Strong warning: grouped split fallback to row-level split because grouped split failed on this dataset.",
            RuntimeWarning,
            stacklevel=2,
        )
        idx = np.arange(n_samples)
        train_idx, test_idx, _, _ = _safe_train_test_split(idx, y_arr, test_size=test_size, random_seed=random_seed)
        return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def _grouped_train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create grouped train/validation/test splits.

    Args:
        x: Feature/sequence matrix.
        y: Binary labels.
        groups: Group IDs used to prevent leakage across splits.
        test_size: Fraction for held-out test split.
        val_size: Fraction (of remaining train data) used for validation.
        random_seed: RNG seed.

    Returns:
        Tuple with ``x_train, x_val, x_test, y_train, y_val, y_test,
        groups_train, groups_val, groups_test``.
    """
    idx = np.arange(x.shape[0])
    train_idx, test_idx = _grouped_split_indices(y=y, groups=groups, test_size=test_size, random_seed=random_seed)

    # If grouped split failed to produce a holdout on tiny data, force a row-level fallback.
    if test_idx.size == 0 and x.shape[0] >= 2:
        warnings.warn(
            "Strong warning: test split was empty after grouped split; forcing row-level fallback for test holdout.",
            RuntimeWarning,
            stacklevel=2,
        )
        split = max(1, int(round((1.0 - test_size) * x.shape[0])))
        train_idx = idx[:split]
        test_idx = idx[split:]

    rem_x = x[train_idx]
    rem_y = y[train_idx]
    rem_groups = groups[train_idx]

    train_rel_idx, val_rel_idx = _grouped_split_indices(
        y=rem_y,
        groups=rem_groups,
        test_size=val_size,
        random_seed=random_seed + 1,
    )
    if val_rel_idx.size == 0 and rem_x.shape[0] >= 2:
        warnings.warn(
            "Strong warning: validation split was empty after grouped split; forcing row-level fallback for validation holdout.",
            RuntimeWarning,
            stacklevel=2,
        )
        split = max(1, int(round((1.0 - val_size) * rem_x.shape[0])))
        train_rel_idx = np.arange(rem_x.shape[0])[:split]
        val_rel_idx = np.arange(rem_x.shape[0])[split:]

    final_train_idx = train_idx[train_rel_idx]
    final_val_idx = train_idx[val_rel_idx]

    return (
        x[final_train_idx],
        x[final_val_idx],
        x[test_idx],
        y[final_train_idx],
        y[final_val_idx],
        y[test_idx],
        groups[final_train_idx],
        groups[final_val_idx],
        groups[test_idx],
    )


def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Compute mean loss for a dataloader in evaluation mode.

    Args:
        model: Neural network model.
        loader: DataLoader to evaluate.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Mean loss over all items.
    """
    model.eval()
    loss_total = 0.0
    n_items = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_size = xb.size(0)
            loss_total += loss.item() * batch_size
            n_items += batch_size
    return loss_total / max(1, n_items)


def _predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Generate ground-truth labels and predicted probabilities.

    Args:
        model: Trained model.
        loader: DataLoader for inference.
        device: Torch device.

    Returns:
        Tuple ``(y_true, y_prob)``.
    """
    model.eval()
    probs: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            batch_probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(batch_probs)
            truths.append(yb.numpy().reshape(-1))
    return np.concatenate(truths), np.concatenate(probs)


def _train_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    checkpoint_path: Path | None,
) -> tuple[TransitCNN, list[dict[str, float]], float, torch.device, dict[str, float | int]]:
    """Train a CNN while monitoring validation loss.

    Args:
        x_train: Training sequences.
        y_train: Training labels.
        x_val: Validation sequences.
        y_val: Validation labels.
        random_seed: RNG seed.
        batch_size: Batch size.
        learning_rate: Optimizer learning rate.
        epochs: Training epochs.
        checkpoint_path: Optional path for best-model checkpoint.

    Returns:
        Tuple ``(model, history, best_val_loss, device, training_info)``.
    """
    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitCNN(input_length=x_train.shape[1]).to(device)
    n_positive = int(np.sum(np.asarray(y_train) == 1))
    n_negative = int(np.sum(np.asarray(y_train) == 0))
    if n_positive > 0 and n_negative > 0:
        pos_weight_value = float(n_negative / max(n_positive, 1))
    else:
        pos_weight_value = 1.0

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_items = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size_actual = xb.size(0)
            running_loss += loss.item() * batch_size_actual
            n_items += batch_size_actual

        train_loss = running_loss / max(1, n_items)
        val_loss = _evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Keep an in-memory best state to support training runs without disk checkpoints.
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    training_info: dict[str, float | int] = {
        "pos_weight": float(pos_weight_value),
        "train_positive": int(n_positive),
        "train_negative": int(n_negative),
    }
    return model, history, best_val_loss, device, training_info


def _sample_hparam_candidates(
    search_space: dict[str, list[float] | list[int]],
    trials: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    """Randomly sample hyperparameter combinations from discrete grids.

    Args:
        search_space: Mapping from hyperparameter name to candidate values.
        trials: Number of random candidates to sample.
        rng: Random generator.

    Returns:
        List of sampled hyperparameter dictionaries.

    Raises:
        ValueError: If required keys are missing or empty.
    """
    keys = ["batch_size", "learning_rate", "epochs"]
    for key in keys:
        if key not in search_space or not search_space[key]:
            raise ValueError(f"Missing non-empty search-space key: {key}")

    candidates: list[dict[str, float | int]] = []
    for _ in range(trials):
        candidates.append(
            {
                "batch_size": int(rng.choice(search_space["batch_size"])),
                "learning_rate": float(rng.choice(search_space["learning_rate"])),
                "epochs": int(rng.choice(search_space["epochs"])),
            }
        )
    return candidates


def run_cnn_training(
    sequences_path: Path,
    output_dir: Path,
    figures_dir: Path,
    random_seed: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    tune: bool,
    tune_trials: int,
    tune_search_space: dict[str, list[float] | list[int]],
    task_mode: str = "disposition_binary",
    earth_size_max_radius: float = 1.5,
    threshold_metric: str = "balanced_accuracy",
    tracker: ExperimentTracker | None = None,
) -> dict[str, Any]:
    """Train/evaluate CNN and persist artifacts.

    Args:
        sequences_path: Input NPZ containing sequences and labels.
        output_dir: Artifact output directory.
        figures_dir: Figure output directory.
        random_seed: RNG seed.
        batch_size: Base batch size.
        learning_rate: Base learning rate.
        epochs: Base epoch count.
        tune: Enable random hyperparameter search.
        tune_trials: Number of sampled search trials.
        tune_search_space: Candidate values for tunable parameters.
        task_mode: Labeling task mode used during dataset construction.
        earth_size_max_radius: Radius threshold used for Earth-sized task.
        threshold_metric: Validation metric used for threshold selection.
        tracker: Optional experiment tracker.

    Returns:
        Metric dictionary for final test evaluation.
    """
    # ------------------------------ Load dataset ------------------------------
    payload = np.load(sequences_path)
    x = payload["sequences"].astype(np.float32)
    y = payload["labels"].astype(np.int64)
    if "groups" in payload.files:
        groups = payload["groups"]
    else:
        warnings.warn(
            "Strong warning: input sequence NPZ has no 'groups' array; falling back to row-level groups and leakage-safe splitting cannot be guaranteed.",
            RuntimeWarning,
            stacklevel=2,
        )
        groups = np.arange(x.shape[0])

    # ------------------------------- Data split -------------------------------
    x_train, x_val, x_test, y_train, y_val, y_test, groups_train, groups_val, groups_test = _grouped_train_val_test_split(
        x=x,
        y=y,
        groups=np.asarray(groups),
        test_size=0.2,
        val_size=0.25,
        random_seed=random_seed,
    )
    if x_train.shape[0] == 0 or x_val.shape[0] == 0 or x_test.shape[0] == 0:
        raise ValueError(
            "Unable to create non-empty train/validation/test splits. Increase dataset size or adjust split strategy."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "cnn_best.pt"

    # ------------------------ Optional hyperparameter search ------------------
    if tune:
        rng = np.random.default_rng(random_seed)
        candidates = _sample_hparam_candidates(tune_search_space, tune_trials, rng)
        best_params: dict[str, float | int] | None = None
        best_trial_loss = float("inf")

        for idx, params in enumerate(candidates):
            trial_model, _, trial_val_loss, _, _ = _train_with_validation(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                random_seed=random_seed + idx,
                batch_size=int(params["batch_size"]),
                learning_rate=float(params["learning_rate"]),
                epochs=int(params["epochs"]),
                checkpoint_path=None,
            )
            del trial_model
            if trial_val_loss < best_trial_loss:
                best_trial_loss = trial_val_loss
                best_params = params

        if best_params is not None:
            batch_size = int(best_params["batch_size"])
            learning_rate = float(best_params["learning_rate"])
            epochs = int(best_params["epochs"])

    # ------------------------------- Final train ------------------------------
    model, history, best_val_loss, device, training_info = _train_with_validation(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        random_seed=random_seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        checkpoint_path=best_ckpt,
    )

    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    y_true_val, y_prob_val = _predict_probs(model, val_loader, device)
    threshold_result = find_best_threshold(
        y_true=y_true_val.astype(int),
        y_prob=y_prob_val,
        metric=threshold_metric,
    )
    threshold_used = float(threshold_result["best_threshold"])

    y_true_test, y_prob_test = _predict_probs(model, test_loader, device)

    # ---------------------------- Final evaluation ----------------------------
    metrics = compute_classification_metrics(
        y_true=y_true_test.astype(int),
        y_prob=y_prob_test,
        threshold=threshold_used,
    )
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["epochs"] = int(epochs)
    metrics["threshold_used"] = threshold_used
    metrics["threshold_metric"] = str(threshold_result["metric"])
    metrics["threshold_metric_value"] = float(threshold_result["best_metric_value"])
    metrics["task_mode"] = task_mode
    metrics["earth_size_max_radius"] = float(earth_size_max_radius)
    metrics["class_counts"] = {
        "train": {"positive": int(np.sum(y_train == 1)), "negative": int(np.sum(y_train == 0))},
        "val": {"positive": int(np.sum(y_val == 1)), "negative": int(np.sum(y_val == 0))},
        "test": {"positive": int(np.sum(y_test == 1)), "negative": int(np.sum(y_test == 0))},
    }
    metrics["train_pos_weight"] = float(training_info["pos_weight"])
    metrics["group_counts"] = {
        "train": int(np.unique(groups_train).shape[0]),
        "val": int(np.unique(groups_val).shape[0]),
        "test": int(np.unique(groups_test).shape[0]),
    }

    # ----------------------------- Save artifacts -----------------------------
    save_json(metrics, output_dir / "cnn_metrics.json")
    save_json(
        {
            "history": history,
            "threshold_tuning": threshold_result,
        },
        output_dir / "cnn_training_history.json",
    )

    plot_confusion_matrix(y_true_test, y_prob_test, figures_dir / "cnn_confusion_matrix.png", threshold=threshold_used)
    plot_roc_curve(y_true_test, y_prob_test, figures_dir / "cnn_roc_curve.png")
    plot_precision_recall_curve(y_true_test, y_prob_test, figures_dir / "cnn_precision_recall.png")

    # Track run metadata for reproducibility and future comparisons.
    if tracker is not None:
        tracker.log_run(
            stage="cnn",
            params={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "random_seed": random_seed,
                "tune": tune,
                "tune_trials": tune_trials,
                "threshold_metric": threshold_metric,
                "task_mode": task_mode,
                "earth_size_max_radius": earth_size_max_radius,
            },
            metrics=metrics,
            artifacts={
                "checkpoint": str(best_ckpt),
                "metrics": str(output_dir / "cnn_metrics.json"),
                "history": str(output_dir / "cnn_training_history.json"),
            },
        )

    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for CNN training.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        type=Path,
        default=PROCESSED_DATA_DIR / "sequences.npz",
        help="Input sequence NPZ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "models",
        help="Output directory for model artifacts.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory for figures.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="Enable random hyperparameter search before final training.")
    parser.add_argument("--tune-trials", type=int, default=8)
    parser.add_argument(
        "--tune-batch-sizes",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Candidate batch sizes for CNN search.",
    )
    parser.add_argument(
        "--tune-learning-rates",
        nargs="+",
        type=float,
        default=[0.0003, 0.001, 0.003],
        help="Candidate learning rates for CNN search.",
    )
    parser.add_argument(
        "--tune-epochs",
        nargs="+",
        type=int,
        default=[10, 15, 20],
        help="Candidate epoch counts for CNN search.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Optional experiment tracking directory for JSON logs.",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        default="disposition_binary",
        choices=["disposition_binary", "earth_sized_binary"],
        help="Task mode metadata for persisted CNN metrics.",
    )
    parser.add_argument(
        "--earth-size-max-radius",
        type=float,
        default=1.5,
        help="Earth-size radius threshold used by Earth-sized task mode.",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="balanced_accuracy",
        choices=["f1", "balanced_accuracy"],
        help="Validation metric used to tune decision threshold.",
    )
    return parser


def main() -> None:
    """Run CNN training from the command line.

    Returns:
        None.
    """
    ensure_project_directories()
    args = build_parser().parse_args()
    tracker = ExperimentTracker(args.experiment_dir) if args.experiment_dir is not None else None
    metrics = run_cnn_training(
        sequences_path=args.sequences,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        tune=args.tune,
        tune_trials=args.tune_trials,
        tune_search_space={
            "batch_size": args.tune_batch_sizes,
            "learning_rate": args.tune_learning_rates,
            "epochs": args.tune_epochs,
        },
        task_mode=args.task_mode,
        earth_size_max_radius=args.earth_size_max_radius,
        threshold_metric=args.threshold_metric,
        tracker=tracker,
    )
    print("CNN training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
