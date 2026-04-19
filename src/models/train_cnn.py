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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.cnn_model import TransitCNN
from src.models.evaluation import compute_classification_metrics
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
) -> tuple[TransitCNN, list[dict[str, float]], float, torch.device]:
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
        Tuple ``(model, history, best_val_loss, device)``.
    """
    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitCNN(input_length=x_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
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

    return model, history, best_val_loss, device


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
    tracker: ExperimentTracker | None = None,
) -> dict[str, float]:
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
        tracker: Optional experiment tracker.

    Returns:
        Metric dictionary for final test evaluation.
    """
    # ------------------------------ Load dataset ------------------------------
    payload = np.load(sequences_path)
    x = payload["sequences"].astype(np.float32)
    y = payload["labels"].astype(np.int64)

    # ------------------------------- Data split -------------------------------
    x_train, x_test, y_train, y_test = _safe_train_test_split(
        x=x,
        y=y,
        test_size=0.2,
        random_seed=random_seed,
    )
    x_train, x_val, y_train, y_val = _safe_train_test_split(
        x=x_train,
        y=y_train,
        test_size=0.25,
        random_seed=random_seed,
    )

    test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

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
            trial_model, _, trial_val_loss, _ = _train_with_validation(
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
    model, history, best_val_loss, device = _train_with_validation(
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

    y_true_test, y_prob_test = _predict_probs(model, test_loader, device)

    # ---------------------------- Final evaluation ----------------------------
    metrics = compute_classification_metrics(y_true=y_true_test.astype(int), y_prob=y_prob_test)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["epochs"] = int(epochs)

    # ----------------------------- Save artifacts -----------------------------
    save_json(metrics, output_dir / "cnn_metrics.json")
    save_json({"history": history}, output_dir / "cnn_training_history.json")

    plot_confusion_matrix(y_true_test, y_prob_test, figures_dir / "cnn_confusion_matrix.png")
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
        tracker=tracker,
    )
    print("CNN training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
