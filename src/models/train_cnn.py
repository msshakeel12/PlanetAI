"""Train a 1D CNN on fixed-length light-curve sequences.

This module handles grouped split logic, training/validation monitoring,
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
from typing import Any, Callable

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
from src.utils.experiment_tracking import ExperimentTracker
from src.utils.io import save_json
from src.utils.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_project_directories


class SequenceDataset(Dataset):
    """Torch dataset wrapper for legacy sequence classification.

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


class MultiViewDataset(Dataset):
    """Torch dataset for multiview CNN training.

    Args:
        global_views: Array shaped ``(n_samples, global_view_length)``.
        local_views: Array shaped ``(n_samples, local_view_length)``.
        labels: Binary label array shaped ``(n_samples,)``.
        aux_features: Optional array shaped ``(n_samples, n_aux_features)``.
    """

    def __init__(
        self,
        global_views: np.ndarray,
        local_views: np.ndarray,
        labels: np.ndarray,
        aux_features: np.ndarray | None = None,
        odd_even_views: np.ndarray | None = None,
        secondary_views: np.ndarray | None = None,
    ) -> None:
        self.global_views = torch.tensor(global_views, dtype=torch.float32).unsqueeze(1)
        self.local_views = torch.tensor(local_views, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.aux_features = None
        self.odd_even_views = None
        self.secondary_views = None
        if aux_features is not None:
            self.aux_features = torch.tensor(aux_features, dtype=torch.float32)
        if odd_even_views is not None:
            self.odd_even_views = torch.tensor(odd_even_views, dtype=torch.float32)
        if secondary_views is not None:
            self.secondary_views = torch.tensor(secondary_views, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return int(self.global_views.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "global_view": self.global_views[idx],
            "local_view": self.local_views[idx],
            "label": self.labels[idx],
        }
        if self.aux_features is not None:
            item["aux_features"] = self.aux_features[idx]
        if self.odd_even_views is not None:
            item["odd_even_view"] = self.odd_even_views[idx]
        if self.secondary_views is not None:
            item["secondary_view"] = self.secondary_views[idx]
        return item


def _safe_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays with adaptive stratification for tiny datasets."""
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
    """Split indices with group isolation and best-effort stratification."""
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


def _grouped_train_val_test_indices(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create grouped train/validation/test split indices."""
    idx = np.arange(y.shape[0])
    train_idx, test_idx = _grouped_split_indices(y=y, groups=groups, test_size=test_size, random_seed=random_seed)

    if test_idx.size == 0 and y.shape[0] >= 2:
        warnings.warn(
            "Strong warning: test split was empty after grouped split; forcing row-level fallback for test holdout.",
            RuntimeWarning,
            stacklevel=2,
        )
        split = max(1, int(round((1.0 - test_size) * y.shape[0])))
        train_idx = idx[:split]
        test_idx = idx[split:]

    rem_y = y[train_idx]
    rem_groups = groups[train_idx]
    train_rel_idx, val_rel_idx = _grouped_split_indices(
        y=rem_y,
        groups=rem_groups,
        test_size=val_size,
        random_seed=random_seed + 1,
    )

    if val_rel_idx.size == 0 and rem_y.shape[0] >= 2:
        warnings.warn(
            "Strong warning: validation split was empty after grouped split; forcing row-level fallback for validation holdout.",
            RuntimeWarning,
            stacklevel=2,
        )
        split = max(1, int(round((1.0 - val_size) * rem_y.shape[0])))
        train_rel_idx = np.arange(rem_y.shape[0])[:split]
        val_rel_idx = np.arange(rem_y.shape[0])[split:]

    final_train_idx = train_idx[train_rel_idx]
    final_val_idx = train_idx[val_rel_idx]
    return final_train_idx, final_val_idx, test_idx


def _grouped_train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create grouped train/validation/test splits.

    This wrapper is kept for backward compatibility with existing tests and
    callers that expect split arrays instead of indices.
    """
    train_idx, val_idx, test_idx = _grouped_train_val_test_indices(
        y=np.asarray(y),
        groups=np.asarray(groups),
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
    )
    return (
        x[train_idx],
        x[val_idx],
        x[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
        groups[train_idx],
        groups[val_idx],
        groups[test_idx],
    )


def _unpack_batch(
    batch: tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Convert dataloader batch to model inputs and labels on target device."""
    if isinstance(batch, dict):
        yb = batch["label"].to(device)
        inputs = {
            "global_view": batch["global_view"].to(device),
            "local_view": batch["local_view"].to(device),
        }
        if "aux_features" in batch:
            inputs["aux_features"] = batch["aux_features"].to(device)
        if "odd_even_view" in batch:
            inputs["odd_even_view"] = batch["odd_even_view"].to(device)
        if "secondary_view" in batch:
            inputs["secondary_view"] = batch["secondary_view"].to(device)
        return inputs, yb, int(yb.size(0))

    xb, yb = batch
    return {"x": xb.to(device)}, yb.to(device), int(xb.size(0))


def _forward_from_inputs(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run model forward pass for either legacy or multiview batch formats."""
    if "x" in inputs:
        return model(x=inputs["x"])
    return model(
        global_view=inputs["global_view"],
        local_view=inputs["local_view"],
        odd_even_view=inputs.get("odd_even_view"),
        secondary_view=inputs.get("secondary_view"),
        aux_features=inputs.get("aux_features"),
    )


def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Compute mean loss for a dataloader in evaluation mode."""
    model.eval()
    loss_total = 0.0
    n_items = 0
    with torch.no_grad():
        for batch in loader:
            inputs, yb, batch_size = _unpack_batch(batch, device)
            logits = _forward_from_inputs(model, inputs)
            loss = criterion(logits, yb)
            loss_total += loss.item() * batch_size
            n_items += batch_size
    return loss_total / max(1, n_items)


def _predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Generate ground-truth labels and predicted probabilities."""
    model.eval()
    probs: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs, yb, _ = _unpack_batch(batch, device)
            logits = _forward_from_inputs(model, inputs)
            batch_probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(batch_probs)
            truths.append(yb.cpu().numpy().reshape(-1))
    return np.concatenate(truths), np.concatenate(probs)


def _train_with_validation(
    train_dataset: Dataset,
    val_dataset: Dataset,
    y_train: np.ndarray,
    build_model_fn: Callable[[], TransitCNN],
    random_seed: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    checkpoint_path: Path | None,
) -> tuple[TransitCNN, list[dict[str, float]], float, torch.device, dict[str, float | int]]:
    """Train a CNN while monitoring validation loss."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_fn().to(device)

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

        for batch in train_loader:
            inputs, yb, batch_size_actual = _unpack_batch(batch, device)
            optimizer.zero_grad()
            logits = _forward_from_inputs(model, inputs)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size_actual
            n_items += batch_size_actual

        train_loss = running_loss / max(1, n_items)
        val_loss = _evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    """Randomly sample hyperparameter combinations from discrete grids."""
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


def _resolve_model_mode(
    requested_mode: str,
    payload_files: set[str],
) -> tuple[str, bool]:
    """Resolve effective model mode from user request and NPZ contents."""
    has_multiview = {"global_views", "local_views"}.issubset(payload_files)

    if requested_mode == "auto":
        return ("multiview" if has_multiview else "legacy"), has_multiview
    if requested_mode == "legacy":
        return "legacy", has_multiview
    if requested_mode == "multiview":
        if not has_multiview:
            raise ValueError(
                "model_mode='multiview' requires 'global_views' and 'local_views' arrays in the sequence NPZ."
            )
        return "multiview", has_multiview
    raise ValueError("model_mode must be one of: auto, legacy, multiview")


def _normalize_aux_features(
    aux_train: np.ndarray,
    aux_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit normalization on train aux and apply to another split."""
    mean = np.nanmean(aux_train, axis=0)
    std = np.nanstd(aux_train, axis=0)
    std = np.where(std == 0.0, 1.0, std)

    train_norm = (aux_train - mean) / std
    other_norm = (aux_other - mean) / std
    train_norm = np.nan_to_num(train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    other_norm = np.nan_to_num(other_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return train_norm.astype(np.float32), other_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


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
    model_mode: str = "auto",
    use_odd_even_branch: bool = True,
    use_secondary_branch: bool = True,
    disable_aux_features: bool = False,
    fusion_hidden_dim: int = 96,
) -> dict[str, Any]:
    """Train/evaluate CNN and persist artifacts."""
    payload = np.load(sequences_path)
    payload_files = set(payload.files)

    x = np.nan_to_num(payload["sequences"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = payload["labels"].astype(np.int64)
    if "groups" in payload.files:
        groups = np.asarray(payload["groups"])
    else:
        warnings.warn(
            "Strong warning: input sequence NPZ has no 'groups' array; falling back to row-level groups and leakage-safe splitting cannot be guaranteed.",
            RuntimeWarning,
            stacklevel=2,
        )
        groups = np.arange(x.shape[0])

    effective_mode, has_multiview_arrays = _resolve_model_mode(model_mode, payload_files)

    train_idx, val_idx, test_idx = _grouped_train_val_test_indices(
        y=y,
        groups=np.asarray(groups),
        test_size=0.2,
        val_size=0.25,
        random_seed=random_seed,
    )
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            "Unable to create non-empty train/validation/test splits. Increase dataset size or adjust split strategy."
        )

    x_train = x[train_idx]
    x_val = x[val_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    groups_train = groups[train_idx]
    groups_val = groups[val_idx]
    groups_test = groups[test_idx]

    aux_mean: np.ndarray | None = None
    aux_std: np.ndarray | None = None

    if effective_mode == "multiview":
        global_views = np.nan_to_num(payload["global_views"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        local_views = np.nan_to_num(payload["local_views"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        global_train = global_views[train_idx]
        global_val = global_views[val_idx]
        global_test = global_views[test_idx]

        local_train = local_views[train_idx]
        local_val = local_views[val_idx]
        local_test = local_views[test_idx]

        has_odd_even = "odd_even_views" in payload.files
        has_secondary = "secondary_views" in payload.files
        use_odd_even_effective = bool(use_odd_even_branch and has_odd_even)
        use_secondary_effective = bool(use_secondary_branch and has_secondary)

        if use_odd_even_branch and not has_odd_even:
            warnings.warn(
                "Strong warning: odd/even branch requested but 'odd_even_views' is missing; disabling odd/even branch.",
                RuntimeWarning,
                stacklevel=2,
            )
        if use_secondary_branch and not has_secondary:
            warnings.warn(
                "Strong warning: secondary branch requested but 'secondary_views' is missing; disabling secondary branch.",
                RuntimeWarning,
                stacklevel=2,
            )

        odd_even_train: np.ndarray | None = None
        odd_even_val: np.ndarray | None = None
        odd_even_test: np.ndarray | None = None
        if use_odd_even_effective:
            odd_even = np.nan_to_num(payload["odd_even_views"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            odd_even_train = odd_even[train_idx]
            odd_even_val = odd_even[val_idx]
            odd_even_test = odd_even[test_idx]

        secondary_train: np.ndarray | None = None
        secondary_val: np.ndarray | None = None
        secondary_test: np.ndarray | None = None
        if use_secondary_effective:
            secondary = np.nan_to_num(payload["secondary_views"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            secondary_train = secondary[train_idx]
            secondary_val = secondary[val_idx]
            secondary_test = secondary[test_idx]

        aux_train: np.ndarray | None = None
        aux_val: np.ndarray | None = None
        aux_test: np.ndarray | None = None
        use_aux_features_effective = bool((not disable_aux_features) and ("aux_features" in payload.files))
        if use_aux_features_effective:
            aux = np.nan_to_num(payload["aux_features"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            aux_train_raw = aux[train_idx]
            aux_val_raw = aux[val_idx]
            aux_test_raw = aux[test_idx]
            aux_train, aux_val, aux_mean, aux_std = _normalize_aux_features(aux_train_raw, aux_val_raw)
            _, aux_test, _, _ = _normalize_aux_features(aux_train_raw, aux_test_raw)

        train_dataset = MultiViewDataset(
            global_views=global_train,
            local_views=local_train,
            labels=y_train,
            aux_features=aux_train,
            odd_even_views=odd_even_train,
            secondary_views=secondary_train,
        )
        val_dataset = MultiViewDataset(
            global_views=global_val,
            local_views=local_val,
            labels=y_val,
            aux_features=aux_val,
            odd_even_views=odd_even_val,
            secondary_views=secondary_val,
        )
        test_dataset = MultiViewDataset(
            global_views=global_test,
            local_views=local_test,
            labels=y_test,
            aux_features=aux_test,
            odd_even_views=odd_even_test,
            secondary_views=secondary_test,
        )

        aux_dim = int(aux_train.shape[1]) if aux_train is not None else 0
        odd_even_len = int(odd_even_train.shape[2]) if odd_even_train is not None else None
        secondary_len = int(secondary_train.shape[1]) if secondary_train is not None else None

        def _build_model() -> TransitCNN:
            return TransitCNN(
                input_length=int(global_train.shape[1]),
                model_mode="multiview",
                global_input_length=int(global_train.shape[1]),
                local_input_length=int(local_train.shape[1]),
                odd_even_input_length=odd_even_len,
                secondary_input_length=secondary_len,
                aux_feature_dim=aux_dim,
                use_odd_even=use_odd_even_effective,
                use_secondary=use_secondary_effective,
                use_aux_features=use_aux_features_effective,
                fusion_hidden_dim=fusion_hidden_dim,
            )

    else:
        train_dataset = SequenceDataset(x_train, y_train)
        val_dataset = SequenceDataset(x_val, y_val)
        test_dataset = SequenceDataset(x_test, y_test)

        def _build_model() -> TransitCNN:
            return TransitCNN(input_length=x_train.shape[1], model_mode="legacy")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "cnn_best.pt"

    if tune:
        rng = np.random.default_rng(random_seed)
        candidates = _sample_hparam_candidates(tune_search_space, tune_trials, rng)
        best_params: dict[str, float | int] | None = None
        best_trial_loss = float("inf")

        for idx, params in enumerate(candidates):
            trial_model, _, trial_val_loss, _, _ = _train_with_validation(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                y_train=y_train,
                build_model_fn=_build_model,
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

    model, history, best_val_loss, device, training_info = _train_with_validation(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        y_train=y_train,
        build_model_fn=_build_model,
        random_seed=random_seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        checkpoint_path=best_ckpt,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true_val, y_prob_val = _predict_probs(model, val_loader, device)
    threshold_result = find_best_threshold(
        y_true=y_true_val.astype(int),
        y_prob=y_prob_val,
        metric=threshold_metric,
    )
    threshold_used = float(threshold_result["best_threshold"])

    y_true_test, y_prob_test = _predict_probs(model, test_loader, device)

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
    metrics["model_mode"] = effective_mode
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
    metrics["multiview_arrays_available"] = bool(has_multiview_arrays)
    metrics["branches"] = {
        "odd_even_enabled": bool(use_odd_even_effective) if effective_mode == "multiview" else False,
        "secondary_enabled": bool(use_secondary_effective) if effective_mode == "multiview" else False,
        "aux_enabled": bool(use_aux_features_effective) if effective_mode == "multiview" else False,
        "fusion_hidden_dim": int(fusion_hidden_dim),
    }
    if aux_mean is not None and aux_std is not None:
        metrics["aux_norm"] = {
            "mean": aux_mean.tolist(),
            "std": aux_std.tolist(),
        }

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
                "model_mode": model_mode,
                "effective_model_mode": effective_mode,
                "use_odd_even_branch": use_odd_even_branch,
                "use_secondary_branch": use_secondary_branch,
                "disable_aux_features": disable_aux_features,
                "fusion_hidden_dim": fusion_hidden_dim,
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
    """Build CLI parser for CNN training."""
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
    parser.add_argument(
        "--model-mode",
        type=str,
        default="auto",
        choices=["auto", "legacy", "multiview"],
        help="Model input mode: auto-detect, force legacy, or require multiview arrays.",
    )
    parser.add_argument(
        "--use-odd-even-branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable odd/even diagnostic branch in multiview mode when arrays are available.",
    )
    parser.add_argument(
        "--use-secondary-branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable secondary-view branch in multiview mode when arrays are available.",
    )
    parser.add_argument(
        "--disable-aux-features",
        action="store_true",
        help="Disable aux-feature branch even when aux_features are present.",
    )
    parser.add_argument(
        "--fusion-hidden-dim",
        type=int,
        default=96,
        help="Hidden dimension used by multiview fusion classifier.",
    )
    return parser


def main() -> None:
    """Run CNN training from the command line."""
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
        model_mode=args.model_mode,
        use_odd_even_branch=bool(args.use_odd_even_branch),
        use_secondary_branch=bool(args.use_secondary_branch),
        disable_aux_features=bool(args.disable_aux_features),
        fusion_hidden_dim=int(args.fusion_hidden_dim),
    )
    print("CNN training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
