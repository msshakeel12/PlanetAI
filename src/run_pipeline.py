"""Orchestrate the full exoplanet transit classification pipeline.

This script runs all major stages in sequence: metadata download, dataset
construction, baseline training, CNN training, and reporting/tracking outputs.

Main public function:
- ``main`` (CLI entry point).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.dataset_builder import run_dataset_build
from src.data.download_metadata import run_download
from src.models.train_baselines import run_baseline_training
from src.models.train_cnn import run_cnn_training
from src.utils.config import load_yaml_config
from src.utils.paths import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_project_directories,
)
from src.utils.experiment_tracking import ExperimentTracker


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for full-pipeline orchestration.

    Returns:
        Configured parser with ``--config`` option.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config file.",
    )
    return parser


def main() -> None:
    """Run metadata download, dataset build, baseline training, and CNN training.

    Returns:
        None.
    """
    ensure_project_directories()
    args = build_parser().parse_args()

    # ------------------------------- Load config ------------------------------
    cfg = load_yaml_config(args.config)

    download_cfg = cfg["download"]
    dataset_cfg = cfg["dataset"]
    baseline_cfg = cfg["baseline"]
    cnn_cfg = cfg["cnn"]
    tracking_cfg = cfg.get("tracking", {})

    metadata_path = Path(download_cfg.get("output", RAW_DATA_DIR / "kepler_metadata.csv"))
    features_path = Path(dataset_cfg.get("features_output", PROCESSED_DATA_DIR / "features.csv"))
    sequences_path = Path(dataset_cfg.get("sequences_output", PROCESSED_DATA_DIR / "sequences.npz"))
    model_dir = Path(baseline_cfg.get("output_dir", PROCESSED_DATA_DIR / "models"))
    figures_dir = Path(cfg.get("figures_dir", FIGURES_DIR))
    tracker_dir_value = tracking_cfg.get("experiment_dir")
    tracker = ExperimentTracker(Path(tracker_dir_value)) if tracker_dir_value else None

    # ---------------------------- Stage 1: ingest -----------------------------
    run_download(
        output_path=metadata_path,
        table=download_cfg.get("table", "cumulative"),
        where_clause=download_cfg.get("where"),
        timeout_seconds=int(download_cfg.get("timeout", 60)),
        limit=download_cfg.get("limit"),
    )

    # ------------------------- Stage 2: build dataset -------------------------
    run_dataset_build(
        metadata_path=metadata_path,
        features_path=features_path,
        sequences_path=sequences_path,
        mode=dataset_cfg.get("mode", "synthetic"),
        label_column=dataset_cfg.get("label_column", "koi_disposition"),
        positive_labels=dataset_cfg.get("positive_labels", ["CONFIRMED", "CANDIDATE"]),
        sequence_length=int(dataset_cfg.get("sequence_length", 512)),
        random_seed=int(dataset_cfg.get("random_seed", 42)),
        smooth_window=dataset_cfg.get("smooth_window"),
        apply_phase_fold=bool(dataset_cfg.get("phase_fold", False)),
        period_column=dataset_cfg.get("period_column", "koi_period"),
        epoch_column=dataset_cfg.get("epoch_column", "koi_time0bk"),
        mast_cache_dir=Path(dataset_cfg.get("mast_cache_dir", RAW_DATA_DIR / "mast_cache")),
        max_real_targets=dataset_cfg.get("max_real_targets"),
        task_mode=dataset_cfg.get("task_mode", "disposition_binary"),
        earth_size_max_radius=float(dataset_cfg.get("earth_size_max_radius", 1.5)),
        include_candidates_as_positive=bool(dataset_cfg.get("include_candidates_as_positive", False)),
        mast_search_timeout_seconds=dataset_cfg.get("mast_search_timeout_seconds"),
        mast_download_timeout_seconds=dataset_cfg.get("mast_download_timeout_seconds"),
    )

    # ----------------------- Stage 3: baseline models -------------------------
    run_baseline_training(
        features_path=features_path,
        output_dir=model_dir,
        figures_dir=figures_dir,
        test_size=float(baseline_cfg.get("test_size", 0.2)),
        random_seed=int(baseline_cfg.get("random_seed", 42)),
        label_column=baseline_cfg.get("label_column", "label"),
        tune=bool(baseline_cfg.get("tune", False)),
        tune_iterations=int(baseline_cfg.get("tune_iterations", 20)),
        tune_cv_folds=int(baseline_cfg.get("tune_cv_folds", 5)),
        tracker=tracker,
    )

    # ------------------------- Stage 4: CNN training --------------------------
    run_cnn_training(
        sequences_path=sequences_path,
        output_dir=model_dir,
        figures_dir=figures_dir,
        random_seed=int(cnn_cfg.get("random_seed", 42)),
        batch_size=int(cnn_cfg.get("batch_size", 64)),
        learning_rate=float(cnn_cfg.get("learning_rate", 1e-3)),
        epochs=int(cnn_cfg.get("epochs", 20)),
        tune=bool(cnn_cfg.get("tune", False)),
        tune_trials=int(cnn_cfg.get("tune_trials", 8)),
        tune_search_space={
            "batch_size": list(cnn_cfg.get("tune_batch_sizes", [32, 64, 128])),
            "learning_rate": list(cnn_cfg.get("tune_learning_rates", [0.0003, 0.001, 0.003])),
            "epochs": list(cnn_cfg.get("tune_epochs", [10, 15, 20])),
        },
        task_mode=dataset_cfg.get("task_mode", "disposition_binary"),
        earth_size_max_radius=float(dataset_cfg.get("earth_size_max_radius", 1.5)),
        threshold_metric=cnn_cfg.get("threshold_metric", "balanced_accuracy"),
        tracker=tracker,
    )

    # --------------------------- Stage 5: summary -----------------------------
    print("Full pipeline run complete.")
    print(f"Metadata: {metadata_path}")
    print(f"Features: {features_path}")
    print(f"Sequences: {sequences_path}")
    print(f"Models: {model_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
