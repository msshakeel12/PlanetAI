"""Train classical ML baseline models for transit classification.

This module trains and evaluates Logistic Regression and Random Forest models on
handcrafted features. It also supports optional hyperparameter tuning and
experiment tracking for reproducibility.

Main public functions:
- ``run_baseline_training``: programmatic training entry point.
- ``main``: command-line entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluation import compute_classification_metrics, positive_class_scores_from_model
from src.models.plotting import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.utils.io import save_json
from src.utils.paths import FIGURES_DIR, PROCESSED_DATA_DIR, ensure_project_directories
from src.utils.experiment_tracking import ExperimentTracker


def _safe_train_test_split(
    x: object,
    y: object,
    test_size: float,
    random_seed: int,
) -> tuple[object, object, object, object]:
    """Split data robustly for tiny datasets.

    Args:
        x: Feature matrix-like object.
        y: Label vector-like object.
        test_size: Desired test fraction.
        random_seed: RNG seed.

    Returns:
        Tuple of train/test splits: ``x_train, x_test, y_train, y_test``.

    Notes:
        If strict stratification is infeasible (for example very small class
        counts), this helper falls back to an unstratified split to keep smoke
        and low-sample runs executable.
    """
    y_arr = pd.Series(y).to_numpy()
    n_samples = len(y_arr)
    class_counts = pd.Series(y_arr).value_counts().to_dict()
    n_classes = len(class_counts)

    raw_test_n = max(1, int(round(test_size * n_samples)))
    min_required = max(1, n_classes)
    test_n = max(raw_test_n, min_required)
    test_n = min(test_n, n_samples - 1)
    eff_test_size = test_n / n_samples

    use_stratify = all(v >= 2 for v in class_counts.values()) and test_n >= n_classes
    stratify_arg = y_arr if use_stratify else None

    return train_test_split(
        x,
        y_arr,
        test_size=eff_test_size,
        random_state=random_seed,
        stratify=stratify_arg,
    )


def _tune_model(
    model_name: str,
    model: object,
    x_train: object,
    y_train: object,
    random_seed: int,
    n_iter: int,
    cv_folds: int,
) -> object:
    """Tune baseline hyperparameters via randomized cross-validation.

    Args:
        model_name: Name used to pick parameter search space.
        model: Estimator instance.
        x_train: Training features.
        y_train: Training labels.
        random_seed: RNG seed.
        n_iter: Number of random parameter samples.
        cv_folds: Number of CV folds.

    Returns:
        Best estimator found by search (or input model if unsupported name).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    if model_name == "logistic_regression":
        distributions = {
            "model__C": loguniform(1e-3, 1e2),
            "model__penalty": ["l2"],
        }
    elif model_name == "random_forest":
        distributions = {
            "n_estimators": randint(200, 700),
            "max_depth": [None, 4, 8, 12, 16],
            "min_samples_split": randint(2, 12),
            "min_samples_leaf": randint(1, 8),
            "max_features": uniform(0.3, 0.7),
        }
    else:
        return model

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        random_state=random_seed,
        refit=True,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def run_baseline_training(
    features_path: Path,
    output_dir: Path,
    figures_dir: Path,
    test_size: float,
    random_seed: int,
    label_column: str,
    tune: bool,
    tune_iterations: int,
    tune_cv_folds: int,
    tracker: ExperimentTracker | None = None,
) -> dict[str, dict[str, float]]:
    """Train baseline models, evaluate, and persist artifacts.

    Args:
        features_path: Input feature CSV path.
        output_dir: Directory for model/metric artifacts.
        figures_dir: Directory for evaluation figures.
        test_size: Test split fraction.
        random_seed: RNG seed.
        label_column: Name of binary target column.
        tune: Whether to enable hyperparameter search.
        tune_iterations: Number of random search iterations.
        tune_cv_folds: Number of CV folds for tuning.
        tracker: Optional experiment tracker instance.

    Returns:
        Mapping of model name to metric dictionary.

    Raises:
        ValueError: If ``label_column`` is missing.
    """
    # ------------------------------- Load data --------------------------------
    df = pd.read_csv(features_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' was not found in {features_path}")

    x = df.drop(columns=[label_column]).to_numpy()
    y = df[label_column].to_numpy().astype(int)

    # ------------------------------ Train split -------------------------------
    x_train, x_test, y_train, y_test = _safe_train_test_split(
        x=x,
        y=y,
        test_size=test_size,
        random_seed=random_seed,
    )

    # ---------------------------- Model registry ------------------------------
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_model: dict[str, dict[str, float]] = {}

    # --------------------- Train / evaluate each baseline ---------------------
    for model_name, model in models.items():
        if tune:
            model = _tune_model(
                model_name=model_name,
                model=model,
                x_train=x_train,
                y_train=y_train,
                random_seed=random_seed,
                n_iter=tune_iterations,
                cv_folds=tune_cv_folds,
            )

        model.fit(x_train, y_train)
        y_prob = positive_class_scores_from_model(model, x_test)
        metrics = compute_classification_metrics(y_true=y_test, y_prob=y_prob)
        metrics_by_model[model_name] = metrics

        # Save explicit per-model artifacts for easier reproducibility/audits.
        model_path = output_dir / f"{model_name}.joblib"
        metrics_path = output_dir / f"{model_name}_metrics.json"
        joblib.dump(model, model_path)
        save_json(metrics, metrics_path)

        plot_confusion_matrix(y_true=y_test, y_prob=y_prob, output_path=figures_dir / f"{model_name}_confusion_matrix.png")
        plot_roc_curve(y_true=y_test, y_prob=y_prob, output_path=figures_dir / f"{model_name}_roc_curve.png")
        plot_precision_recall_curve(
            y_true=y_test,
            y_prob=y_prob,
            output_path=figures_dir / f"{model_name}_precision_recall.png",
        )

        # Optional run logging keeps baseline experiments comparable over time.
        if tracker is not None:
            tracker.log_run(
                stage=f"baseline_{model_name}",
                params={
                    "test_size": test_size,
                    "random_seed": random_seed,
                    "tune": tune,
                    "tune_iterations": tune_iterations,
                    "tune_cv_folds": tune_cv_folds,
                },
                metrics=metrics,
                artifacts={
                    "model": str(model_path),
                    "metrics": str(metrics_path),
                },
            )

    # --------------------------- Aggregate reporting ---------------------------
    plot_model_comparison(metrics_by_model=metrics_by_model, output_path=figures_dir / "baseline_model_comparison.png")
    save_json(metrics_by_model, output_dir / "baseline_metrics_summary.json")
    return metrics_by_model


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for baseline training.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DATA_DIR / "features.csv",
        help="Input feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "models",
        help="Directory for model artifacts and metrics.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory for evaluation figures.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--tune", action="store_true", help="Enable randomized hyperparameter search.")
    parser.add_argument("--tune-iterations", type=int, default=20)
    parser.add_argument("--tune-cv-folds", type=int, default=5)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Optional experiment tracking directory for JSON logs.",
    )
    return parser


def main() -> None:
    """Run baseline training from the command line.

    Returns:
        None.
    """
    ensure_project_directories()
    args = build_parser().parse_args()
    tracker = ExperimentTracker(args.experiment_dir) if args.experiment_dir is not None else None
    metrics = run_baseline_training(
        features_path=args.features,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        test_size=args.test_size,
        random_seed=args.random_seed,
        label_column=args.label_column,
        tune=args.tune,
        tune_iterations=args.tune_iterations,
        tune_cv_folds=args.tune_cv_folds,
        tracker=tracker,
    )
    print("Baseline training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
