"""Build supervised learning artifacts from exoplanet metadata.

This module bridges metadata labels to model-ready inputs. It supports three
dataset modes used at different maturity stages of the project:

- ``synthetic``: fully local synthetic light curves for deterministic testing.
- ``real``: real Kepler light-curve downloads via MAST/Lightkurve.
- ``real_stub``: explicit placeholder behavior when real ingestion is not used.

Main public functions:
- ``run_dataset_build``: end-to-end builder used by CLI and orchestrator.
- ``derive_binary_labels``: disposition-to-binary target mapping.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.mast_kepler import download_kepler_light_curve
from src.features.feature_engineering import build_feature_table
from src.features.preprocessing import phase_fold_light_curve, preprocess_light_curve, resample_phase_curve
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_directories


def derive_binary_labels(
    metadata: pd.DataFrame,
    label_column: str,
    positive_labels: Iterable[str],
) -> np.ndarray:
    """Map disposition labels to binary targets.

    Args:
        metadata: Metadata dataframe from Exoplanet Archive.
        label_column: Column containing disposition/class labels.
        positive_labels: Label values treated as positive transit evidence.

    Returns:
        Integer numpy array of 0/1 labels.

    Raises:
        ValueError: If ``label_column`` is not present in ``metadata``.
    """
    positives = {v.upper() for v in positive_labels}
    if label_column not in metadata.columns:
        raise ValueError(f"Missing label column: {label_column}")

    labels = (
        metadata[label_column]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .isin(positives)
        .astype(int)
        .to_numpy()
    )
    return labels


def _simulate_transit_curve(label: int, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy transit-like curve for pipeline bootstrapping.

    Args:
        label: Binary class; ``1`` includes an injected dip.
        length: Number of time samples.
        rng: Random generator for reproducibility.

    Returns:
        Tuple of ``(time, flux)`` arrays.

    Notes:
        This is intentionally a simplified phenomenological simulator, not a
        physically faithful transit model.
    """
    time = np.linspace(0.0, 30.0, length)
    baseline = 0.01 * np.sin(2.0 * np.pi * time / 10.0)
    noise = rng.normal(0.0, 0.005, size=length)
    flux = 1.0 + baseline + noise

    if label == 1:
        # NOTE: Dip shape is Gaussian for convenience; real transits are better
        # modeled by limb-darkened transit equations.
        center = rng.uniform(8.0, 22.0)
        width = rng.uniform(0.2, 0.8)
        depth = rng.uniform(0.01, 0.03)
        dip = depth * np.exp(-0.5 * ((time - center) / width) ** 2)
        flux = flux - dip

    # Introduce occasional missing values to exercise preprocessing paths.
    missing_idx = rng.choice(length, size=max(1, length // 20), replace=False)
    flux[missing_idx] = np.nan
    return time, flux


def build_synthetic_dataset(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    sequence_length: int,
    random_seed: int,
    smooth_window: int | None,
    apply_phase_fold: bool,
    period_column: str,
    epoch_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a fully synthetic dataset from metadata labels.

    Args:
        metadata: Metadata rows used for label count and optional fold columns.
        labels: Binary labels per row.
        sequence_length: Target sequence length after preprocessing.
        random_seed: RNG seed for reproducibility.
        smooth_window: Optional smoothing window.
        apply_phase_fold: Whether to phase-fold after preprocessing.
        period_column: Metadata column containing period estimates.
        epoch_column: Metadata column containing epoch estimates.

    Returns:
        Tuple ``(times, fluxes, labels)`` as batched numpy arrays.
    """
    rng = np.random.default_rng(random_seed)

    time_batch: list[np.ndarray] = []
    flux_batch: list[np.ndarray] = []

    for idx, label in enumerate(labels):
        raw_time, raw_flux = _simulate_transit_curve(int(label), sequence_length, rng)

        # NOTE: Synthetic mode reuses metadata period/epoch only to test phase
        # folding code paths; values are not tied to a physical simulator.
        period = float(metadata.iloc[idx].get(period_column, 10.0) or 10.0)
        epoch = float(metadata.iloc[idx].get(epoch_column, 0.0) or 0.0)

        proc_time, proc_flux = preprocess_light_curve(
            time=raw_time,
            flux=raw_flux,
            target_length=sequence_length,
            normalize_method="zscore",
            detrend_order=2,
            smooth_window=smooth_window,
        )
        if apply_phase_fold and period > 0:
            phase, phase_flux = phase_fold_light_curve(proc_time, proc_flux, period=period, epoch=epoch)
            proc_time, proc_flux = resample_phase_curve(phase, phase_flux, target_length=sequence_length)

        time_batch.append(proc_time)
        flux_batch.append(proc_flux)

    return np.asarray(time_batch), np.asarray(flux_batch), labels.astype(int)


def build_real_dataset(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    sequence_length: int,
    random_seed: int,
    smooth_window: int | None,
    apply_phase_fold: bool,
    period_column: str,
    epoch_column: str,
    mast_cache_dir: Path,
    max_targets: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dataset from real Kepler light curves fetched from MAST.

    Args:
        metadata: Metadata dataframe including ``kepid`` and label columns.
        labels: Binary labels aligned with metadata rows.
        sequence_length: Target sequence length after preprocessing.
        random_seed: Seed placeholder for deterministic extension points.
        smooth_window: Optional smoothing window.
        apply_phase_fold: Whether to phase-fold using period/epoch metadata.
        period_column: Metadata period column.
        epoch_column: Metadata transit epoch column.
        mast_cache_dir: Local cache for downloaded MAST files.
        max_targets: Optional cap on number of rows processed.

    Returns:
        Tuple ``(times, fluxes, labels)`` for successfully ingested targets.

    Raises:
        RuntimeError: If no targets are ingested successfully.

    Notes:
        Per-target failures are tolerated and logged; the builder continues so
        that partial progress can still be used for experimentation.
    """
    np.random.default_rng(random_seed)  # Keeps function signature deterministic even if expanded.

    times_out: list[np.ndarray] = []
    fluxes_out: list[np.ndarray] = []
    labels_out: list[int] = []

    # Cap processing to support bounded-cost smoke runs against remote services.
    cap = len(metadata) if max_targets is None else min(len(metadata), max_targets)

    for idx in range(cap):
        row = metadata.iloc[idx]
        kepid_raw = row.get("kepid")
        if pd.isna(kepid_raw):
            continue

        try:
            kepid = int(kepid_raw)
            lc = download_kepler_light_curve(kepid=kepid, download_dir=mast_cache_dir)

            proc_time, proc_flux = preprocess_light_curve(
                time=lc.time,
                flux=lc.flux,
                target_length=sequence_length,
                normalize_method="zscore",
                detrend_order=2,
                smooth_window=smooth_window,
            )

            if apply_phase_fold:
                # Domain assumption: metadata period/epoch are approximate and
                # may include uncertainty; fold is a pragmatic signal-alignment
                # step rather than a full transit-timing treatment.
                period = float(row.get(period_column, 0.0) or 0.0)
                epoch = float(row.get(epoch_column, 0.0) or 0.0)
                if period > 0:
                    phase, phase_flux = phase_fold_light_curve(proc_time, proc_flux, period=period, epoch=epoch)
                    proc_time, proc_flux = resample_phase_curve(phase, phase_flux, target_length=sequence_length)

            times_out.append(proc_time)
            fluxes_out.append(proc_flux)
            labels_out.append(int(labels[idx]))
        except Exception as exc:
            print(f"Skipping KIC {kepid_raw} due to ingestion error: {exc}")

    if not labels_out:
        raise RuntimeError(
            "No real Kepler light curves were ingested successfully. "
            "Verify network access, metadata KEPID values, and optional lightkurve dependency."
        )

    return np.asarray(times_out), np.asarray(fluxes_out), np.asarray(labels_out, dtype=int)


def build_real_stub_dataset(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    sequence_length: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic placeholder arrays for non-MAST workflows.

    Args:
        metadata: Metadata dataframe (used for sample count only).
        labels: Binary labels.
        sequence_length: Output sequence length.
        random_seed: Seed controlling placeholder values.

    Returns:
        Tuple ``(times, fluxes, labels)`` with synthetic placeholder fluxes.

    Notes:
        TODO: Keep this mode as an explicit bridge until all environments have
        dependable access to real Kepler downloads.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = len(metadata)
    times = np.tile(np.linspace(0.0, 30.0, sequence_length), (n_samples, 1))
    fluxes = rng.normal(0.0, 1.0, size=(n_samples, sequence_length))
    return times, fluxes, labels.astype(int)


def run_dataset_build(
    metadata_path: Path,
    features_path: Path,
    sequences_path: Path,
    mode: str,
    label_column: str,
    positive_labels: Iterable[str],
    sequence_length: int,
    random_seed: int,
    smooth_window: int | None,
    apply_phase_fold: bool,
    period_column: str,
    epoch_column: str,
    mast_cache_dir: Path,
    max_real_targets: int | None,
) -> tuple[Path, Path]:
    """Build feature and sequence artifacts from metadata.

    Args:
        metadata_path: Input metadata CSV path.
        features_path: Output feature CSV path.
        sequences_path: Output sequence NPZ path.
        mode: One of ``synthetic``, ``real``, or ``real_stub``.
        label_column: Metadata column used for label derivation.
        positive_labels: Positive label values.
        sequence_length: Target sequence length.
        random_seed: RNG seed.
        smooth_window: Optional smoothing window.
        apply_phase_fold: Whether to apply phase-folding transform.
        period_column: Metadata period column for phase fold.
        epoch_column: Metadata epoch column for phase fold.
        mast_cache_dir: MAST cache directory for real mode.
        max_real_targets: Optional ingestion cap in real mode.

    Returns:
        Tuple containing paths to the saved feature CSV and sequence NPZ.

    Raises:
        ValueError: If an unsupported mode is provided.
        RuntimeError: If real mode ingests zero valid targets.
    """
    # ------------------------------ Load metadata -----------------------------
    metadata = pd.read_csv(metadata_path)
    metadata.columns = [c.lower() for c in metadata.columns]

    labels = derive_binary_labels(metadata=metadata, label_column=label_column.lower(), positive_labels=positive_labels)

    # ------------------------------- Build data -------------------------------
    if mode == "synthetic":
        times, fluxes, labels_out = build_synthetic_dataset(
            metadata=metadata,
            labels=labels,
            sequence_length=sequence_length,
            random_seed=random_seed,
            smooth_window=smooth_window,
            apply_phase_fold=apply_phase_fold,
            period_column=period_column,
            epoch_column=epoch_column,
        )
    elif mode == "real":
        times, fluxes, labels_out = build_real_dataset(
            metadata=metadata,
            labels=labels,
            sequence_length=sequence_length,
            random_seed=random_seed,
            smooth_window=smooth_window,
            apply_phase_fold=apply_phase_fold,
            period_column=period_column,
            epoch_column=epoch_column,
            mast_cache_dir=mast_cache_dir,
            max_targets=max_real_targets,
        )
    elif mode == "real_stub":
        times, fluxes, labels_out = build_real_stub_dataset(
            metadata=metadata,
            labels=labels,
            sequence_length=sequence_length,
            random_seed=random_seed,
        )
    else:
        raise ValueError("mode must be one of: synthetic, real, real_stub")

    # -------------------------- Feature engineering ---------------------------
    feature_df = build_feature_table(times=times, fluxes=fluxes)
    feature_df["label"] = labels_out

    # ----------------------------- Save artifacts -----------------------------
    features_path.parent.mkdir(parents=True, exist_ok=True)
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(features_path, index=False)
    np.savez_compressed(sequences_path, times=times, sequences=fluxes, labels=labels_out)

    return features_path, sequences_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for dataset generation.

    Returns:
        Configured parser for dataset-building options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=RAW_DATA_DIR / "kepler_metadata.csv",
        help="Input metadata CSV.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "real", "real_stub"],
        default="synthetic",
        help="Dataset construction mode.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="koi_disposition",
        help="Metadata column used for binary label mapping.",
    )
    parser.add_argument(
        "--positive-labels",
        nargs="+",
        default=["CONFIRMED", "CANDIDATE"],
        help="Labels treated as positive class.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Fixed sequence length after preprocessing.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Optional moving-average smoothing window.",
    )
    parser.add_argument(
        "--phase-fold",
        action="store_true",
        help="Apply phase folding using metadata period/epoch after preprocessing.",
    )
    parser.add_argument(
        "--period-column",
        type=str,
        default="koi_period",
        help="Metadata column used for orbital period in phase folding.",
    )
    parser.add_argument(
        "--epoch-column",
        type=str,
        default="koi_time0bk",
        help="Metadata column used for transit epoch in phase folding.",
    )
    parser.add_argument(
        "--mast-cache-dir",
        type=Path,
        default=RAW_DATA_DIR / "mast_cache",
        help="Directory used to cache downloaded MAST light-curve products.",
    )
    parser.add_argument(
        "--max-real-targets",
        type=int,
        default=None,
        help="Optional cap on number of metadata rows ingested in real mode.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=PROCESSED_DATA_DIR / "features.csv",
        help="Output feature CSV path.",
    )
    parser.add_argument(
        "--sequences-output",
        type=Path,
        default=PROCESSED_DATA_DIR / "sequences.npz",
        help="Output sequence NPZ path.",
    )
    return parser


def main() -> None:
    """Run dataset builder from the command line.

    Returns:
        None.
    """
    ensure_project_directories()
    args = build_parser().parse_args()
    features_path, sequences_path = run_dataset_build(
        metadata_path=args.metadata,
        features_path=args.features_output,
        sequences_path=args.sequences_output,
        mode=args.mode,
        label_column=args.label_column,
        positive_labels=args.positive_labels,
        sequence_length=args.sequence_length,
        random_seed=args.random_seed,
        smooth_window=args.smooth_window,
        apply_phase_fold=args.phase_fold,
        period_column=args.period_column,
        epoch_column=args.epoch_column,
        mast_cache_dir=args.mast_cache_dir,
        max_real_targets=args.max_real_targets,
    )
    print(f"Saved feature table to {features_path}")
    print(f"Saved sequence arrays to {sequences_path}")


if __name__ == "__main__":
    main()
