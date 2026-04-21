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
from src.features.preprocessing import (
    build_aux_features,
    build_global_view,
    build_local_view,
    build_odd_even_view,
    build_secondary_view,
    phase_fold_light_curve,
    preprocess_light_curve,
    resample_phase_curve,
)
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


def derive_earth_sized_labels(
    metadata: pd.DataFrame,
    label_column: str,
    earth_size_max_radius: float,
    include_candidates_as_positive: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Derive labels for Earth-sized planet detection.

    Args:
        metadata: Metadata dataframe from Exoplanet Archive.
        label_column: Disposition label column.
        earth_size_max_radius: Maximum ``koi_prad`` for Earth-sized positives.
        include_candidates_as_positive: Whether Earth-sized candidates are
            treated as positives.

    Returns:
        Tuple ``(labels, filtered_metadata)`` where filtered metadata keeps only
        rows participating in the Earth-sized binary task.

    Raises:
        ValueError: If required columns are missing.
    """
    if label_column not in metadata.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if "koi_prad" not in metadata.columns:
        raise ValueError("Earth-sized task requires 'koi_prad' column")

    dispositions = metadata[label_column].fillna("UNKNOWN").astype(str).str.upper()
    planet_radius = pd.to_numeric(metadata["koi_prad"], errors="coerce")

    is_confirmed = dispositions == "CONFIRMED"
    is_candidate = dispositions == "CANDIDATE"
    is_false_positive = dispositions == "FALSE POSITIVE"
    is_earth_sized = planet_radius.notna() & (planet_radius <= float(earth_size_max_radius))

    positive_mask = is_confirmed & is_earth_sized
    if include_candidates_as_positive:
        positive_mask = positive_mask | (is_candidate & is_earth_sized)

    negative_mask = is_false_positive | ((is_confirmed | is_candidate) & (~is_earth_sized))
    keep_mask = positive_mask | negative_mask

    filtered_metadata = metadata.loc[keep_mask].reset_index(drop=True)
    labels = positive_mask.loc[keep_mask].astype(int).to_numpy()
    return labels, filtered_metadata


def derive_task_labels(
    metadata: pd.DataFrame,
    task_mode: str,
    label_column: str,
    positive_labels: Iterable[str],
    earth_size_max_radius: float,
    include_candidates_as_positive: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Derive labels for configured task mode.

    Args:
        metadata: Input metadata dataframe.
        task_mode: Labeling strategy name.
        label_column: Disposition/label column.
        positive_labels: Positive labels for disposition mode.
        earth_size_max_radius: Radius threshold for Earth-sized mode.
        include_candidates_as_positive: Include Earth-sized candidates as
            positives in Earth-sized mode.

    Returns:
        Tuple ``(labels, filtered_metadata)``.
    """
    mode = task_mode.strip().lower()
    if mode == "disposition_binary":
        labels = derive_binary_labels(
            metadata=metadata,
            label_column=label_column,
            positive_labels=positive_labels,
        )
        return labels, metadata.reset_index(drop=True)
    if mode == "earth_sized_binary":
        return derive_earth_sized_labels(
            metadata=metadata,
            label_column=label_column,
            earth_size_max_radius=earth_size_max_radius,
            include_candidates_as_positive=include_candidates_as_positive,
        )
    raise ValueError("task_mode must be one of: disposition_binary, earth_sized_binary")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        Tuple ``(times, fluxes, labels, groups)`` as batched numpy arrays.
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

    groups = (
        metadata["kepid"].to_numpy() if "kepid" in metadata.columns else np.arange(len(labels), dtype=np.int64)
    )
    return np.asarray(time_batch), np.asarray(flux_batch), labels.astype(int), np.asarray(groups)


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
    mast_search_timeout_seconds: float | None = None,
    mast_download_timeout_seconds: float | None = None,
    return_kept_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
        mast_search_timeout_seconds: Optional timeout for MAST search stage.
        mast_download_timeout_seconds: Optional timeout for MAST download stage.

    Returns:
        Tuple ``(times, fluxes, labels, groups)`` for successfully ingested targets.

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
    groups_out: list[int] = []
    kept_indices: list[int] = []
    skipped_missing_kepid = 0
    skipped_errors = 0

    # Cap processing to support bounded-cost smoke runs against remote services.
    cap = len(metadata) if max_targets is None else min(len(metadata), max_targets)

    for idx in range(cap):
        progress = f"[{idx + 1}/{cap}]"
        row = metadata.iloc[idx]
        kepid_raw = row.get("kepid")
        if pd.isna(kepid_raw):
            skipped_missing_kepid += 1
            print(f"{progress} SKIP missing KEPID")
            continue

        try:
            kepid = int(kepid_raw)
            print(f"{progress} Fetching KIC {kepid}...")
            lc = download_kepler_light_curve(
                kepid=kepid,
                download_dir=mast_cache_dir,
                search_timeout_seconds=mast_search_timeout_seconds,
                download_timeout_seconds=mast_download_timeout_seconds,
            )

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
            groups_out.append(kepid)
            kept_indices.append(idx)
            print(f"{progress} OK KIC {kepid} ({proc_flux.size} points)")
        except Exception as exc:
            skipped_errors += 1
            print(f"{progress} SKIP KIC {kepid_raw}: {exc}")

    print(
        "Real ingestion summary: "
        f"processed={cap}, succeeded={len(labels_out)}, "
        f"skipped_missing_kepid={skipped_missing_kepid}, skipped_errors={skipped_errors}"
    )

    if not labels_out:
        raise RuntimeError(
            "No real Kepler light curves were ingested successfully. "
            "Verify network access, metadata KEPID values, and optional lightkurve dependency."
        )

    base_result = (
        np.asarray(times_out),
        np.asarray(fluxes_out),
        np.asarray(labels_out, dtype=int),
        np.asarray(groups_out, dtype=np.int64),
    )
    if return_kept_indices:
        return (*base_result, np.asarray(kept_indices, dtype=np.int64))
    return base_result


def _safe_numeric_value(row: pd.Series, key: str, fallback: float = 0.0) -> float:
    """Read numeric metadata value with deterministic fallback."""
    value = pd.to_numeric(row.get(key), errors="coerce")
    if pd.isna(value):
        return float(fallback)
    value_f = float(value)
    if not np.isfinite(value_f):
        return float(fallback)
    return value_f


def _build_multiview_arrays(
    times: np.ndarray,
    fluxes: np.ndarray,
    metadata: pd.DataFrame,
    period_column: str,
    epoch_column: str,
    global_view_length: int,
    local_view_length: int,
    odd_even_view_length: int,
    secondary_view_length: int,
    enable_odd_even_view: bool,
    enable_secondary_view: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build global/local/aux arrays from preprocessed curves and metadata."""
    global_views: list[np.ndarray] = []
    local_views: list[np.ndarray] = []
    aux_features: list[np.ndarray] = []
    odd_even_views: list[np.ndarray] = []
    secondary_views: list[np.ndarray] = []

    n_samples = int(times.shape[0])
    for idx in range(n_samples):
        row = metadata.iloc[idx]
        period = _safe_numeric_value(row, period_column, fallback=0.0)
        epoch = _safe_numeric_value(row, epoch_column, fallback=0.0)
        duration_hours = _safe_numeric_value(row, "koi_duration", fallback=0.0)
        global_view = build_global_view(
            time=times[idx],
            flux=fluxes[idx],
            period=period,
            epoch=epoch,
            target_length=global_view_length,
        )
        local_view = build_local_view(
            time=times[idx],
            flux=fluxes[idx],
            period=period,
            epoch=epoch,
            target_length=local_view_length,
            duration_hours=duration_hours,
        )
        aux = build_aux_features(row)

        if enable_odd_even_view:
            odd_even = build_odd_even_view(
                time=times[idx],
                flux=fluxes[idx],
                period=period,
                epoch=epoch,
                target_length=odd_even_view_length,
                duration_hours=duration_hours,
            )
            odd_even_views.append(np.nan_to_num(odd_even, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        if enable_secondary_view:
            secondary = build_secondary_view(
                time=times[idx],
                flux=fluxes[idx],
                period=period,
                epoch=epoch,
                target_length=secondary_view_length,
                duration_hours=duration_hours,
            )
            secondary_views.append(np.nan_to_num(secondary, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

        global_views.append(np.nan_to_num(global_view, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        local_views.append(np.nan_to_num(local_view, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        aux_features.append(np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))

    odd_even_out = np.asarray(odd_even_views, dtype=np.float32) if enable_odd_even_view else None
    secondary_out = np.asarray(secondary_views, dtype=np.float32) if enable_secondary_view else None

    return (
        np.asarray(global_views, dtype=np.float32),
        np.asarray(local_views, dtype=np.float32),
        np.asarray(aux_features, dtype=np.float32),
        odd_even_out,
        secondary_out,
    )


def build_real_stub_dataset(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    sequence_length: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate deterministic placeholder arrays for non-MAST workflows.

    Args:
        metadata: Metadata dataframe (used for sample count only).
        labels: Binary labels.
        sequence_length: Output sequence length.
        random_seed: Seed controlling placeholder values.

    Returns:
        Tuple ``(times, fluxes, labels, groups)`` with synthetic placeholder fluxes.

    Notes:
        TODO: Keep this mode as an explicit bridge until all environments have
        dependable access to real Kepler downloads.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = len(metadata)
    times = np.tile(np.linspace(0.0, 30.0, sequence_length), (n_samples, 1))
    fluxes = rng.normal(0.0, 1.0, size=(n_samples, sequence_length))
    groups = (
        metadata["kepid"].to_numpy() if "kepid" in metadata.columns else np.arange(n_samples, dtype=np.int64)
    )
    return times, fluxes, labels.astype(int), np.asarray(groups)


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
    task_mode: str = "disposition_binary",
    earth_size_max_radius: float = 1.5,
    include_candidates_as_positive: bool = False,
    mast_search_timeout_seconds: float | None = None,
    mast_download_timeout_seconds: float | None = None,
    global_view_length: int = 1024,
    local_view_length: int = 256,
    odd_even_view_length: int = 256,
    secondary_view_length: int = 256,
    enable_odd_even_view: bool = True,
    enable_secondary_view: bool = True,
) -> tuple[Path, Path]:
    """Build feature and sequence artifacts from metadata.

    Args:
        metadata_path: Input metadata CSV path.
        features_path: Output feature CSV path.
        sequences_path: Output sequence NPZ path.
        mode: One of ``synthetic``, ``real``, or ``real_stub``.
        label_column: Metadata column used for label derivation.
        positive_labels: Positive label values.
        task_mode: Labeling task mode.
        earth_size_max_radius: Radius threshold for Earth-sized task mode.
        include_candidates_as_positive: Include candidates in positive class for
            Earth-sized task mode.
        sequence_length: Target sequence length.
        random_seed: RNG seed.
        smooth_window: Optional smoothing window.
        apply_phase_fold: Whether to apply phase-folding transform.
        period_column: Metadata period column for phase fold.
        epoch_column: Metadata epoch column for phase fold.
        mast_cache_dir: MAST cache directory for real mode.
        max_real_targets: Optional ingestion cap in real mode.
        mast_search_timeout_seconds: Optional timeout for MAST search stage.
        mast_download_timeout_seconds: Optional timeout for MAST download stage.
        global_view_length: Fixed length for global folded view.
        local_view_length: Fixed length for local transit-centered view.
        odd_even_view_length: Fixed length for odd/even diagnostic view bins.
        secondary_view_length: Fixed length for secondary-eclipse view.
        enable_odd_even_view: Whether to save pass-2 odd/even diagnostic views.
        enable_secondary_view: Whether to save pass-2 secondary diagnostic views.

    Returns:
        Tuple containing paths to the saved feature CSV and sequence NPZ.

    Raises:
        ValueError: If an unsupported mode is provided.
        RuntimeError: If real mode ingests zero valid targets.
    """
    # ------------------------------ Load metadata -----------------------------
    metadata = pd.read_csv(metadata_path)
    metadata.columns = [c.lower() for c in metadata.columns]
    labels, metadata = derive_task_labels(
        metadata=metadata,
        task_mode=task_mode,
        label_column=label_column.lower(),
        positive_labels=positive_labels,
        earth_size_max_radius=earth_size_max_radius,
        include_candidates_as_positive=include_candidates_as_positive,
    )

    # ------------------------------- Build data -------------------------------
    if mode == "synthetic":
        times, fluxes, labels_out, groups_out = build_synthetic_dataset(
            metadata=metadata,
            labels=labels,
            sequence_length=sequence_length,
            random_seed=random_seed,
            smooth_window=smooth_window,
            apply_phase_fold=apply_phase_fold,
            period_column=period_column,
            epoch_column=epoch_column,
        )
        metadata_out = metadata.reset_index(drop=True)
    elif mode == "real":
        times, fluxes, labels_out, groups_out, kept_indices = build_real_dataset(
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
            mast_search_timeout_seconds=mast_search_timeout_seconds,
            mast_download_timeout_seconds=mast_download_timeout_seconds,
            return_kept_indices=True,
        )
        metadata_out = metadata.iloc[kept_indices].reset_index(drop=True)
    elif mode == "real_stub":
        times, fluxes, labels_out, groups_out = build_real_stub_dataset(
            metadata=metadata,
            labels=labels,
            sequence_length=sequence_length,
            random_seed=random_seed,
        )
        metadata_out = metadata.reset_index(drop=True)
    else:
        raise ValueError("mode must be one of: synthetic, real, real_stub")

    global_views, local_views, aux_features, odd_even_views, secondary_views = _build_multiview_arrays(
        times=times,
        fluxes=fluxes,
        metadata=metadata_out,
        period_column=period_column,
        epoch_column=epoch_column,
        global_view_length=global_view_length,
        local_view_length=local_view_length,
        odd_even_view_length=odd_even_view_length,
        secondary_view_length=secondary_view_length,
        enable_odd_even_view=enable_odd_even_view,
        enable_secondary_view=enable_secondary_view,
    )

    # -------------------------- Feature engineering ---------------------------
    feature_df = build_feature_table(times=times, fluxes=fluxes)
    feature_df["label"] = labels_out

    # ----------------------------- Save artifacts -----------------------------
    features_path.parent.mkdir(parents=True, exist_ok=True)
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(features_path, index=False)
    save_payload: dict[str, np.ndarray] = {
        "times": np.nan_to_num(times, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "sequences": np.nan_to_num(fluxes, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "labels": labels_out,
        "groups": groups_out,
        "global_views": global_views,
        "local_views": local_views,
        "aux_features": aux_features,
    }
    if odd_even_views is not None:
        save_payload["odd_even_views"] = odd_even_views
    if secondary_views is not None:
        save_payload["secondary_views"] = secondary_views
    np.savez_compressed(sequences_path, **save_payload)

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
        "--task-mode",
        type=str,
        default="disposition_binary",
        choices=["disposition_binary", "earth_sized_binary"],
        help="Labeling task mode.",
    )
    parser.add_argument(
        "--earth-size-max-radius",
        type=float,
        default=1.5,
        help="Maximum radius (Earth radii) for Earth-sized positives.",
    )
    parser.add_argument(
        "--include-candidates-as-positive",
        action="store_true",
        help="Treat Earth-sized CANDIDATE rows as positive in Earth-sized mode.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Fixed sequence length after preprocessing.",
    )
    parser.add_argument(
        "--global-view-length",
        type=int,
        default=1024,
        help="Fixed length for the global phase-folded view.",
    )
    parser.add_argument(
        "--local-view-length",
        type=int,
        default=256,
        help="Fixed length for the local transit-centered view.",
    )
    parser.add_argument(
        "--secondary-view-length",
        type=int,
        default=128,
        help="Fixed length for the secondary-eclipse view.",
    )
    parser.add_argument(
        "--odd-even-view-length",
        type=int,
        default=256,
        help="Fixed length for odd/even diagnostic views.",
    )
    parser.add_argument(
        "--enable-odd-even-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save pass-2 odd/even diagnostic views in the sequence NPZ.",
    )
    parser.add_argument(
        "--enable-secondary-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save pass-2 secondary diagnostic views in the sequence NPZ.",
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
        "--mast-search-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout (seconds) for MAST search requests per KIC in real mode.",
    )
    parser.add_argument(
        "--mast-download-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout (seconds) for MAST download/stitch per KIC in real mode.",
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
        task_mode=args.task_mode,
        earth_size_max_radius=args.earth_size_max_radius,
        include_candidates_as_positive=args.include_candidates_as_positive,
        sequence_length=args.sequence_length,
        random_seed=args.random_seed,
        smooth_window=args.smooth_window,
        apply_phase_fold=args.phase_fold,
        period_column=args.period_column,
        epoch_column=args.epoch_column,
        mast_cache_dir=args.mast_cache_dir,
        max_real_targets=args.max_real_targets,
        mast_search_timeout_seconds=args.mast_search_timeout_seconds,
        mast_download_timeout_seconds=args.mast_download_timeout_seconds,
        global_view_length=args.global_view_length,
        local_view_length=args.local_view_length,
        odd_even_view_length=args.odd_even_view_length,
        secondary_view_length=args.secondary_view_length,
        enable_odd_even_view=args.enable_odd_even_view,
        enable_secondary_view=args.enable_secondary_view,
    )
    print(f"Saved feature table to {features_path}")
    print(f"Saved sequence arrays to {sequences_path}")


if __name__ == "__main__":
    main()
