"""Reusable light-curve preprocessing utilities.

This module defines transform functions shared across dataset building and model
training. Functions are intentionally composable and unit-testable so that the
pipeline can evolve from synthetic to real Kepler data without rewriting core
signal-processing steps.

Main public functions:
- ``preprocess_light_curve``: standard preprocessing chain.
- ``phase_fold_light_curve`` / ``resample_phase_curve``: optional alignment
    around candidate transit period/epoch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def impute_missing_flux(time: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute missing flux values using linear interpolation.

    Args:
        time: Time array.
        flux: Flux array that may contain NaNs.

    Returns:
        Tuple ``(time, flux_imputed)`` with NaNs filled.

    Raises:
        ValueError: If ``time`` and ``flux`` shapes differ.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if time.shape != flux.shape:
        raise ValueError("time and flux must have the same shape")

    series = pd.Series(flux)
    flux_imputed = series.interpolate(method="linear", limit_direction="both").to_numpy()
    return time, flux_imputed


def detrend_flux(time: np.ndarray, flux: np.ndarray, poly_order: int = 2) -> np.ndarray:
    """Remove a low-order polynomial trend from flux.

    Args:
        time: Time array.
        flux: Flux array.
        poly_order: Polynomial order for trend fit.

    Returns:
        Detrended flux array.

    Raises:
        ValueError: If ``poly_order`` is negative.

    Notes:
        This is a pragmatic baseline detrending approach; it is not a
        mission-grade systematic correction pipeline.
    """
    if poly_order < 0:
        raise ValueError("poly_order must be non-negative")

    coeffs = np.polyfit(time, flux, deg=poly_order)
    trend = np.polyval(coeffs, time)
    return flux - trend


def normalize_flux(flux: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize flux using z-score or min-max scaling.

    Args:
        flux: Flux array.
        method: Either ``zscore`` or ``minmax``.

    Returns:
        Normalized flux array.

    Raises:
        ValueError: If ``method`` is unsupported.
    """
    flux = np.asarray(flux, dtype=float)

    if method == "zscore":
        std = float(np.std(flux))
        if std == 0:
            return flux - float(np.mean(flux))
        return (flux - float(np.mean(flux))) / std

    if method == "minmax":
        fmin = float(np.min(flux))
        fmax = float(np.max(flux))
        if fmax == fmin:
            return np.zeros_like(flux)
        return (flux - fmin) / (fmax - fmin)

    raise ValueError(f"Unsupported normalization method: {method}")


def resample_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a curve to fixed length via interpolation.

    Args:
        time: Input time array.
        flux: Input flux array.
        target_length: Number of output samples.

    Returns:
        Tuple ``(new_time, new_flux)``.

    Raises:
        ValueError: If ``target_length`` is not greater than 1.
    """
    if target_length <= 1:
        raise ValueError("target_length must be greater than 1")

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    new_time = np.linspace(time.min(), time.max(), target_length)
    new_flux = np.interp(new_time, time, flux)
    return new_time, new_flux


def smooth_flux(flux: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving-average smoothing.

    Args:
        flux: Flux array.
        window: Moving-average window size.

    Returns:
        Smoothed flux array.

    Raises:
        ValueError: If ``window`` is less than 1.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if window == 1:
        return np.asarray(flux, dtype=float)
    return uniform_filter1d(np.asarray(flux, dtype=float), size=window, mode="nearest")


def phase_fold_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold a light curve around an estimated period and epoch.

    Args:
        time: Time array.
        flux: Flux array.
        period: Candidate orbital period.
        epoch: Candidate transit epoch.

    Returns:
        Tuple ``(phase_sorted, flux_sorted)`` with phase in ``[-0.5, 0.5)``.

    Raises:
        ValueError: If ``period`` is not positive.

    Notes:
        Folding assumes constant period and ignores transit timing variations.
    """
    if period <= 0:
        raise ValueError("period must be greater than 0")

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    phase = ((time - float(epoch) + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def resample_phase_curve(phase: np.ndarray, flux: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample a phase-folded curve onto a fixed phase grid.

    Args:
        phase: Phase array, typically in ``[-0.5, 0.5)``.
        flux: Flux array aligned with ``phase``.
        target_length: Number of phase bins in output.

    Returns:
        Tuple ``(phase_grid, flux_grid)``.

    Raises:
        ValueError: If ``target_length`` is not greater than 1.
    """
    if target_length <= 1:
        raise ValueError("target_length must be greater than 1")

    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)
    phase_grid = np.linspace(-0.5, 0.5, target_length)
    flux_grid = np.interp(phase_grid, phase, flux)
    return phase_grid, flux_grid


def preprocess_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    target_length: int,
    normalize_method: str = "zscore",
    detrend_order: int = 2,
    smooth_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the standard preprocessing chain for one light curve.

    Args:
        time: Input time array.
        flux: Input flux array.
        target_length: Fixed output length for downstream models.
        normalize_method: Normalization method (``zscore`` or ``minmax``).
        detrend_order: Polynomial order used for detrending.
        smooth_window: Optional smoothing window size.

    Returns:
        Tuple ``(time_resampled, flux_preprocessed)``.

    Notes:
        The sequence of operations is designed for robustness in mixed-quality
        inputs: impute -> detrend -> optional smooth -> resample -> normalize.
    """
    # Fill gaps first so detrending/interpolation are numerically stable.
    time_clean, flux_clean = impute_missing_flux(time=time, flux=flux)
    flux_detrended = detrend_flux(time=time_clean, flux=flux_clean, poly_order=detrend_order)

    if smooth_window is not None:
        flux_detrended = smooth_flux(flux=flux_detrended, window=smooth_window)

    time_resampled, flux_resampled = resample_light_curve(
        time=time_clean,
        flux=flux_detrended,
        target_length=target_length,
    )
    flux_norm = normalize_flux(flux=flux_resampled, method=normalize_method)
    return time_resampled, flux_norm
