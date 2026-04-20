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


def _sanitize_numeric_array(values: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Return finite float array with NaN/Inf replaced by fallback."""
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(arr, nan=fallback, posinf=fallback, neginf=fallback)


def safe_fixed_length_view(
    x: np.ndarray,
    y: np.ndarray,
    target_length: int,
    x_min: float | None = None,
    x_max: float | None = None,
    fallback_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Safely resample ``(x, y)`` onto a fixed-length grid.

    Args:
        x: Input x-axis values.
        y: Input signal values.
        target_length: Number of output samples.
        x_min: Optional left edge for output grid.
        x_max: Optional right edge for output grid.
        fallback_value: Value used if resampling cannot be computed.

    Returns:
        Tuple ``(x_grid, y_grid)`` with deterministic finite arrays.
    """
    if target_length <= 1:
        raise ValueError("target_length must be greater than 1")

    y_fallback = np.full(target_length, float(fallback_value), dtype=float)
    x_fallback_min = -0.5 if x_min is None else float(x_min)
    x_fallback_max = 0.5 if x_max is None else float(x_max)
    if x_fallback_max <= x_fallback_min:
        x_fallback_max = x_fallback_min + 1.0
    x_fallback = np.linspace(x_fallback_min, x_fallback_max, target_length)

    x_arr = _sanitize_numeric_array(x, fallback=x_fallback_min)
    y_arr = _sanitize_numeric_array(y, fallback=fallback_value)
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.shape != y_arr.shape:
        return x_fallback, y_fallback

    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    # Remove repeated x values so interpolation remains well-defined.
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if unique_x.size < 2:
        return x_fallback, y_fallback
    unique_y = y_sorted[unique_indices]

    left = float(unique_x.min()) if x_min is None else float(x_min)
    right = float(unique_x.max()) if x_max is None else float(x_max)
    if right <= left:
        return x_fallback, y_fallback

    x_grid = np.linspace(left, right, target_length)
    y_grid = np.interp(x_grid, unique_x, unique_y)
    y_grid = _sanitize_numeric_array(y_grid, fallback=fallback_value)
    return x_grid, y_grid


def safe_phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Safely phase-fold a curve, returning deterministic finite arrays.

    If period/epoch are invalid, this returns a stable fallback phase axis with
    flux values sanitized from input data.
    """
    time_arr = _sanitize_numeric_array(time, fallback=0.0)
    flux_arr = _sanitize_numeric_array(flux, fallback=0.0)
    n = int(min(time_arr.size, flux_arr.size))
    if n < 2:
        return np.linspace(-0.5, 0.5, max(n, 2)), np.zeros(max(n, 2), dtype=float)

    time_arr = time_arr[:n]
    flux_arr = flux_arr[:n]

    period_f = float(period) if np.isfinite(period) else 0.0
    epoch_f = float(epoch) if np.isfinite(epoch) else 0.0

    # Treat already phase-like inputs as pre-folded and just sort by phase.
    looks_like_phase = float(np.min(time_arr)) >= -0.6 and float(np.max(time_arr)) <= 0.6
    if looks_like_phase:
        order = np.argsort(time_arr)
        return time_arr[order], flux_arr[order]

    if period_f <= 0.0:
        phase_fallback = np.linspace(-0.5, 0.5, n)
        _, flux_fallback = safe_fixed_length_view(
            x=np.arange(n, dtype=float),
            y=flux_arr,
            target_length=n,
            x_min=0.0,
            x_max=float(n - 1),
            fallback_value=0.0,
        )
        return phase_fallback, flux_fallback

    try:
        phase, folded_flux = phase_fold_light_curve(time_arr, flux_arr, period=period_f, epoch=epoch_f)
        phase = _sanitize_numeric_array(phase, fallback=0.0)
        folded_flux = _sanitize_numeric_array(folded_flux, fallback=0.0)
        if phase.size < 2 or folded_flux.size < 2:
            raise ValueError("Insufficient points after phase fold")
        return phase, folded_flux
    except Exception:
        phase_fallback = np.linspace(-0.5, 0.5, n)
        _, flux_fallback = safe_fixed_length_view(
            x=np.arange(n, dtype=float),
            y=flux_arr,
            target_length=n,
            x_min=0.0,
            x_max=float(n - 1),
            fallback_value=0.0,
        )
        return phase_fallback, flux_fallback


def build_global_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    target_length: int,
) -> np.ndarray:
    """Build a fixed-length global phase-folded transit view."""
    phase, folded_flux = safe_phase_fold(time=time, flux=flux, period=period, epoch=epoch)
    _, global_flux = safe_fixed_length_view(
        x=phase,
        y=folded_flux,
        target_length=target_length,
        x_min=-0.5,
        x_max=0.5,
        fallback_value=0.0,
    )
    return _sanitize_numeric_array(global_flux, fallback=0.0).astype(np.float32)


def build_local_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    target_length: int,
    window_half_width: float = 0.1,
) -> np.ndarray:
    """Build a fixed-length local transit-centered phase view."""
    phase, folded_flux = safe_phase_fold(time=time, flux=flux, period=period, epoch=epoch)

    half_width = float(window_half_width)
    if half_width <= 0.0 or half_width > 0.5:
        half_width = 0.1
    local_mask = np.abs(phase) <= half_width
    if int(np.sum(local_mask)) < 2:
        local_phase = phase
        local_flux = folded_flux
    else:
        local_phase = phase[local_mask]
        local_flux = folded_flux[local_mask]

    _, local_view = safe_fixed_length_view(
        x=local_phase,
        y=local_flux,
        target_length=target_length,
        x_min=-half_width,
        x_max=half_width,
        fallback_value=0.0,
    )
    return _sanitize_numeric_array(local_view, fallback=0.0).astype(np.float32)


def _select_phase_window(
    phase: np.ndarray,
    flux: np.ndarray,
    center_phase: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a circular phase window and return centered phase offsets."""
    phase_arr = _sanitize_numeric_array(phase, fallback=0.0)
    flux_arr = _sanitize_numeric_array(flux, fallback=0.0)

    phase01 = (phase_arr + 0.5) % 1.0
    center01 = (float(center_phase) + 0.5) % 1.0
    delta = ((phase01 - center01 + 0.5) % 1.0) - 0.5

    width = float(half_width)
    if width <= 0.0 or width > 0.5:
        width = 0.1
    mask = np.abs(delta) <= width
    if int(np.sum(mask)) < 2:
        return delta, flux_arr
    return delta[mask], flux_arr[mask]


def build_odd_even_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    target_length: int,
    window_half_width: float = 0.1,
) -> np.ndarray:
    """Build a 2-channel odd/even transit diagnostic view.

    Fallback logic:
    - If period/epoch are invalid, or fewer than 3 odd/even transit cycles are
      identified, this returns two duplicated local-view channels.

    Debug examples:
    - ``print(build_odd_even_view(t, f, 10.0, 0.2, 256).shape)``
    - ``print(np.isfinite(build_odd_even_view(t, f, -1.0, 0.0, 256)).all())``
    """

    def _fallback_from_local() -> np.ndarray:
        local = build_local_view(
            time=time,
            flux=flux,
            period=period,
            epoch=epoch,
            target_length=target_length,
        )
        return np.stack([local, local], axis=0).astype(np.float32)

    def _build_branch(mask: np.ndarray | None = None) -> np.ndarray:
        if mask is not None and int(np.sum(mask)) >= 2:
            phase_branch, flux_branch = safe_phase_fold(
                time=np.asarray(time)[mask],
                flux=np.asarray(flux)[mask],
                period=period,
                epoch=epoch,
            )
        else:
            phase_branch, flux_branch = safe_phase_fold(time=time, flux=flux, period=period, epoch=epoch)
        delta, flux_window = _select_phase_window(
            phase=phase_branch,
            flux=flux_branch,
            center_phase=0.0,
            half_width=window_half_width,
        )
        _, view = safe_fixed_length_view(
            x=delta,
            y=flux_window,
            target_length=target_length,
            x_min=-abs(float(window_half_width)),
            x_max=abs(float(window_half_width)),
            fallback_value=0.0,
        )
        return _sanitize_numeric_array(view, fallback=0.0).astype(np.float32)

    period_f = float(period) if np.isfinite(period) else 0.0
    epoch_f = float(epoch) if np.isfinite(epoch) else 0.0
    time_arr = _sanitize_numeric_array(time, fallback=0.0)

    if period_f <= 0.0 or time_arr.size < 2:
        return _fallback_from_local()

    cycle_idx = np.floor((time_arr - epoch_f) / period_f).astype(np.int64)
    unique_cycles = np.unique(cycle_idx)
    if unique_cycles.size < 3:
        return _fallback_from_local()

    odd_mask = (cycle_idx % 2) != 0
    even_mask = ~odd_mask

    odd_cycles = np.unique(cycle_idx[odd_mask])
    even_cycles = np.unique(cycle_idx[even_mask])
    if odd_cycles.size < 1 or even_cycles.size < 1:
        return _fallback_from_local()

    odd_view = _build_branch(mask=odd_mask)
    even_view = _build_branch(mask=even_mask)
    return np.stack([odd_view, even_view], axis=0).astype(np.float32)


def build_secondary_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    target_length: int,
    window_half_width: float = 0.1,
) -> np.ndarray:
    """Build a fixed-length phase view centered near secondary eclipse (phase 0.5).

    Fallback logic:
    - If a narrow secondary window is unreliable, this resamples the folded
      full-curve signal onto the secondary-centered window.

    Debug examples:
    - ``print(build_secondary_view(t, f, 12.0, 0.4, 128).shape)``
    - ``print(np.isfinite(build_secondary_view(t, f, 0.0, 0.0, 128)).all())``
    """
    phase, folded_flux = safe_phase_fold(time=time, flux=flux, period=period, epoch=epoch)
    delta, flux_window = _select_phase_window(
        phase=phase,
        flux=folded_flux,
        center_phase=0.5,
        half_width=window_half_width,
    )
    width = abs(float(window_half_width))
    if width <= 0.0 or width > 0.5:
        width = 0.1

    if delta.size < 2 or flux_window.size < 2:
        phase01 = (phase + 0.5) % 1.0
        center01 = (0.5 + 0.5) % 1.0
        delta_full = ((phase01 - center01 + 0.5) % 1.0) - 0.5
        delta = delta_full
        flux_window = folded_flux

    _, secondary_view = safe_fixed_length_view(
        x=delta,
        y=flux_window,
        target_length=target_length,
        x_min=-width,
        x_max=width,
        fallback_value=0.0,
    )
    return _sanitize_numeric_array(secondary_view, fallback=0.0).astype(np.float32)


def build_aux_features(row: pd.Series) -> np.ndarray:
    """Return aux features as [period, duration, depth, planet_radius, stellar_teff]."""

    def _safe_value(key: str) -> float:
        value = pd.to_numeric(row.get(key), errors="coerce")
        if pd.isna(value) or not np.isfinite(float(value)):
            return 0.0
        return float(value)

    features = np.array(
        [
            _safe_value("koi_period"),
            _safe_value("koi_duration"),
            _safe_value("koi_depth"),
            _safe_value("koi_prad"),
            _safe_value("koi_steff"),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


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
