"""Phase-folding helpers for 1D and 2D folded representations."""

from __future__ import annotations

import numpy as np


def phase_fold(time: np.ndarray, flux: np.ndarray, period: float, n_bins: int = 1000) -> np.ndarray:
    """Create a 1D phase-folded flux representation.

    Args:
        time: Time values.
        flux: Flux values.
        period: Orbital period.
        n_bins: Output bins across phase range [-1, 1].

    Returns:
        Binned folded flux array with length n_bins.
    """
    if period <= 0:
        return np.full(n_bins, float(np.nanmedian(flux)))

    phases = ((time % period) / period - 0.5) * 2.0
    order = np.argsort(phases)
    sorted_phase = phases[order]
    sorted_flux = flux[order]
    return np.interp(np.linspace(-1.0, 1.0, n_bins), sorted_phase, sorted_flux)


def phase_fold_2d(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    n_folds: int = 10,
    n_bins: int = 1000,
) -> np.ndarray:
    """Build a 2D phase-folded image by stacking phase-shifted folds.

    Args:
        time: Time values.
        flux: Flux values.
        period: Orbital period.
        n_folds: Number of stacked phase-shifted rows.
        n_bins: Number of bins per fold.

    Returns:
        Array of shape (n_folds, n_bins).

    Notes:
        Cuellar-style folded images are often built from repeated windows around
        transit phase. Here we emulate that with small phase offsets so the CNN
        can learn local morphology and translational tolerance.
    """
    if period <= 0:
        base = np.full(n_bins, float(np.nanmedian(flux)))
        return np.tile(base, (n_folds, 1))

    folded_rows: list[np.ndarray] = []
    for k in range(n_folds):
        offset = (k / max(1, n_folds)) * 0.2 * period
        row = phase_fold(time + offset, flux, period=period, n_bins=n_bins)
        folded_rows.append(row)
    return np.asarray(folded_rows, dtype=float)
