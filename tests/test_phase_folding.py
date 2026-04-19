"""Unit tests for phase folding and phase-grid resampling helpers."""

import numpy as np

from src.features.preprocessing import phase_fold_light_curve, resample_phase_curve


def test_phase_fold_outputs_sorted_phase() -> None:
    time = np.array([3.0, 1.0, 2.0, 4.0])
    flux = np.array([1.0, 0.9, 1.1, 1.0])
    phase, folded_flux = phase_fold_light_curve(time=time, flux=flux, period=2.0, epoch=0.0)

    assert phase.shape == folded_flux.shape
    assert np.all(np.diff(phase) >= 0)


def test_resample_phase_curve_length() -> None:
    phase = np.linspace(-0.5, 0.5, 30)
    flux = np.sin(2 * np.pi * phase)
    grid, out_flux = resample_phase_curve(phase=phase, flux=flux, target_length=128)

    assert grid.shape == (128,)
    assert out_flux.shape == (128,)
