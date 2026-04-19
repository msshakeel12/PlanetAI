"""Unit tests for preprocessing primitives used in dataset building."""

import numpy as np

from src.features.preprocessing import (
    impute_missing_flux,
    normalize_flux,
    preprocess_light_curve,
    resample_light_curve,
)


def test_impute_missing_flux_fills_nan() -> None:
    time = np.arange(5, dtype=float)
    flux = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    _, out_flux = impute_missing_flux(time=time, flux=flux)
    assert not np.isnan(out_flux).any()


def test_resample_light_curve_length() -> None:
    time = np.linspace(0.0, 1.0, 10)
    flux = np.sin(time)
    out_time, out_flux = resample_light_curve(time=time, flux=flux, target_length=32)
    assert len(out_time) == 32
    assert len(out_flux) == 32


def test_normalize_flux_zscore_mean_close_zero() -> None:
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    norm = normalize_flux(flux, method="zscore")
    assert abs(float(np.mean(norm))) < 1e-8


def test_preprocess_light_curve_output_shape() -> None:
    time = np.linspace(0.0, 4.0, 20)
    flux = np.cos(time)
    out_time, out_flux = preprocess_light_curve(time, flux, target_length=50, smooth_window=3)
    assert out_time.shape == (50,)
    assert out_flux.shape == (50,)
