"""Unit tests for handcrafted light-curve feature extraction."""

import numpy as np

from src.features.feature_engineering import extract_light_curve_features


def test_extract_light_curve_features_contains_required_keys() -> None:
    time = np.linspace(0.0, 10.0, 200)
    flux = np.sin(time) + 0.05 * np.random.default_rng(42).normal(size=time.shape[0])
    features = extract_light_curve_features(time, flux)

    required = {
        "mean",
        "std",
        "min",
        "max",
        "median",
        "skewness",
        "kurtosis",
        "range",
        "depth_proxy",
        "periodogram_peak_frequency",
        "periodogram_peak_power",
    }
    assert required.issubset(features.keys())


def test_depth_proxy_non_negative() -> None:
    time = np.linspace(0.0, 1.0, 100)
    flux = np.ones(100)
    features = extract_light_curve_features(time, flux)
    assert features["depth_proxy"] >= 0.0
