"""Feature engineering for Kepler-style light curves.

This module provides handcrafted summary features for classical ML baselines.
Features include descriptive statistics and simple frequency-domain signals.

Main public functions:
- ``extract_light_curve_features``: compute one feature dictionary.
- ``build_feature_table``: apply extraction to a batch of curves.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew


def extract_light_curve_features(
    time: np.ndarray,
    flux: np.ndarray,
    quantiles: Iterable[float] = (0.1, 0.25, 0.75, 0.9),
) -> dict[str, float]:
    """Extract statistical and periodogram features for one light curve.

    Args:
        time: Time array.
        flux: Flux array.
        quantiles: Quantiles to include as scalar features.

    Returns:
        Dictionary mapping feature name to scalar value.

    Notes:
        ``depth_proxy`` is a simple approximation (median minus minimum flux),
        not a physically fitted transit depth.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    std_val = float(np.std(flux))
    if std_val < 1e-12:
        # Constant sequences can make higher moments numerically unstable.
        skewness_val = 0.0
        kurtosis_val = 0.0
    else:
        skewness_val = float(skew(flux, bias=False))
        kurtosis_val = float(kurtosis(flux, fisher=True, bias=False))

    features: dict[str, float] = {
        "mean": float(np.mean(flux)),
        "std": std_val,
        "min": float(np.min(flux)),
        "max": float(np.max(flux)),
        "median": float(np.median(flux)),
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "range": float(np.max(flux) - np.min(flux)),
        "depth_proxy": float(np.median(flux) - np.min(flux)),
    }

    for q in quantiles:
        key = f"q_{str(q).replace('.', '_')}"
        features[key] = float(np.quantile(flux, q))

    # Median cadence provides a robust sampling-rate estimate for periodogram.
    median_dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    fs = 1.0 / median_dt if median_dt > 0 else 1.0

    freqs, power = periodogram(flux, fs=fs)
    if len(freqs) > 1:
        # Skip the zero-frequency component to focus on periodic content.
        peak_idx = int(np.argmax(power[1:]) + 1)
        features["periodogram_peak_frequency"] = float(freqs[peak_idx])
        features["periodogram_peak_power"] = float(power[peak_idx])
    else:
        features["periodogram_peak_frequency"] = 0.0
        features["periodogram_peak_power"] = 0.0

    return features


def build_feature_table(times: np.ndarray, fluxes: np.ndarray) -> pd.DataFrame:
    """Build a feature dataframe from batched time/flux arrays.

    Args:
        times: Array-like batch of time vectors.
        fluxes: Array-like batch of flux vectors.

    Returns:
        Dataframe with one row per light curve.
    """
    rows = [extract_light_curve_features(t, f) for t, f in zip(times, fluxes)]
    return pd.DataFrame(rows)
