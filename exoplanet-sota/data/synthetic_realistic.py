"""Realistic synthetic light-curve generation with correlated stellar noise.

This module creates synthetic transit-like curves with Gaussian-process-driven
stellar variability and additional instrumental noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


@dataclass
class SyntheticConfig:
    """Configuration for realistic synthetic light-curve generation."""

    n_points: int = 4000
    t_min: float = 0.0
    t_max: float = 27.0
    baseline_noise_std: float = 2e-4
    gp_length_scale: float = 3.0
    gp_nu: float = 1.5
    gp_alpha: float = 1e-4
    use_gp_noise: bool = True


def inject_box_transit(
    time: np.ndarray,
    period: float,
    epoch: float,
    duration_days: float,
    depth: float,
) -> np.ndarray:
    """Inject a simple box-shaped transit profile.

    Args:
        time: Time grid.
        period: Orbital period.
        epoch: Transit epoch.
        duration_days: Transit duration in days.
        depth: Relative transit depth.

    Returns:
        Multiplicative transit model close to unity.
    """
    flux = np.ones_like(time)
    if period <= 0 or duration_days <= 0:
        return flux

    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) <= (0.5 * duration_days)
    flux[in_transit] -= depth
    return flux


def sample_gp_variability(time: np.ndarray, cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample correlated stellar variability using a Matern Gaussian Process.

    Args:
        time: Time grid.
        cfg: Synthetic generation config.
        rng: Random generator.

    Returns:
        Correlated noise realization.
    """
    kernel = Matern(length_scale=cfg.gp_length_scale, nu=cfg.gp_nu) + WhiteKernel(noise_level=cfg.gp_alpha)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=cfg.gp_alpha, random_state=int(rng.integers(0, 1_000_000)))

    # Fit on zero target only to instantiate covariance structure.
    gp.fit(time.reshape(-1, 1), np.zeros_like(time))
    sampled = gp.sample_y(time.reshape(-1, 1), random_state=int(rng.integers(0, 1_000_000))).reshape(-1)
    return sampled


def sample_correlated_noise_fallback(
    time: np.ndarray,
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample lightweight correlated noise without sklearn GP internals.

    Args:
        time: Time grid.
        cfg: Synthetic generation config.
        rng: Random generator.

    Returns:
        Correlated noise realization.
    """
    n = time.shape[0]
    white = rng.normal(0.0, 1.0, size=n)
    # Random-walk-like component plus weak sinusoid to mimic variability.
    walk = np.cumsum(white)
    walk = walk - np.mean(walk)
    walk_std = float(np.std(walk))
    if walk_std > 0:
        walk = walk / walk_std
    phase = rng.uniform(0.0, 2.0 * np.pi)
    freq = rng.uniform(0.02, 0.12)
    sinusoid = np.sin(2.0 * np.pi * freq * (time - time.min()) + phase)
    corr = 0.7 * walk + 0.3 * sinusoid
    corr_std = float(np.std(corr))
    if corr_std > 0:
        corr = corr / corr_std
    return corr * (5.0 * cfg.baseline_noise_std)


def generate_realistic_synthetic_curve(
    label: int,
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Generate one realistic synthetic light curve.

    Args:
        label: Binary label (1 injects transit, 0 no injected transit).
        cfg: Synthetic generation config.
        rng: Random generator.

    Returns:
        Tuple: (time, flux, period, epoch).
    """
    time = np.linspace(cfg.t_min, cfg.t_max, cfg.n_points)

    period = float(rng.uniform(2.0, 15.0))
    epoch = float(rng.uniform(cfg.t_min, cfg.t_min + period))
    duration = float(rng.uniform(0.08, 0.35))
    depth = float(rng.uniform(4e-4, 4e-3))

    clean_transit = inject_box_transit(time, period, epoch, duration_days=duration, depth=depth if label == 1 else 0.0)
    gp_noise = (
        sample_gp_variability(time, cfg=cfg, rng=rng)
        if cfg.use_gp_noise
        else sample_correlated_noise_fallback(time, cfg=cfg, rng=rng)
    )
    instrumental_noise = rng.normal(0.0, cfg.baseline_noise_std, size=time.shape[0])

    flux_realistic = clean_transit + gp_noise + instrumental_noise
    return time, flux_realistic, period, epoch
