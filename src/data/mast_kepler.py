"""Utilities for downloading and reading Kepler light curves from MAST.

This module isolates external real-data access so dataset-building logic can
remain agnostic to remote API details. It currently uses Lightkurve as the
high-level client for MAST search/download/stitch operations.

Main public functions/classes:
- ``KeplerLightCurve``: typed container for one stitched curve.
- ``download_kepler_light_curve``: download + basic cleaning for one KIC ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class KeplerLightCurve:
    """Container for one stitched Kepler light curve.

    Attributes:
        kepid: KIC identifier.
        time: Time array (typically BKJD days from Lightkurve).
        flux: Flux array aligned with ``time``.
    """

    kepid: int
    time: np.ndarray
    flux: np.ndarray


def _import_lightkurve() -> object:
    """Import Lightkurve lazily.

    Returns:
        Imported ``lightkurve`` module object.

    Raises:
        ImportError: If optional real-ingestion dependencies are missing.
    """
    try:
        import lightkurve as lk  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "lightkurve is required for real MAST ingestion. "
            "Install optional dependencies with: pip install lightkurve astropy"
        ) from exc
    return lk


def download_kepler_light_curve(
    kepid: int,
    download_dir: Path,
    mission: str = "Kepler",
    cadence: str = "long",
) -> KeplerLightCurve:
    """Download and stitch Kepler light curves for one KIC target.

    Args:
        kepid: Kepler Input Catalog identifier.
        download_dir: Local directory for cached products.
        mission: Mission name passed to Lightkurve search.
        cadence: Cadence type (for example, ``long`` or ``short``).

    Returns:
        ``KeplerLightCurve`` containing cleaned time/flux arrays.

    Raises:
        ValueError: If no data is found or download/stitch produces empty data.
        ImportError: If Lightkurve is unavailable.

    Notes:
        The cleaning here is intentionally lightweight (remove NaNs and simple
        sigma clipping). Downstream preprocessing performs additional detrending
        and normalization.
    """
    lk = _import_lightkurve()

    target = f"KIC {int(kepid)}"
    search_result = lk.search_lightcurve(target, mission=mission, cadence=cadence)
    if len(search_result) == 0:
        raise ValueError(f"No light curves found in MAST for {target}")

    download_dir.mkdir(parents=True, exist_ok=True)
    collection = search_result.download_all(download_dir=str(download_dir))
    if collection is None or len(collection) == 0:
        raise ValueError(f"Failed to download light curves for {target}")

    stitched = collection.stitch().remove_nans()
    # Robust cleaning to reduce obvious outliers before shared preprocessing.
    stitched = stitched.remove_outliers(sigma=7)

    time = np.asarray(stitched.time.value, dtype=float)
    flux = np.asarray(stitched.flux.value, dtype=float)
    if time.size == 0 or flux.size == 0:
        raise ValueError(f"Empty light curve after stitching for {target}")

    return KeplerLightCurve(kepid=int(kepid), time=time, flux=flux)
