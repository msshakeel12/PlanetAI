"""Utilities for downloading and reading Kepler light curves from MAST.

This module isolates external real-data access so dataset-building logic can
remain agnostic to remote API details. It currently uses Lightkurve as the
high-level client for MAST search/download/stitch operations.

Main public functions/classes:
- ``KeplerLightCurve``: typed container for one stitched curve.
- ``download_kepler_light_curve``: download + basic cleaning for one KIC ID.
"""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

T = TypeVar("T")


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


def _run_with_optional_timeout(
    fn: Callable[[], T],
    timeout_seconds: float | None,
    stage_name: str,
) -> T:
    """Run a callable with an optional wall-clock timeout.

    Args:
        fn: Zero-argument callable to execute.
        timeout_seconds: Optional timeout budget in seconds.
        stage_name: Human-readable stage label for error messages.

    Returns:
        Result returned by ``fn``.

    Raises:
        TimeoutError: If execution exceeds ``timeout_seconds``.

    Notes:
        Timeout enforcement uses ``SIGALRM`` when available and when running on
        the main thread. In other contexts, timeout values are accepted but not
        enforced to preserve compatibility across platforms/runtimes.
    """
    if timeout_seconds is None or timeout_seconds <= 0:
        return fn()

    supports_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    if not supports_alarm or threading.current_thread() is not threading.main_thread():
        return fn()

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _timeout_handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"{stage_name} timed out after {float(timeout_seconds):.1f}s")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def download_kepler_light_curve(
    kepid: int,
    download_dir: Path,
    mission: str = "Kepler",
    cadence: str = "long",
    search_timeout_seconds: float | None = None,
    download_timeout_seconds: float | None = None,
) -> KeplerLightCurve:
    """Download and stitch Kepler light curves for one KIC target.

    Args:
        kepid: Kepler Input Catalog identifier.
        download_dir: Local directory for cached products.
        mission: Mission name passed to Lightkurve search.
        cadence: Cadence type (for example, ``long`` or ``short``).
        search_timeout_seconds: Optional timeout budget for MAST search.
        download_timeout_seconds: Optional timeout budget for MAST download and
            stitch operations.

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
    search_result = _run_with_optional_timeout(
        lambda: lk.search_lightcurve(target, mission=mission, cadence=cadence),
        timeout_seconds=search_timeout_seconds,
        stage_name=f"MAST search for {target}",
    )
    if len(search_result) == 0:
        raise ValueError(f"No light curves found in MAST for {target}")

    download_dir.mkdir(parents=True, exist_ok=True)
    collection = _run_with_optional_timeout(
        lambda: search_result.download_all(download_dir=str(download_dir)),
        timeout_seconds=download_timeout_seconds,
        stage_name=f"MAST download for {target}",
    )
    if collection is None or len(collection) == 0:
        raise ValueError(f"Failed to download light curves for {target}")

    stitched = _run_with_optional_timeout(
        lambda: collection.stitch().remove_nans(),
        timeout_seconds=download_timeout_seconds,
        stage_name=f"MAST stitch for {target}",
    )
    # Robust cleaning to reduce obvious outliers before shared preprocessing.
    stitched = stitched.remove_outliers(sigma=7)

    time = np.asarray(stitched.time.value, dtype=float)
    flux = np.asarray(stitched.flux.value, dtype=float)
    if time.size == 0 or flux.size == 0:
        raise ValueError(f"Empty light curve after stitching for {target}")

    return KeplerLightCurve(kepid=int(kepid), time=time, flux=flux)
