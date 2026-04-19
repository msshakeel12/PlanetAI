"""Kepler DR25 metadata and light-curve loading utilities.

This module downloads DR25 KOI/TCE-style metadata from the NASA Exoplanet
Archive and optionally fetches corresponding Kepler light curves from MAST via
Lightkurve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


@dataclass
class KeplerSeries:
    """Container for a single Kepler time series."""

    sample_id: str
    kepid: int
    time: list[float]
    flux: list[float]
    label: int
    period: float
    epoch: float


def build_dr25_query(limit: int | None = None, where: str | None = None) -> str:
    """Build ADQL query for DR25 KOI table.

    Args:
        limit: Optional row cap.
        where: Optional ADQL WHERE clause.

    Returns:
        ADQL query string.
    """
    cols = [
        "kepid",
        "kepoi_name",
        "koi_disposition",
        "koi_pdisposition",
        "koi_period",
        "koi_time0bk",
        "koi_duration",
        "koi_depth",
        "koi_prad",
        "koi_steff",
    ]
    top = f"TOP {int(limit)} " if limit is not None else ""
    query = f"SELECT {top}{', '.join(cols)} FROM q1_q17_dr25_koi"
    if where:
        query += f" WHERE {where}"
    return query


def download_dr25_metadata(output_csv: Path, limit: int | None = None, where: str | None = None) -> pd.DataFrame:
    """Download DR25 KOI metadata and save as CSV.

    Args:
        output_csv: Output metadata CSV path.
        limit: Optional row cap.
        where: Optional ADQL WHERE clause.

    Returns:
        Metadata dataframe with normalized lowercase columns.

    Raises:
        requests.HTTPError: If TAP endpoint returns non-2xx response.
    """
    q = build_dr25_query(limit=limit, where=where)
    resp = requests.get(TAP_URL, params={"query": q, "format": "csv"}, timeout=120)
    resp.raise_for_status()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text(resp.text, encoding="utf-8")

    df = pd.read_csv(output_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def make_binary_labels(
    metadata: pd.DataFrame,
    positive_dispositions: Iterable[str] = ("CONFIRMED",),
    negative_dispositions: Iterable[str] = ("FALSE POSITIVE", "CANDIDATE"),
) -> pd.DataFrame:
    """Map dispositions to a binary label for supervised learning.

    Args:
        metadata: Metadata dataframe.
        positive_dispositions: Dispositions mapped to class 1.
        negative_dispositions: Dispositions mapped to class 0.

    Returns:
        Filtered dataframe with additional `label` column.

    Notes:
        This is a pragmatic label policy. Many studies treat CANDIDATE
        separately; here it is default-negative to align with a stricter
        planet-vs-nonplanet binary objective.
    """
    df = metadata.copy()
    pos = {x.upper() for x in positive_dispositions}
    neg = {x.upper() for x in negative_dispositions}

    disp = df["koi_disposition"].astype(str).str.upper()
    keep_mask = disp.isin(pos.union(neg))
    df = df.loc[keep_mask].copy()
    df["label"] = disp.loc[keep_mask].isin(pos).astype(int)
    return df


def load_kepler_lightcurves_via_lightkurve(
    metadata: pd.DataFrame,
    cache_dir: Path,
    max_targets: int | None = None,
    cadence: str = "long",
) -> list[KeplerSeries]:
    """Download and stitch Kepler light curves from MAST.

    Args:
        metadata: Labeled metadata dataframe with kepid/period/epoch/label.
        cache_dir: Directory for downloaded products.
        max_targets: Optional cap to bound runtime.
        cadence: Lightkurve cadence selector.

    Returns:
        List of loaded KeplerSeries entries.

    Raises:
        ImportError: If Lightkurve is not installed.
    """
    try:
        import lightkurve as lk  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lightkurve is required. Install from requirements.txt") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = metadata.head(max_targets) if max_targets is not None else metadata

    out: list[KeplerSeries] = []
    for _, row in rows.iterrows():
        kepid = int(row["kepid"])
        target = f"KIC {kepid}"
        try:
            sr = lk.search_lightcurve(target, mission="Kepler", cadence=cadence)
            if len(sr) == 0:
                continue
            coll = sr.download_all(download_dir=str(cache_dir))
            if coll is None or len(coll) == 0:
                continue
            lc = coll.stitch().remove_nans().remove_outliers(sigma=7)

            out.append(
                KeplerSeries(
                    sample_id=str(row.get("kepoi_name", target)),
                    kepid=kepid,
                    time=lc.time.value.tolist(),
                    flux=lc.flux.value.tolist(),
                    label=int(row["label"]),
                    period=float(row.get("koi_period", 0.0) or 0.0),
                    epoch=float(row.get("koi_time0bk", 0.0) or 0.0),
                )
            )
        except Exception:
            # Keep ingestion robust to per-target failures.
            continue

    return out
