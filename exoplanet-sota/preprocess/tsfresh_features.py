"""tsfresh feature extraction utilities for light-curve baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


def sequences_to_long_dataframe(times: list[np.ndarray], fluxes: list[np.ndarray]) -> pd.DataFrame:
    """Convert sequence arrays to tsfresh long-format dataframe.

    Args:
        times: List of time arrays.
        fluxes: List of flux arrays.

    Returns:
        Long dataframe with columns: id, time, flux.
    """
    chunks: list[pd.DataFrame] = []
    for i, (t, f) in enumerate(zip(times, fluxes)):
        chunks.append(pd.DataFrame({"id": i, "time": t, "flux": f}))
    return pd.concat(chunks, ignore_index=True)


def extract_tsfresh_features(times: list[np.ndarray], fluxes: list[np.ndarray]) -> pd.DataFrame:
    """Extract a large tsfresh feature matrix.

    Args:
        times: List of time arrays.
        fluxes: List of flux arrays.

    Returns:
        Feature dataframe indexed by sample id.
    """
    long_df = sequences_to_long_dataframe(times, fluxes)
    feats = extract_features(long_df, column_id="id", column_sort="time", disable_progressbar=True)
    impute(feats)
    return feats
