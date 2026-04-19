"""IO helpers for JSON and tabular artifacts.

This module centralizes common persistence routines used by training and
evaluation scripts to keep output formatting consistent.

Main public functions:
- ``save_json``
- ``save_dataframe``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Persist a dictionary as pretty-printed JSON.

    Args:
        data: Mapping to serialize.
        output_path: Destination path.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Persist dataframe to CSV without index.

    Args:
        df: Dataframe to write.
        output_path: Destination path.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
