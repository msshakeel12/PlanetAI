"""NASA Exoplanet Archive TAP ingestion helpers.

This module encapsulates query construction and response parsing for metadata
retrieval from the Exoplanet Archive TAP endpoint. It is used by the metadata
download CLI and by the full orchestrator pipeline.

Main public functions:
- ``build_tap_query``: construct ADQL query text.
- ``download_metadata_dataframe``: execute query and return parsed dataframe.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import requests

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def build_tap_query(
    table: str,
    columns: Iterable[str],
    where_clause: str | None = None,
    limit: int | None = None,
) -> str:
    """Build an ADQL query string for the TAP service.

    Args:
        table: Archive table name (for example, ``q1_q17_dr25_koi``).
        columns: Column names to select.
        where_clause: Optional ADQL ``WHERE`` predicate.
        limit: Optional row cap applied via ``TOP N`` syntax.

    Returns:
        Query string compatible with Exoplanet Archive TAP.

    Notes:
        The archive endpoint expects ADQL-like syntax; ``TOP`` is used instead
        of SQL ``LIMIT`` for cross-service compatibility.
    """
    col_expr = ", ".join(columns)
    top_clause = f"TOP {int(limit)} " if limit is not None else ""
    query = f"SELECT {top_clause}{col_expr} FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    return query


def fetch_tap_csv(query: str, timeout_seconds: int = 60) -> str:
    """Execute a TAP query and return the raw CSV response body.

    Args:
        query: ADQL query text.
        timeout_seconds: HTTP request timeout.

    Returns:
        CSV payload as text.

    Raises:
        requests.HTTPError: If the endpoint returns a non-2xx response.
        requests.RequestException: For connection/timeout failures.
    """
    response = requests.get(
        TAP_SYNC_URL,
        params={"query": query, "format": "csv"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.text


def parse_metadata_csv(csv_text: str) -> pd.DataFrame:
    """Parse Exoplanet Archive CSV text into a normalized dataframe.

    Args:
        csv_text: CSV string returned by the TAP endpoint.

    Returns:
        Dataframe with lowercase, trimmed column names.
    """
    from io import StringIO

    df = pd.read_csv(StringIO(csv_text))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def download_metadata_dataframe(query: str, timeout_seconds: int = 60) -> pd.DataFrame:
    """Download and parse metadata from a TAP query.

    Args:
        query: ADQL query text.
        timeout_seconds: HTTP request timeout.

    Returns:
        Parsed dataframe containing metadata rows.

    Raises:
        requests.HTTPError: If remote query execution fails.
    """
    raw_csv = fetch_tap_csv(query=query, timeout_seconds=timeout_seconds)
    return parse_metadata_csv(raw_csv)
