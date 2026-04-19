"""CLI for downloading Kepler candidate metadata from NASA Exoplanet Archive.

This module provides the data-ingestion entry point for the pipeline. It keeps
remote query logic separate from CLI concerns by delegating retrieval/parsing to
``src.data.exoplanet_archive``.

Main public functions:
- ``run_download``: programmatic metadata download utility.
- ``main``: command-line entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.exoplanet_archive import build_tap_query, download_metadata_dataframe
from src.utils.paths import RAW_DATA_DIR, ensure_project_directories


DEFAULT_COLUMNS = [
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_steff",
]


def run_download(
    output_path: Path,
    table: str,
    where_clause: str | None,
    timeout_seconds: int,
    limit: int | None,
) -> Path:
    """Download metadata and persist it to CSV.

    Args:
        output_path: Destination CSV path.
        table: Exoplanet Archive table name.
        where_clause: Optional ADQL filter clause.
        timeout_seconds: HTTP request timeout.
        limit: Optional maximum number of rows.

    Returns:
        Path to the written metadata CSV.

    Raises:
        requests.HTTPError: If the remote TAP query fails.
    """
    # Build query first so data-source intent is explicit in logs/debugging.
    query = build_tap_query(
        table=table,
        columns=DEFAULT_COLUMNS,
        where_clause=where_clause,
        limit=limit,
    )
    df = download_metadata_dataframe(query=query, timeout_seconds=timeout_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for metadata download CLI.

    Returns:
        Configured ``ArgumentParser`` for metadata download options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DATA_DIR / "kepler_metadata.csv",
        help="Output CSV path for downloaded metadata.",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="cumulative",
        help="Exoplanet Archive table name.",
    )
    parser.add_argument(
        "--where",
        type=str,
        default=None,
        help="Optional SQL WHERE clause.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick experiments.",
    )
    return parser


def main() -> None:
    """Run metadata download from the command line.

    Returns:
        None.
    """
    # Ensure directory structure exists before any writes.
    ensure_project_directories()
    args = build_parser().parse_args()
    output = run_download(
        output_path=args.output,
        table=args.table,
        where_clause=args.where,
        timeout_seconds=args.timeout,
        limit=args.limit,
    )
    print(f"Saved metadata to {output}")


if __name__ == "__main__":
    main()
