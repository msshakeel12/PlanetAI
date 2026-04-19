"""Path helpers for reproducible artifact locations.

This module provides canonical project-relative paths used across scripts so
artifact locations remain stable and discoverable.

Main public items:
- directory constants (for example ``RAW_DATA_DIR``).
- ``ensure_project_directories``.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def ensure_project_directories() -> None:
    """Create expected directory structure if missing.

    Returns:
        None.
    """
    for path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
