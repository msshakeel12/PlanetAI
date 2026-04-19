"""Configuration loading utilities.

This module currently provides YAML loading used by the pipeline orchestrator.
Keeping config I/O in one place makes future validation/schema checks easier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration from disk.

    Args:
        path: YAML configuration file path.

    Returns:
        Parsed configuration dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
