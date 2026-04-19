"""Pytest configuration for repository-local imports.

This file ensures tests can import ``src`` modules when running from the
project root without requiring package installation.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
