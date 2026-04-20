"""Compatibility wrapper for CNN training module name.

This file preserves legacy script/module paths and forwards to
src.models.train_cnn.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_cnn import *  # noqa: F401,F403
from src.models.train_cnn import main


if __name__ == "__main__":
    main()
