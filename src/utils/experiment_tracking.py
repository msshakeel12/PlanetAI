"""Minimal experiment tracking utilities based on JSON and JSONL logs.

This module provides lightweight, dependency-free run tracking. It is intended
as a transparent bridge before adopting a larger experiment platform.

Main public class:
- ``ExperimentTracker``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.utils.io import save_json


@dataclass
class ExperimentTracker:
    """Track experiment runs and metrics under one output directory.

    Args:
        root_dir: Root tracking directory containing per-run JSON files and a
            JSONL summary file.
    """

    root_dir: Path

    def __post_init__(self) -> None:
        """Initialize tracking directories and summary paths.

        Returns:
            None.
        """
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.root_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.root_dir / "experiments.jsonl"

    def log_run(
        self,
        stage: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        artifacts: dict[str, str] | None = None,
    ) -> str:
        """Persist one run record and return its run identifier.

        Args:
            stage: Pipeline stage name (for example ``cnn``).
            params: Hyperparameters and run settings.
            metrics: Evaluation metrics.
            artifacts: Optional mapping of artifact labels to paths.

        Returns:
            Stable run id string.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stage}_{ts}"

        payload = {
            "run_id": run_id,
            "timestamp_utc": ts,
            "stage": stage,
            "params": params,
            "metrics": metrics,
            "artifacts": artifacts or {},
        }

        run_path = self.runs_dir / f"{run_id}.json"
        save_json(payload, run_path)

        # Append one valid JSON object per line for easy downstream parsing.
        with self.summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

        return run_id
