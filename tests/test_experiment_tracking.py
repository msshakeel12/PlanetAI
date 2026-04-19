"""Unit tests for lightweight experiment tracking utilities."""

from pathlib import Path

from src.utils.experiment_tracking import ExperimentTracker


def test_tracker_writes_run_files(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments")
    run_id = tracker.log_run(
        stage="unit_test",
        params={"a": 1},
        metrics={"accuracy": 0.9},
        artifacts={"model": "x.joblib"},
    )

    run_path = tracker.runs_dir / f"{run_id}.json"
    assert run_path.exists()
    assert tracker.summary_path.exists()
