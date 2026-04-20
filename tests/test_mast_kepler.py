"""Unit tests for MAST Kepler ingestion helpers without network access."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.data import mast_kepler


class _FakeStitched:
    def __init__(self) -> None:
        self.time = type("_Time", (), {"value": np.array([1.0, 2.0, 3.0])})()
        self.flux = type("_Flux", (), {"value": np.array([0.1, 0.2, 0.3])})()

    def remove_nans(self) -> "_FakeStitched":
        return self

    def remove_outliers(self, sigma: float = 7.0) -> "_FakeStitched":
        _ = sigma
        return self


class _FakeCollection:
    def __len__(self) -> int:
        return 1

    def stitch(self) -> _FakeStitched:
        return _FakeStitched()


class _FakeSearchResult:
    def __len__(self) -> int:
        return 1

    def download_all(self, download_dir: str) -> _FakeCollection:
        _ = download_dir
        return _FakeCollection()


class _FakeLightkurve:
    def search_lightcurve(self, target: str, mission: str, cadence: str) -> _FakeSearchResult:
        _ = (target, mission, cadence)
        return _FakeSearchResult()


def test_download_kepler_light_curve_non_network_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mast_kepler, "_import_lightkurve", lambda: _FakeLightkurve())

    lc = mast_kepler.download_kepler_light_curve(
        kepid=123456,
        download_dir=tmp_path,
        search_timeout_seconds=1.0,
        download_timeout_seconds=1.0,
    )

    assert lc.kepid == 123456
    assert lc.time.shape == (3,)
    assert lc.flux.shape == (3,)


def test_download_kepler_light_curve_search_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _SlowLightkurve:
        def search_lightcurve(self, target: str, mission: str, cadence: str) -> _FakeSearchResult:
            _ = (target, mission, cadence)
            time.sleep(0.2)
            return _FakeSearchResult()

    monkeypatch.setattr(mast_kepler, "_import_lightkurve", lambda: _SlowLightkurve())

    with pytest.raises(TimeoutError):
        mast_kepler.download_kepler_light_curve(
            kepid=123456,
            download_dir=tmp_path,
            search_timeout_seconds=0.05,
            download_timeout_seconds=1.0,
        )
