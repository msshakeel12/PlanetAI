"""Unit tests for task labeling and grouped CNN split helpers."""

import numpy as np
import pandas as pd

from src.data.dataset_builder import derive_earth_sized_labels
from src.models.train_cnn import _grouped_train_val_test_split


def test_derive_earth_sized_labels_filters_and_maps_binary() -> None:
    metadata = pd.DataFrame(
        {
            "koi_disposition": ["CONFIRMED", "CONFIRMED", "FALSE POSITIVE", "CANDIDATE", "NOT DISPOSITIONED"],
            "koi_prad": [1.1, 2.4, 1.0, 2.1, 1.2],
            "kepid": [1001, 1002, 1003, 1004, 1005],
        }
    )

    labels, filtered = derive_earth_sized_labels(
        metadata=metadata,
        label_column="koi_disposition",
        earth_size_max_radius=1.5,
        include_candidates_as_positive=False,
    )

    assert labels.tolist() == [1, 0, 0, 0]
    assert filtered["kepid"].tolist() == [1001, 1002, 1003, 1004]


def test_grouped_train_val_test_split_prevents_group_leakage() -> None:
    n_samples = 24
    x = np.random.default_rng(42).normal(size=(n_samples, 32)).astype(np.float32)
    y = np.array([0, 1] * 12, dtype=np.int64)
    groups = np.repeat(np.arange(12), 2)

    _, _, _, _, _, _, g_train, g_val, g_test = _grouped_train_val_test_split(
        x=x,
        y=y,
        groups=groups,
        test_size=0.2,
        val_size=0.25,
        random_seed=42,
    )

    train_groups = set(g_train.tolist())
    val_groups = set(g_val.tolist())
    test_groups = set(g_test.tolist())
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
