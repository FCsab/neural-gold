from __future__ import annotations

import numpy as np
import pytest

from crypto_neural.config import FeatureConfig, LabelConfig
from crypto_neural.dataset import build_dataset, walk_forward_split
from crypto_neural.features import build_features


def test_dataset_basics(bars):
    ds = build_dataset(bars, FeatureConfig(), LabelConfig())
    assert len(ds) > 0
    assert ds.frame["label"].dtype == np.int8
    assert set(ds.frame["label"].unique()) <= {-1, 0, 1}
    assert not ds.X.isna().any().any()
    assert ds.frame["open_time"].is_monotonic_increasing
    assert ds.frame["open_time"].is_unique
    assert set(ds.frame["open_time"]).issubset(set(bars["open_time"]))
    assert ds.X.shape[1] == len(ds.feature_columns)


def test_feature_rows_match_the_standalone_builder(bars):
    cfg = FeatureConfig(feature_lag=1)
    ds = build_dataset(bars, cfg, LabelConfig())
    feats = build_features(bars, cfg)
    positions = {int(t): i for i, t in enumerate(bars["open_time"])}

    for row_index in (5, 200, len(ds) - 1):
        open_time = int(ds.frame["open_time"].iloc[row_index])
        expected = feats.iloc[positions[open_time]]
        for column in ds.feature_columns:
            assert ds.frame[column].iloc[row_index] == pytest.approx(
                float(expected[column]), rel=1e-12, abs=1e-15
            )


def test_perturbing_a_bar_leaves_its_own_features_untouched(bars):
    """With feature_lag=1, bar t's features may not see bar t at all."""
    cfg = FeatureConfig(feature_lag=1)
    labels_cfg = LabelConfig()
    base = build_dataset(bars, cfg, labels_cfg)

    index = 1500
    open_time = int(bars["open_time"].iloc[index])
    modified = bars.copy()
    for j in range(index, index + 6):
        modified.loc[j, "high"] *= 1.2
        modified.loc[j, "low"] *= 0.8
        modified.loc[j, "close"] *= 1.1

    after = build_dataset(modified, cfg, labels_cfg)
    base_row = base.frame[base.frame["open_time"] == open_time].iloc[0]
    after_row = after.frame[after.frame["open_time"] == open_time].iloc[0]

    for column in base.feature_columns:
        assert base_row[column] == pytest.approx(float(after_row[column]), rel=1e-12)


def test_walk_forward_split_is_chronological_and_complete(bars):
    ds = build_dataset(bars, FeatureConfig(), LabelConfig())
    split = walk_forward_split(ds)

    assert len(split.train) and len(split.validation) and len(split.test)
    assert (
        len(split.train) + len(split.validation) + len(split.test) == len(ds)
    )
    assert split.train["open_time"].max() < split.validation["open_time"].min()
    assert split.validation["open_time"].max() < split.test["open_time"].min()


def test_short_history_is_rejected():
    from .conftest import flat_bars

    with pytest.raises(ValueError):
        build_dataset(flat_bars([100.0] * 5), FeatureConfig(), LabelConfig())
