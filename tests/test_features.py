from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold_neural.config import FeatureConfig
from gold_neural.features import build_features, feature_names, required_history


def test_column_names_and_order(bars):
    cfg = FeatureConfig()
    frame = build_features(bars, cfg)
    assert list(frame.columns) == feature_names(cfg)
    assert len(frame) == len(bars)


def test_no_infinities_and_all_finite_after_warmup(bars):
    cfg = FeatureConfig()
    frame = build_features(bars, cfg)
    assert not np.isinf(frame.to_numpy(dtype=float)).any()

    start = required_history(cfg) + 5
    tail = frame.iloc[start:]
    assert tail.notna().all().all(), tail.isna().sum()[lambda s: s > 0]


def test_features_are_prefix_invariant(bars):
    cfg = FeatureConfig()
    full = build_features(bars, cfg)
    for index in (300, 1000, 2000, 2999):
        prefix = build_features(bars.iloc[: index + 1].reset_index(drop=True), cfg)
        pd.testing.assert_series_equal(
            full.iloc[index], prefix.iloc[-1], check_names=False
        )


def test_feature_lag_shifts_values(bars):
    lagged_cfg = FeatureConfig(feature_lag=1)
    unlagged_cfg = FeatureConfig(feature_lag=0)
    lagged = build_features(bars, lagged_cfg)
    unlagged = build_features(bars, unlagged_cfg)
    pd.testing.assert_series_equal(
        lagged.iloc[100], unlagged.iloc[99], check_names=False
    )


def test_centered_windows_would_be_caught(bars):
    """Guard the guard: a deliberately centred feature must break invariance."""
    cfg = FeatureConfig()
    full = build_features(bars, cfg)
    centred = bars["close"].rolling(5, center=True).mean()
    assert not np.allclose(
        full["ret_1"].iloc[: len(centred)].fillna(0).to_numpy(),
        centred.fillna(0).to_numpy()[: len(full)],
        atol=0.5,
    )


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        build_features(pd.DataFrame({"close": [1.0, 2.0]}), FeatureConfig())


def test_bad_macd_config_rejected(bars):
    with pytest.raises(ValueError, match="macd"):
        build_features(bars, FeatureConfig(macd=(12, 26)))
