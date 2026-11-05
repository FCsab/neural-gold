from __future__ import annotations

import numpy as np
import pytest

from crypto_neural.config import FeatureConfig, LabelConfig
from crypto_neural.leakage import (
    LeakageError,
    assert_no_leakage,
    check_dataset_invariants,
    check_prefix_invariance,
    mutual_information_bits,
    run_leakage_suite,
)


def test_suite_passes_on_clean_pipeline(bars):
    report = run_leakage_suite(bars, FeatureConfig(), LabelConfig(), samples=5)
    assert report.passed, report.format()
    names = [check.name for check in report.checks]
    assert "features_prefix_invariant" in names
    assert "labels_depend_on_future" in names
    assert "labels_ignore_far_future" in names
    assert "dataset_invariants" in names
    assert "shuffled_label_correlation" in names


def test_prefix_check_catches_a_leaked_feature(bars, monkeypatch):
    import crypto_neural.leakage as leakage

    real_builder = leakage.build_features

    def leaky_builder(frame, cfg):
        out = real_builder(frame, cfg)
        out["future_ret"] = frame["close"].shift(-1) / frame["close"] - 1.0
        return out

    monkeypatch.setattr(leakage, "build_features", leaky_builder)
    check = check_prefix_invariance(bars, FeatureConfig(), samples=3)
    assert not check.passed
    assert "future_ret" in check.detail


def test_assert_no_leakage_raises_when_a_check_fails(bars, monkeypatch):
    import crypto_neural.leakage as leakage

    def broken(_bars, _features_cfg, samples=10):
        return leakage.Check(name="features_prefix_invariant", passed=False, detail="boom")

    monkeypatch.setattr(leakage, "check_prefix_invariance", broken)
    with pytest.raises(LeakageError, match="features_prefix_invariant"):
        assert_no_leakage(bars, FeatureConfig(), LabelConfig(), samples=1)


def test_shuffled_label_check_is_a_harness_test(bars):
    from crypto_neural.dataset import build_dataset
    from crypto_neural.leakage import check_shuffled_labels

    ds = build_dataset(bars, FeatureConfig(), LabelConfig())
    check = check_shuffled_labels(ds)
    assert check.passed
    assert check.metrics["max_abs_correlation"] <= check.metrics["threshold"]


def test_dataset_invariants_detect_misalignment(bars):
    from crypto_neural.dataset import build_dataset

    ds = build_dataset(bars, FeatureConfig(), LabelConfig())
    assert check_dataset_invariants(ds, LabelConfig().max_holding_bars).passed

    broken = ds.frame.copy()
    broken.loc[broken.index[10], "exit_time"] = broken.loc[broken.index[10], "open_time"]
    from crypto_neural.dataset import Dataset

    check = check_dataset_invariants(
        Dataset(broken, ds.feature_columns), LabelConfig().max_holding_bars
    )
    assert not check.passed
    assert "exit_time" in check.detail


def test_mutual_information_behaviour():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    dependent = np.where(x > 0, 1.0, 0.0)
    independent = rng.permutation(dependent)

    assert mutual_information_bits(x, dependent) > 0.5
    assert mutual_information_bits(x, independent) < 0.02
