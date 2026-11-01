from __future__ import annotations

import pytest

from gold_neural.backtest import BacktestConfig
from gold_neural.config import FeatureConfig, LabelConfig
from gold_neural.dataset import build_dataset
from gold_neural.training import TrainConfig
from gold_neural.walkforward import build_folds, evaluate_fold, summarise

from .conftest import synthetic_bars


def _frame(n: int = 3000):
    bars = synthetic_bars(n=n)
    dataset = build_dataset(bars, FeatureConfig(), LabelConfig(max_holding_bars=12))
    return dataset.frame.reset_index(drop=True), dataset.feature_columns


def test_folds_are_ordered_and_their_test_blocks_abut():
    frame, _ = _frame()
    folds = build_folds(frame, test_bars=200, val_bars=200, min_train_bars=600, window=8)

    assert len(folds) >= 3
    for previous, following in zip(folds, folds[1:]):
        assert following.test[0] == previous.test[1], "test blocks must not overlap or gap"
        assert following.index == previous.index + 1


def test_each_fold_has_train_before_validation_before_test():
    frame, _ = _frame()
    folds = build_folds(frame, test_bars=200, val_bars=200, min_train_bars=600, window=8)

    for fold in folds:
        assert fold.train[0] == 7  # window - 1: the first row with a full window
        assert fold.train[1] <= fold.validation[0]
        assert fold.validation[1] == fold.test[0]
        assert fold.test[1] - fold.test[0] == 200
        assert fold.train[1] - fold.train[0] >= 8
        assert fold.test[1] <= len(frame)


def test_summarise_reports_the_distribution_not_one_number():
    rows = [
        {
            "status": "ok",
            "return": 0.06,
            "buy_hold": 0.02,
            "trades": 10,
            "max_drawdown": 0.05,
            "balanced_accuracy": 0.40,
        },
        {
            "status": "ok",
            "return": -0.02,
            "buy_hold": 0.01,
            "trades": 12,
            "max_drawdown": 0.08,
            "balanced_accuracy": 0.36,
        },
        {"status": "failed"},
    ]

    summary = summarise(rows)

    assert summary["folds"] == 2
    assert summary["median_return"] == pytest.approx(0.02)
    assert summary["share_over_5pct"] == 0.5
    assert summary["share_beating_hold"] == 0.5
    assert summary["worst_return"] == pytest.approx(-0.02)


def test_evaluate_fold_trains_and_trades_its_block(tmp_path):
    frame, features = _frame()
    folds = build_folds(frame, test_bars=200, val_bars=200, min_train_bars=600, window=8)
    fold = folds[1]

    config = TrainConfig(
        window=8,
        epochs=2,
        batch_size=64,
        d_model=16,
        nhead=2,
        layers=1,
        dim_feedforward=32,
        patience=0,
        max_minutes=1.0,
        device="cpu",
        num_threads=1,
    )

    # An empty price lookup makes prepare_backtest_frame fall back to close,
    # which is fine for a structural test.
    row = evaluate_fold(
        frame, features, fold, config, tmp_path, BacktestConfig(), {}
    )

    assert row["status"] == "ok"
    assert row["fold"] == fold.index
    assert row["test_start_ms"] == fold.test_start_ms
    assert set(row) >= {"return", "buy_hold", "excess", "trades", "max_drawdown"}
