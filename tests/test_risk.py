from __future__ import annotations

import numpy as np
import pytest

from crypto_neural.config import ConfigError, FeatureConfig, RiskConfig
from crypto_neural.features import build_features, feature_names
from crypto_neural.leakage import LeakageError
from crypto_neural.risk import (
    HOLD,
    net_returns,
    next_bar_features,
    performance,
    placebo_p_value,
    risk_folds,
    risk_positions,
    run_risk_walkforward,
    trend_reading,
)

from .conftest import small_risk_config, synthetic_bars

BARS_PER_YEAR = 288 * 365


def test_positions_exit_above_the_line_and_wait_before_reentry():
    risk = np.array([0.1, 0.9, 0.1, 0.1, 0.1, 0.1])
    line = np.full(6, 0.5)

    positions = risk_positions(risk, line, line * 0.9, min_flat_bars=3)

    # Out on the warning bar, then three flat bars before coming back.
    assert positions.tolist() == [1, 0, 0, 0, 1, 1]


def test_positions_do_not_reenter_between_the_lines():
    risk = np.array([0.6, 0.48, 0.48, 0.3])
    line = np.full(4, 0.5)

    positions = risk_positions(risk, line, line * 0.9, min_flat_bars=0)

    # 0.48 is under the exit line but above the re-entry line (0.45).
    assert positions.tolist() == [0, 0, 0, 1]


def test_every_switch_pays_the_cost():
    positions = np.array([1.0, 0.0, 1.0])
    flat_market = np.zeros(3)

    returns = net_returns(positions, flat_market, cost=0.001)

    # Buy, sell, buy again: three charges, no market move.
    assert returns.tolist() == pytest.approx([-0.001, -0.001, -0.001])


def test_performance_reports_drawdown_from_the_starting_capital():
    returns = np.array([-0.1, 0.0, 0.05])
    stats = performance(returns, np.ones(3), bars_per_year=3)

    assert stats["max_drawdown"] == pytest.approx(0.1)
    assert stats["total_return"] == pytest.approx(0.9 * 1.05 - 1, abs=1e-4)


def test_placebo_is_one_for_plain_holding():
    assert placebo_p_value(np.ones(50), np.zeros(50), 0.001, 1.0, 50) == 1.0


def test_folds_purge_labels_that_reach_into_the_test_block():
    rows = np.arange(100)
    labelled = np.ones(100, dtype=bool)
    exit_index = rows + 10  # each label looks 10 bars ahead

    folds = risk_folds(rows, labelled, exit_index, min_train_bars=40, test_bars=20)

    assert len(folds) == 3
    for train, test in folds:
        assert test[0] == train[-1] + 11  # the 10 overlapping rows are gone
        assert (exit_index[train] < test[0]).all()
    # Test blocks abut and do not overlap.
    assert [test[0] for _, test in folds] == [40, 60, 80]


def test_walkforward_runs_end_to_end_with_costs_and_baselines():
    pytest.importorskip("lightgbm")
    report = run_risk_walkforward(
        synthetic_bars(3000), small_risk_config(), FeatureConfig(),
        bars_per_year=BARS_PER_YEAR, check_samples=3,
    )

    assert report.period["folds"] >= 3
    assert report.status in {"candidate", "draft"}
    # The first block has no history to choose a rule from: it holds.
    assert report.folds[0]["rule"] == "hold"
    assert {row["rule"] for row in report.candidates} >= {"hold", "q0.8", "q0.9"}
    assert report.hold["exposure"] == 1.0
    assert report.feature_check["passed"] is True
    assert report.today["action"] in {"long", "flat", "keep"}
    assert len(report.equity["time"]) == len(report.equity["strategy"])
    # The trend rule gets the same luck test as the model.
    assert 0.0 <= report.trend["placebo_p"] <= 1.0
    # Today's reading is for the bar after the newest stored one.
    bars = synthetic_bars(3000)
    step = int(bars["open_time"].iloc[1] - bars["open_time"].iloc[0])
    assert report.today["for_bar_ms"] == int(bars["open_time"].iloc[-1]) + step
    assert report.today["trend"]["action"] in {"long", "flat"}


def test_earlier_blocks_do_not_change_when_later_history_is_added():
    """The walk-forward analogue of prefix invariance.

    A block's model, its exit lines and its chosen rule may only depend on
    data before it, so cutting history off after block k must leave blocks
    0..k exactly as they were.
    """
    pytest.importorskip("lightgbm")
    bars = synthetic_bars(3000)
    cfg = small_risk_config()
    full = run_risk_walkforward(bars, cfg, FeatureConfig(), bars_per_year=BARS_PER_YEAR, check_samples=2)
    short = run_risk_walkforward(
        bars.iloc[:2700], cfg, FeatureConfig(), bars_per_year=BARS_PER_YEAR, check_samples=2
    )

    assert 3 <= len(short.folds) < len(full.folds)
    for before, after in zip(short.folds, full.folds):
        assert before == after


def test_config_rejects_nonsense():
    with pytest.raises(ConfigError):
        RiskConfig(drawdown=8.0)  # a typo for 0.08 must fail loudly
    with pytest.raises(ConfigError):
        RiskConfig(exit_quantiles=(0.9, 1.5))
    with pytest.raises(ConfigError):
        RiskConfig(sma_windows=(1,))
    assert HOLD == 1.0


def test_history_starts_after_the_last_long_gap():
    from crypto_neural.risk import contiguous_tail

    bars = synthetic_bars(100)
    bars.loc[60:, "open_time"] += 10 * 86_400_000  # a ten-day delisting

    tail, skipped = contiguous_tail(bars, max_gap_ms=86_400_000)

    assert len(tail) == 40
    assert tail["open_time"].iloc[0] == bars["open_time"].iloc[60]
    assert skipped["bars_dropped"] == 60 and skipped["days"] >= 10
    assert contiguous_tail(synthetic_bars(50), 86_400_000)[1] is None


@pytest.mark.parametrize("model", ["mlp", "ensemble"])
def test_neural_and_ensemble_models_run_the_same_walkforward(model):
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    cfg = small_risk_config(model=model, mlp_epochs=3, mlp_hidden=(8,))
    report = run_risk_walkforward(
        synthetic_bars(3000), cfg, FeatureConfig(), bars_per_year=BARS_PER_YEAR, check_samples=2
    )

    assert report.params["model"] == model
    assert report.period["folds"] >= 3
    assert 0.0 <= report.today["risk"] <= 1.0


def test_mlp_is_deterministic_for_a_seed():
    pytest.importorskip("torch")
    from crypto_neural.risk import _fit_mlp

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5))
    y = (x[:, 0] + rng.normal(scale=0.5, size=400) > 1).astype(float)
    cfg = small_risk_config(model="mlp", mlp_epochs=5, horizon_bars=10)

    first = _fit_mlp(x, y, [x[:20]], cfg)[0]
    second = _fit_mlp(x, y, [x[:20]], cfg)[0]

    np.testing.assert_allclose(first, second)
    assert ((first > 0) & (first < 1)).all()


def test_config_rejects_an_unknown_model():
    with pytest.raises(ConfigError):
        RiskConfig(model="xgboost")


def test_next_bar_row_equals_the_row_built_once_that_bar_exists():
    """The reading made before a bar opens must be the row the walk-forward
    would compute for that bar after it closed: same inputs, nothing unseen."""
    bars = synthetic_bars(800)
    cfg = FeatureConfig()
    names = feature_names(cfg)
    for cut in (500, 650, 799):
        row, open_ms = next_bar_features(bars.iloc[:cut], cfg)
        full = build_features(bars.iloc[: cut + 1], cfg)[names].to_numpy()[-1]
        assert open_ms == int(bars["open_time"].iloc[cut])
        np.testing.assert_array_equal(row, full)


def test_next_bar_reading_refuses_a_lag_of_zero():
    with pytest.raises(LeakageError):
        next_bar_features(synthetic_bars(300), FeatureConfig(feature_lag=0))


def test_trend_reading_uses_the_last_close_against_its_average():
    import pandas as pd

    rising = pd.Series(np.linspace(100.0, 200.0, 50))
    falling = pd.Series(np.linspace(200.0, 100.0, 50))
    up = trend_reading(rising, 20, 0, bars_per_day=1)
    down = trend_reading(falling, 20, 0, bars_per_day=1)
    assert up["action"] == "long" and up["distance"] > 0
    assert down["action"] == "flat" and down["distance"] < 0
    assert "20-day average" in up["text"]
    assert trend_reading(rising.iloc[:10], 20, 0, bars_per_day=1)["action"] == "long"
