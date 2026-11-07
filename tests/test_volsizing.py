from __future__ import annotations

import numpy as np
import pytest

from crypto_neural.config import FeatureConfig
from crypto_neural.volsizing import (
    decide,
    ewma_forecast,
    run_vol_sizing,
    size_positions,
    vol_target,
)

from .conftest import small_risk_config, synthetic_bars

BARS_PER_YEAR = 288 * 365


def test_target_is_the_volatility_of_the_next_six_open_to_open_returns():
    bars = synthetic_bars(100)
    y, exit_index = vol_target(bars, 6)
    opens = bars["open"].to_numpy()
    r = np.log(opens[11:17] / opens[10:16])
    assert y[10] == pytest.approx(0.5 * np.log(np.sum(r * r)))
    assert exit_index[10] == 16
    # The last six rows need an open that does not exist yet.
    assert np.isnan(y[-6:]).all() and np.isfinite(y[-7])


def test_target_depends_on_opens_up_to_its_exit_and_no_further():
    bars = synthetic_bars(100)
    y, _ = vol_target(bars, 6)
    later = bars.copy()
    later.loc[17:, "open"] *= 1.5  # after row 10's exit (bar 16)
    assert vol_target(later, 6)[0][10] == y[10]
    at_exit = bars.copy()
    at_exit.loc[16, "open"] *= 1.01
    assert vol_target(at_exit, 6)[0][10] != y[10]


def test_ewma_forecast_is_prefix_invariant_and_ignores_the_current_bar():
    bars = synthetic_bars(400)
    full = ewma_forecast(bars, 6, 30)
    for i in (40, 150, 399):
        assert ewma_forecast(bars.iloc[: i + 1], 6, 30)[i] == pytest.approx(full[i], rel=1e-12)
    moved = bars.copy()
    moved.loc[200, "close"] *= 1.2
    assert ewma_forecast(moved, 6, 30)[200] == full[200]
    assert np.isnan(full[:30]).all()


def test_positions_move_only_when_the_weight_moves_by_the_band():
    raw = np.array([1.0, 0.95, 0.85, 0.8, 0.5, 0.55, 1.0])
    assert size_positions(raw, 0.1).tolist() == [1.0, 1.0, 0.85, 0.85, 0.5, 0.5, 1.0]


def test_decision_rule_as_registered():
    hold = {"sharpe": 0.8, "max_drawdown": 0.7}
    good = {"sharpe": 1.0, "max_drawdown": 0.5, "placebo_p": 0.01}
    better_forecast = {"mse_model": 0.1, "mse_ewma": 0.2, "blocks": 10, "blocks_model_better": 6}
    worse_forecast = {**better_forecast, "mse_model": 0.3}
    ewma = {"sharpe": 0.9, "max_drawdown": 0.55, "placebo_p": 0.02}

    criteria, outcome, _ = decide(better_forecast, hold, ewma, good)
    assert outcome == "works" and all(criteria.values())
    # The formula does the job and the model forecasts no better.
    _, outcome, _ = decide(worse_forecast, hold, ewma, good)
    assert outcome == "formula only"
    # 5 of 10 blocks is under the registered 60%.
    _, outcome, text = decide({**better_forecast, "blocks_model_better": 5}, hold,
                              {**ewma, "placebo_p": 0.5}, good)
    assert outcome == "does not work" and "model_forecast_beats_ewma" in text
    _, outcome, text = decide(better_forecast, hold, ewma, {**good, "placebo_p": 0.06})
    assert outcome == "does not work" and "timing_beats_chance" in text


def test_walkforward_runs_end_to_end():
    pytest.importorskip("lightgbm")
    cfg = small_risk_config(seeds=1)
    report = run_vol_sizing(
        synthetic_bars(3000), cfg, FeatureConfig(), bars_per_year=BARS_PER_YEAR,
        check_samples=3,
    )
    assert report.period["folds"] >= 3
    assert report.hold["exposure"] == 1.0
    assert 0.0 < report.model_sized["exposure"] <= 1.0
    assert 0.0 < report.ewma_sized["exposure"] <= 1.0
    assert report.outcome in {"works", "formula only", "does not work"}
    assert set(report.criteria) == {
        "model_forecast_beats_ewma", "sharpe_above_hold_and_ewma",
        "drop_below_hold", "timing_beats_chance",
    }
    assert len(report.equity["time"]) == len(report.equity["model"])
    assert "volatility sizing" in report.format()
    assert report.to_json()


def test_cli_volsize_writes_its_report_and_leaves_the_predictions_log_alone(tmp_path):
    pytest.importorskip("lightgbm")
    import argparse
    import json
    from dataclasses import replace

    from crypto_neural.cli import cmd_volsize
    from crypto_neural.config import AppConfig, DataConfig
    from crypto_neural.data.store import store_from_config

    config = AppConfig(data=DataConfig(root=str(tmp_path / "data")),
                       risk=small_risk_config(seeds=1))
    local = replace(config, exchange=replace(config.exchange, symbol="BTCUSDC", interval="5m"))
    store = store_from_config(local)
    store.write_bars(synthetic_bars(3000))
    store.close()

    assert cmd_volsize(argparse.Namespace(), config) == 0
    report = json.loads((tmp_path / "data" / "reports" / "volsizing.json").read_text())
    assert report["outcome"] in {"works", "formula only", "does not work"}
    assert not (tmp_path / "data" / "predictions.sqlite").exists()
