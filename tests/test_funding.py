from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_neural.config import FeatureConfig
from crypto_neural.funding import (
    FUNDING_FEATURES,
    funding_builder,
    funding_features,
    settlement_features,
)
from crypto_neural.leakage import LeakageError
from crypto_neural.risk import ExternalFeatures, check_extra_prefix, run_risk_walkforward

from .conftest import BAR_MS, small_risk_config, synthetic_bars


def settlements_for(bars: pd.DataFrame, every: int = 3, seed: int = 0) -> pd.DataFrame:
    """A settlement every ``every`` bars, landing exactly on bar opens (as
    Binance's 00/08/16 UTC settlements land on 4h bar opens)."""
    rng = np.random.default_rng(seed)
    times = bars["open_time"].to_numpy()[::every]
    return pd.DataFrame(
        {"funding_time": times.astype("int64"), "funding_rate": rng.normal(1e-4, 5e-5, len(times))}
    )


def test_a_settlement_at_the_bar_open_is_not_visible_to_that_bar():
    open_time = np.array([1_000, 2_000, 3_000], dtype=np.int64)
    funding = pd.DataFrame({"funding_time": [1_000, 2_000], "funding_rate": [0.1, 0.2]})
    out = funding_features(open_time, funding)
    assert np.isnan(out["fund_last"].iloc[0])  # 1_000 settles as the bar opens
    assert out["fund_last"].iloc[1] == 0.1
    assert out["fund_last"].iloc[2] == 0.2


def test_settlement_windows_need_full_history():
    funding = pd.DataFrame({"funding_time": np.arange(100, dtype=np.int64),
                            "funding_rate": np.linspace(0, 1, 100)})
    stats = settlement_features(funding)
    assert stats["fund_mean_9"].isna().sum() == 8
    assert stats["fund_mean_90"].isna().sum() == 89
    assert list(stats.columns[1:]) == FUNDING_FEATURES


def test_the_builder_passes_the_prefix_check():
    bars = synthetic_bars(800)
    passed, detail = check_extra_prefix(bars, funding_builder(settlements_for(bars)), samples=12)
    assert passed, detail


def test_a_builder_that_peeks_at_the_next_settlement_is_caught():
    bars = synthetic_bars(800)
    funding = settlements_for(bars)

    def peek(b: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        # Same columns, but matched to the settlement at or *after* the open.
        stats = settlement_features(table).rename(columns={"funding_time": "key"})
        keys = pd.DataFrame({"key": b["open_time"].to_numpy(dtype=np.int64)})
        merged = pd.merge_asof(keys, stats, on="key", direction="forward")
        return merged[FUNDING_FEATURES].set_axis(b.index)

    leaky = ExternalFeatures("funding", funding, "funding_time", peek)
    passed, detail = check_extra_prefix(bars, leaky, samples=12)
    assert not passed and "funding row" in detail


def test_walkforward_with_funding_records_the_extra_columns():
    pytest.importorskip("lightgbm")
    bars = synthetic_bars(3000)
    funding = settlements_for(bars)
    report = run_risk_walkforward(
        bars, small_risk_config(), FeatureConfig(), bars_per_year=288 * 365,
        check_samples=3, extra_features=funding_builder(funding),
    )
    assert report.params["extra_features"] == FUNDING_FEATURES
    assert report.features[-4:] == FUNDING_FEATURES
    assert "later bars and records hidden" in report.feature_check["detail"]
    assert report.today["for_bar_ms"] == int(bars["open_time"].iloc[-1]) + BAR_MS

    plain = run_risk_walkforward(
        bars, small_risk_config(), FeatureConfig(), bars_per_year=288 * 365, check_samples=3,
    )
    assert "extra_features" not in plain.params


def test_walkforward_refuses_leaky_extra_features():
    pytest.importorskip("lightgbm")
    bars = synthetic_bars(3000)

    def future_close(b: pd.DataFrame, _table: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"next": b["close"].shift(-1)}, index=b.index)

    nothing = pd.DataFrame({"t": pd.Series(dtype="int64")})
    with pytest.raises(LeakageError):
        run_risk_walkforward(
            bars, small_risk_config(), FeatureConfig(), bars_per_year=288 * 365,
            check_samples=3,
            extra_features=ExternalFeatures("next close", nothing, "t", future_close),
        )


def test_constant_funding_has_zero_deviation_not_a_gap():
    funding = pd.DataFrame({"funding_time": np.arange(120, dtype=np.int64),
                            "funding_rate": np.full(120, 1e-4)})
    stats = settlement_features(funding)
    assert stats["fund_z"].iloc[89:].eq(0.0).all()
    assert stats["fund_z"].iloc[:89].isna().all()


def _report(auc: float, block_aucs: list, rows: int = 100) -> dict:
    return {
        "prediction": {"auc": auc, "rows": rows},
        "folds": [{"start_ms": i * 10, "end_ms": i * 10 + 9, "auc": a, "falls": 1}
                  for i, a in enumerate(block_aucs)],
        "strategy": {"cagr": 0.1, "max_drawdown": 0.3, "sharpe": 0.5,
                     "exposure": 0.9, "placebo_p": 0.5},
        "hold": {"cagr": 0.2, "max_drawdown": 0.5, "sharpe": 0.6},
        "period": {},
    }


def test_the_decision_rule_needs_both_the_auc_gain_and_the_block_share():
    from crypto_neural.funding import compare_arms

    base = _report(0.650, [0.6, 0.6, 0.6, 0.6, 0.6])
    # Gain large enough, but wins only 2 of 5 blocks.
    assert not compare_arms(base, _report(0.670, [0.7, 0.7, 0.5, 0.5, 0.5]))["adds_information"]
    # Wins every block, but the pooled gain is under 0.010.
    assert not compare_arms(base, _report(0.655, [0.61] * 5))["adds_information"]
    # Both conditions met; a block without both outcomes is not counted.
    result = compare_arms(base, _report(0.661, [0.7, 0.7, 0.7, 0.5, None]))
    assert result["adds_information"]
    assert result["blocks_compared"] == 4 and result["blocks_won"] == 3


def test_arms_on_different_rows_or_blocks_are_refused():
    from crypto_neural.funding import compare_arms

    base = _report(0.65, [0.6, 0.6])
    with pytest.raises(ValueError):
        compare_arms(base, _report(0.7, [0.7, 0.7], rows=99))
    other = _report(0.7, [0.7, 0.7])
    other["folds"][1]["start_ms"] += 1
    with pytest.raises(ValueError):
        compare_arms(base, other)


def test_experiment_bars_start_after_the_long_window_of_settlements():
    from crypto_neural.funding import LONG, experiment_bars

    bars = synthetic_bars(600)
    funding = settlements_for(bars, every=2)
    trimmed = experiment_bars(bars, funding)
    first = int(trimmed["open_time"].iloc[0])
    assert (funding["funding_time"] < first).sum() == LONG
    assert funding_features(trimmed["open_time"].to_numpy(), funding).notna().all().all()
