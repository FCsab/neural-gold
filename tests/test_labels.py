from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_neural.config import LabelConfig
from crypto_neural.labels import label_distribution, label_triple_barrier

from .conftest import flat_bars

# atr_period=1 means ATR == true range, so barriers are exactly close +- 2.0
# for flat_bars(..., wick=1.0) with k=1.
CFG = LabelConfig(k_up=1.0, k_down=1.0, max_holding_bars=3, atr_period=1)


def _perturb(high=None, low=None, open_=None, close=None) -> pd.DataFrame:
    frame = flat_bars([100.0] * 6, wick=1.0)
    for column, overrides in (("high", high), ("low", low), ("open", open_), ("close", close)):
        for index, value in (overrides or {}).items():
            frame.loc[index, column] = value
    return frame


def test_profit_target_first():
    bars = _perturb(high={1: 103.0})
    labels = label_triple_barrier(bars, CFG)
    row = labels.iloc[0]
    assert row["valid"]
    assert int(row["label"]) == 1
    assert int(row["exit_index"]) == 1
    assert float(row["exit_price"]) == pytest.approx(102.0)
    assert int(row["holding_bars"]) == 1
    assert float(row["realized_return"]) == pytest.approx(0.02)


def test_stop_first_after_gap_of_bars():
    bars = _perturb(low={2: 97.0})
    labels = label_triple_barrier(bars, CFG)
    row = labels.iloc[0]
    assert int(row["label"]) == -1
    assert int(row["exit_index"]) == 2
    assert float(row["exit_price"]) == pytest.approx(98.0)
    assert int(row["holding_bars"]) == 2
    assert float(row["realized_return"]) < 0


def test_timeout_exits_at_horizon_close():
    bars = _perturb()
    labels = label_triple_barrier(bars, CFG)
    row = labels.iloc[0]
    assert int(row["label"]) == 0
    assert int(row["exit_index"]) == CFG.max_holding_bars
    assert float(row["exit_price"]) == pytest.approx(100.0)
    assert int(row["holding_bars"]) == CFG.max_holding_bars
    assert float(row["realized_return"]) == pytest.approx(0.0)


def test_same_bar_ambiguity_is_pessimistic_by_default():
    bars = _perturb(high={1: 103.0}, low={1: 97.0})
    pessimistic = label_triple_barrier(bars, CFG)
    assert int(pessimistic.iloc[0]["label"]) == -1

    optimistic_cfg = LabelConfig(k_up=1.0, k_down=1.0, max_holding_bars=3, atr_period=1, ambiguity="up_priority")
    optimistic = label_triple_barrier(bars, optimistic_cfg)
    assert int(optimistic.iloc[0]["label"]) == 1


def test_gap_through_target_fills_at_open():
    bars = _perturb(high={1: 106.0}, low={1: 104.0}, open_={1: 105.0}, close={1: 105.0})
    labels = label_triple_barrier(bars, CFG)
    row = labels.iloc[0]
    assert int(row["label"]) == 1
    # Barrier is 102; the bar opened at 105, so the fill is the open.
    assert float(row["exit_price"]) == pytest.approx(105.0)


def test_gap_through_stop_fills_at_open():
    bars = _perturb(high={1: 96.0}, low={1: 94.0}, open_={1: 95.0}, close={1: 95.0})
    labels = label_triple_barrier(bars, CFG)
    row = labels.iloc[0]
    assert int(row["label"]) == -1
    # Barrier is 98; the bar opened at 95, so the fill is the open.
    assert float(row["exit_price"]) == pytest.approx(95.0)


def test_warmup_rows_are_invalid_not_zero():
    bars = flat_bars([100.0] * 30, wick=1.0)
    cfg = LabelConfig(k_up=1.0, k_down=1.0, max_holding_bars=3, atr_period=10)
    labels = label_triple_barrier(bars, cfg)
    assert labels["valid"].iloc[:9].sum() == 0
    assert labels["label"].iloc[:9].isna().all()
    assert labels["valid"].iloc[10]


def test_holding_never_exceeds_horizon():
    cfg = LabelConfig(k_up=0.1, k_down=0.1, max_holding_bars=5, atr_period=2)
    rng = np.random.default_rng(3)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    frame = flat_bars(list(closes), wick=1.0)
    labels = label_triple_barrier(frame, cfg)
    valid = labels[labels["valid"]]
    assert valid["holding_bars"].between(1, 5).all()
    assert valid["exit_index"].to_numpy().max() <= len(frame) - 1


def test_label_distribution_counts():
    bars = _perturb(high={1: 103.0})
    labels = label_triple_barrier(bars, CFG)
    counts = label_distribution(labels)
    assert counts["total"] == int(labels["valid"].sum())
    assert counts["long"] == 1


def test_empty_when_history_shorter_than_horizon():
    bars = flat_bars([100.0, 100.0], wick=1.0)
    labels = label_triple_barrier(bars, CFG)
    assert labels.empty
    assert list(labels.columns)[0] == "open_time"


# -- drawdown label (risk model) ------------------------------------------
def test_drawdown_label_uses_the_lowest_low_inside_the_horizon():
    from crypto_neural.labels import label_drawdown

    bars = flat_bars([100.0] * 8, wick=0.0)
    bars.loc[3, "low"] = 91.0  # a 9% intrabar fall at bar 3

    labels = label_drawdown(bars, drawdown=0.08, horizon_bars=3)

    # Bars 0, 1 and 2 have bar 3 inside their next three bars; bar 3 itself
    # looks at 4..6 and does not.
    assert labels["label"].iloc[:4].tolist() == [1, 1, 1, 0]
    assert labels["exit_index"].iloc[0] == 3


def test_drawdown_label_leaves_incomplete_rows_unlabelled():
    from crypto_neural.labels import label_drawdown

    labels = label_drawdown(flat_bars([100.0] * 6, wick=0.0), 0.05, 3)

    assert labels["valid"].tolist() == [True, True, True, False, False, False]
    assert labels["label"].iloc[3:].isna().all()  # never a fabricated 0


def test_drawdown_label_only_depends_on_its_own_window():
    """Truncating history after t+H must not change the label at t."""
    from crypto_neural.labels import label_drawdown

    from .conftest import synthetic_bars

    bars = synthetic_bars(n=600)
    full = label_drawdown(bars, 0.005, 24)
    for index in (0, 100, 400, 575):
        cut = label_drawdown(bars.iloc[: index + 25].reset_index(drop=True), 0.005, 24)
        assert cut["label"].iloc[index] == full["label"].iloc[index]
