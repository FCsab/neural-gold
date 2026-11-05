from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_neural.backtest import BacktestConfig, run_backtest, verdict

BAR_MS = 300_000


def _frame(rows: int = 12, entry: float = 100.0, exit_price: float = 102.0, holding: int = 3):
    open_time = np.arange(rows, dtype=np.int64) * BAR_MS + 1_700_000_000_000
    return pd.DataFrame(
        {
            "open_time": open_time,
            "entry_price": np.full(rows, entry),
            "exit_time": open_time + holding * BAR_MS,
            "exit_price": np.full(rows, exit_price),
            "holding_bars": np.full(rows, holding),
            "label": np.ones(rows, dtype=np.int64),
        }
    )


def _probs(rows: int, long_prob: float = 0.7) -> np.ndarray:
    """argmax = 2 -> label +1 (long)."""
    rest = (1.0 - long_prob) / 2.0
    return np.tile(np.array([rest, rest, long_prob]), (rows, 1))


def test_trades_are_taken_and_compounded():
    frame = _frame()
    result = run_backtest(frame, _probs(len(frame)), BacktestConfig(fee_bps=0, slippage_bps=0))

    # 12 rows, 3-bar holds, no overlap: entries at 0, 4, 8
    assert len(result.trades) == 3
    assert result.metrics["trades"] == 3
    assert result.metrics["final_equity"] > result.metrics["initial_equity"]
    for trade in result.trades:
        assert trade.gross_return == pytest.approx(0.02)
        assert trade.net_return == pytest.approx(0.02)


def test_costs_reduce_the_return():
    frame = _frame()
    free = run_backtest(frame, _probs(len(frame)), BacktestConfig(fee_bps=0, slippage_bps=0))
    costly = run_backtest(frame, _probs(len(frame)), BacktestConfig(fee_bps=10, slippage_bps=2))

    assert costly.metrics["final_equity"] < free.metrics["final_equity"]
    # 10 bps fee + 2 bps slippage per side is ~24 bps of round-trip drag
    per_trade = costly.trades[0].net_return
    assert per_trade == pytest.approx(0.02 - 0.0024, abs=2e-4)


def test_no_signals_means_no_trades():
    frame = _frame()
    flat = np.tile(np.array([0.1, 0.8, 0.1]), (len(frame), 1))
    result = run_backtest(frame, flat, BacktestConfig())

    assert result.trades == []
    assert result.metrics["trades"] == 0
    assert result.metrics["final_equity"] == result.metrics["initial_equity"]
    assert "note" not in result.metrics  # the note only appears for an empty frame
    assert "No trades" in verdict(result.metrics)["text"]


def test_min_confidence_filters_weak_signals():
    frame = _frame()
    weak = _probs(len(frame), long_prob=0.5)
    taken = run_backtest(frame, weak, BacktestConfig(min_confidence=0.0))
    skipped = run_backtest(frame, weak, BacktestConfig(min_confidence=0.9))

    assert len(taken.trades) > 0
    assert skipped.trades == []


def test_trades_do_not_overlap():
    frame = _frame(rows=12, holding=5)
    result = run_backtest(frame, _probs(len(frame)), BacktestConfig())
    entry_indices = [trade.entry_time_ms for trade in result.trades]

    # with 5-bar holds over 12 rows: entry at row 0, next allowed at row 6
    assert len(result.trades) == 2
    assert entry_indices[1] - entry_indices[0] == 6 * BAR_MS


def test_buy_and_hold_is_the_same_window():
    frame = _frame(entry=100.0, exit_price=110.0)
    config = BacktestConfig(fee_bps=0, slippage_bps=0)
    result = run_backtest(frame, _probs(len(frame)), config)

    assert result.metrics["buy_hold_return"] == pytest.approx(0.10)


def test_drawdown_tracks_losses():
    open_time = np.arange(6, dtype=np.int64) * BAR_MS + 1_700_000_000_000
    frame = pd.DataFrame(
        {
            "open_time": open_time,
            "entry_price": np.full(6, 100.0),
            "exit_time": open_time + BAR_MS,
            "exit_price": np.array([90.0, 90.0, 90.0, 90.0, 90.0, 90.0]),
            "holding_bars": np.ones(6, dtype=np.int64),
            "label": np.full(6, -1, dtype=np.int64),
        }
    )
    result = run_backtest(frame, _probs(6), BacktestConfig(fee_bps=0, slippage_bps=0))

    assert result.metrics["total_return"] < 0
    assert result.metrics["max_drawdown"] > 0.05
    assert verdict(result.metrics)["tone"] == "bad"


def test_verdict_branches():
    assert verdict({"trades": 0})["tone"] == "warn"

    profitable_edge = {
        "trades": 10,
        "total_return": 0.05,
        "buy_hold_return": 0.01,
        "excess_return": 0.04,
        "pnl": 500.0,
        "days": 30.0,
    }
    assert verdict(profitable_edge)["tone"] == "ok"

    profitable_no_edge = dict(profitable_edge, buy_hold_return=0.10, excess_return=-0.05)
    assert "Profit, no edge" in verdict(profitable_no_edge)["text"]

    defensive = {
        "trades": 10,
        "total_return": -0.02,
        "buy_hold_return": -0.10,
        "excess_return": 0.08,
        "pnl": -200.0,
        "days": 30.0,
    }
    assert "Defensive" in verdict(defensive)["text"]


def test_config_validation():
    with pytest.raises(ValueError):
        BacktestConfig(initial_equity=0)
    with pytest.raises(ValueError):
        BacktestConfig(min_confidence=1.0)
    with pytest.raises(ValueError):
        BacktestConfig(position_fraction=2.0)


def test_to_dict_caps_payload():
    frame = _frame(rows=400, holding=1)
    result = run_backtest(frame, _probs(len(frame)), BacktestConfig())
    payload = result.to_dict(max_trades=5, max_points=10)

    assert len(payload["trades"]) <= 5
    assert len(payload["equity_curve"]) <= 11
    assert payload["trade_count"] == len(result.trades)


def test_cost_larger_than_target_is_flagged():
    # +2 bps of movement against 24 bps of round-trip cost
    frame = _frame(entry=100.0, exit_price=100.02)
    result = run_backtest(frame, _probs(len(frame)), BacktestConfig(fee_bps=10, slippage_bps=2))

    assert result.metrics["cost_exceeds_target"] is True
    assert result.metrics["avg_gross_win"] == pytest.approx(0.0002)
    assert result.metrics["round_trip_cost"] == pytest.approx(0.0024)
    assert "round trip" in verdict(result.metrics)["text"].lower()


def test_healthy_costs_are_not_flagged():
    frame = _frame(entry=100.0, exit_price=110.0)
    result = run_backtest(frame, _probs(len(frame)), BacktestConfig(fee_bps=10, slippage_bps=2))

    assert result.metrics["cost_exceeds_target"] is False
    assert result.metrics["avg_gross_win"] == pytest.approx(0.10)


def test_buy_hold_uses_the_block_close_not_a_future_label_exit():
    """A long holding time must not let the baseline borrow future returns."""
    frame = _frame(rows=4, entry=100.0, exit_price=130.0, holding=3)
    frame["close"] = [100.0, 101.0, 102.0, 105.0]

    result = run_backtest(
        frame, _probs(len(frame)), BacktestConfig(fee_bps=0, slippage_bps=0)
    )

    assert result.metrics["buy_hold_return"] == pytest.approx(0.05)
