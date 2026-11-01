"""Backtest: what would this model have made?

Classification metrics say whether the model is better than guessing. They do
not say whether it makes money. This module turns predictions into trades and
trades into an equity curve, with the costs that make the difference:

* **Fees** on both sides (10 bps each way by default: Binance spot, no BNB
  discount).
* **Slippage** on both sides (2 bps default). You never fill at the mid.
* **Entry at the next bar's open**, not at the close of the signal bar. The
  label is anchored on the signal bar's close, but you cannot trade at a price
  that has already happened.
* **Exit at the labelled exit price**: the barrier that was hit, or the timeout
  close. That price is already gap-adjusted by the labeller.

The result is compared against simply holding PAXG over the same window,
because "+3%" means nothing without that number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DAY_MS = 86_400_000


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 10_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    # 0.0 = take every long signal. The plan's live default is 0.55.
    min_confidence: float = 0.0
    position_fraction: float = 1.0
    annualisation_days: float = 365.0

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not 0.0 <= self.min_confidence < 1.0:
            raise ValueError("min_confidence must be within [0, 1)")
        if not 0.0 < self.position_fraction <= 1.0:
            raise ValueError("position_fraction must be within (0, 1]")


@dataclass
class Trade:
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    pnl: float
    equity_after: float
    holding_bars: int
    outcome: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_ms": self.entry_time_ms,
            "exit_ms": self.exit_time_ms,
            "entry": round(self.entry_price, 4),
            "exit": round(self.exit_price, 4),
            "gross_pct": round(self.gross_return * 100, 4),
            "net_pct": round(self.net_return * 100, 4),
            "pnl": round(self.pnl, 2),
            "equity": round(self.equity_after, 2),
            "bars": self.holding_bars,
            "outcome": self.outcome,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[dict[str, float]]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, max_trades: int = 50, max_points: int = 1200) -> dict[str, Any]:
        curve = self.equity_curve
        if len(curve) > max_points:
            step = math.ceil(len(curve) / max_points)
            curve = curve[::step] + [curve[-1]]
        recent = self.trades[-max_trades:]
        return {
            "metrics": self.metrics,
            "equity_curve": curve,
            "trades": [trade.to_dict() for trade in recent],
            "trade_count": len(self.trades),
        }


def _daily_sharpe(
    curve: list[dict[str, float]],
    start_ms: int,
    end_ms: int,
    initial: float,
    annualisation_days: float,
) -> float | None:
    if not curve:
        return None
    by_day: dict[int, float] = {}
    for point in curve:
        by_day[int(point["time"] * 1000) // DAY_MS] = float(point["equity"])

    first = start_ms // DAY_MS
    last = end_ms // DAY_MS
    if last - first < 5:
        return None

    series: list[float] = []
    value = initial
    for day in range(first, last + 1):
        value = by_day.get(day, value)
        series.append(value)

    returns = np.diff(series) / np.asarray(series[:-1])
    if len(returns) < 2:
        return None
    deviation = float(returns.std(ddof=1))
    if deviation == 0.0:
        return None
    return float(returns.mean() / deviation * np.sqrt(annualisation_days))


def run_backtest(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    config: BacktestConfig,
) -> BacktestResult:
    """Simulate long-only trades from model probabilities.

    ``probabilities`` has shape (rows, 3) ordered (-1, 0, +1) and must align
    row-for-row with ``frame``. Expected frame columns: ``open_time``,
    ``entry_price`` (next bar's open), ``exit_time``, ``exit_price``,
    ``holding_bars``, ``label``.
    """
    if len(frame) != len(probabilities):
        raise ValueError("frame and probabilities must have the same length")
    if len(frame) == 0:
        return BacktestResult(
            trades=[],
            equity_curve=[],
            metrics={"trades": 0, "note": "no rows to simulate"},
        )

    required = {"open_time", "entry_price", "exit_time", "exit_price", "holding_bars", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")

    predictions = probabilities.argmax(axis=1) - 1  # 0,1,2 -> -1,0,+1
    confidence = probabilities.max(axis=1)

    fee = config.fee_bps / 10_000.0
    slip = config.slippage_bps / 10_000.0

    equity = float(config.initial_equity)
    peak = equity
    max_drawdown = 0.0

    trades: list[Trade] = []
    curve: list[dict[str, float]] = []

    open_time = frame["open_time"].to_numpy(dtype=np.int64)
    entry_price = frame["entry_price"].to_numpy(dtype=np.float64)
    exit_time = frame["exit_time"].to_numpy(dtype=np.int64)
    exit_price = frame["exit_price"].to_numpy(dtype=np.float64)
    holding = frame["holding_bars"].to_numpy(dtype=np.int64)
    labels = frame["label"].to_numpy(dtype=np.int64)

    start_ms = int(open_time[0])
    end_ms = int(open_time[-1])

    i = 0
    rows = len(frame)
    while i < rows:
        entry_ok = np.isfinite(entry_price[i]) and entry_price[i] > 0
        exit_ok = np.isfinite(exit_price[i]) and exit_price[i] > 0
        if predictions[i] != 1 or confidence[i] < config.min_confidence or not (entry_ok and exit_ok):
            i += 1
            continue

        # Costs: the buyer pays up and the seller receives less.
        entry_fill = entry_price[i] * (1.0 + slip + fee)
        exit_fill = exit_price[i] * (1.0 - slip - fee)
        if entry_fill <= 0:
            i += 1
            continue

        gross = exit_price[i] / entry_price[i] - 1.0
        net = exit_fill / entry_fill - 1.0

        previous = equity
        equity *= 1.0 + config.position_fraction * net
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)

        trades.append(
            Trade(
                entry_time_ms=int(open_time[i]),
                exit_time_ms=int(exit_time[i]),
                entry_price=float(entry_fill),
                exit_price=float(exit_fill),
                gross_return=float(gross),
                net_return=float(net),
                pnl=float(equity - previous),
                equity_after=float(equity),
                holding_bars=int(holding[i]),
                outcome=int(labels[i]),
                confidence=float(confidence[i]),
            )
        )
        curve.append({"time": int(exit_time[i]) / 1000.0, "equity": round(float(equity), 2)})

        # Cannot open a new position before this one closes.
        i += max(1, int(holding[i]) + 1)

    # Buy-and-hold over the same window, with one round trip of costs.
    first_entry = entry_price[0] if np.isfinite(entry_price[0]) else exit_price[0]
    last_exit = exit_price[-1] if np.isfinite(exit_price[-1]) else first_entry
    buy_hold = 0.0
    if first_entry and last_exit and first_entry > 0:
        buy_hold = (last_exit * (1 - slip - fee)) / (first_entry * (1 + slip + fee)) - 1.0

    net_returns = np.array([trade.net_return for trade in trades], dtype=np.float64)
    gross_returns = np.array([trade.gross_return for trade in trades], dtype=np.float64)
    winning_gross = np.array(
        [trade.gross_return for trade in trades if trade.outcome == 1], dtype=np.float64
    )
    round_trip_cost = 2.0 * (config.fee_bps + config.slippage_bps) / 10_000.0
    avg_gross_win = float(winning_gross.mean()) if winning_gross.size else None
    # If even the trades that went the right way earn less than the round trip
    # costs, no model can save this configuration.
    cost_exceeds_target = avg_gross_win is not None and avg_gross_win < round_trip_cost

    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns < 0]
    profit_factor = (
        float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else None
    )

    metrics: dict[str, Any] = {
        "trades": len(trades),
        "initial_equity": round(config.initial_equity, 2),
        "final_equity": round(equity, 2),
        "total_return": round(equity / config.initial_equity - 1.0, 6),
        "pnl": round(equity - config.initial_equity, 2),
        "buy_hold_return": round(float(buy_hold), 6),
        "excess_return": round(float(equity / config.initial_equity - 1.0 - buy_hold), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "win_rate": round(float((net_returns > 0).mean()), 4) if net_returns.size else None,
        "avg_trade": round(float(net_returns.mean()), 6) if net_returns.size else None,
        "profit_factor": None if profit_factor is None else round(profit_factor, 3),
        "avg_gross_trade": round(float(gross_returns.mean()), 6) if gross_returns.size else None,
        "avg_gross_win": None if avg_gross_win is None else round(avg_gross_win, 6),
        "round_trip_cost": round(round_trip_cost, 6),
        "cost_exceeds_target": bool(cost_exceeds_target),
        "fees_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "min_confidence": config.min_confidence,
        "exposure": round(
            float(min(1.0, sum(trade.holding_bars for trade in trades) / max(1, rows))), 4
        ),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "days": round((end_ms - start_ms) / DAY_MS, 1),
        "sharpe": _daily_sharpe(
            curve, start_ms, end_ms, config.initial_equity, config.annualisation_days
        ),
        "costs_applied": True,
    }
    if metrics["sharpe"] is not None:
        metrics["sharpe"] = round(float(metrics["sharpe"]), 3)

    return BacktestResult(trades=trades, equity_curve=curve, metrics=metrics)


def verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Turn the numbers into one sentence a human can act on."""
    trades = metrics.get("trades", 0)
    if not trades:
        return {
            "tone": "warn",
            "text": (
                "No trades were taken on this slice. The model never produced a "
                "long signal above the confidence threshold."
            ),
        }

    if metrics.get("cost_exceeds_target"):
        win_bps = (metrics.get("avg_gross_win") or 0.0) * 10_000
        cost_bps = (metrics.get("round_trip_cost") or 0.0) * 10_000
        return {
            "tone": "bad",
            "text": (
                f"Even the trades that went the right way captured only {win_bps:.0f} bps on "
                f"average, while a round trip costs {cost_bps:.0f} bps in fees and slippage. "
                "No model can be profitable at these label settings. Widen the profit target, "
                "the stop, or the holding time until the target is several times the cost — "
                "then retrain."
            ),
        }

    total = metrics.get("total_return") or 0.0
    hold = metrics.get("buy_hold_return") or 0.0
    excess = metrics.get("excess_return") or 0.0
    pnl = metrics.get("pnl") or 0.0
    sign = "+" if pnl >= 0 else ""

    if total > 0 and excess > 0:
        return {
            "tone": "ok",
            "text": (
                f"Made {sign}{pnl:,.2f} ({total * 100:+.1f}%) over {metrics['days']:.0f} days, "
                f"beating simply holding PAXG by {excess * 100:+.1f} points. "
                "One slice, one set of costs — this is evidence, not a promise."
            ),
        }
    if total > 0:
        return {
            "tone": "",
            "text": (
                f"Made {sign}{pnl:,.2f} ({total * 100:+.1f}%), but holding PAXG would have "
                f"returned {hold * 100:+.1f}% over the same window. Profit, no edge."
            ),
        }
    if excess > 0:
        return {
            "tone": "",
            "text": (
                f"Lost {pnl:,.2f} ({total * 100:+.1f}%), but lost less than holding PAXG "
                f"({hold * 100:+.1f}%). Defensive, not profitable."
            ),
        }
    return {
        "tone": "bad",
        "text": (
            f"Lost {pnl:,.2f} ({total * 100:+.1f}%) over {metrics['days']:.0f} days and "
            f"did worse than just holding PAXG ({hold * 100:+.1f}%). Do not trade this."
        ),
    }


def open_price_lookup(config) -> dict[int, float]:
    """open_time -> open price, used to fill entries at the next bar's open."""
    from .data.store import store_from_config

    bars = store_from_config(config).load_bars()
    return dict(
        zip(
            bars["open_time"].astype("int64").tolist(),
            bars["open"].astype("float64").tolist(),
        )
    )


def prepare_backtest_frame(frame, rows, open_by_time: dict[int, float]):
    """Slice the dataset to ``rows`` and attach a tradeable entry price."""
    import numpy as np
    import pandas as pd

    sliced = frame.iloc[rows].reset_index(drop=True)
    opens = np.array(
        [open_by_time.get(int(t), float("nan")) for t in sliced["open_time"]],
        dtype=np.float64,
    )
    next_open = np.full(len(opens), np.nan, dtype=np.float64)
    if len(opens) > 1:
        next_open[:-1] = opens[1:]
    closes = sliced["close"].to_numpy(dtype=np.float64)
    entry = np.where(np.isfinite(next_open), next_open, closes)

    return pd.DataFrame(
        {
            "open_time": sliced["open_time"].to_numpy(dtype=np.int64),
            "entry_price": entry,
            "exit_time": sliced["exit_time"].to_numpy(dtype=np.int64),
            "exit_price": sliced["exit_price"].to_numpy(dtype=np.float64),
            "holding_bars": sliced["holding_bars"].to_numpy(dtype=np.int64),
            "label": sliced["label"].to_numpy(dtype=np.int64),
        }
    )
