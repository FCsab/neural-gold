"""Long-by-default policy: does the signal know when to step aside?

The barrier labels are ignored here. The position is driven bar by bar — stay
long, go flat when the model's top class is "stop", re-enter when it is not —
so the comparison against buy-and-hold is direct. That is the whole question:
can the model beat simply holding?

Deliberately minimal: no position sizing and no stop-loss of its own. The
one addition is a Schmitt trigger around the stop threshold, because an
over-eager exit rule churns the position and pays fees on every round trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, Trade


def run_hold_policy(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    config: BacktestConfig,
    *,
    stop_threshold: float | None = None,
    reentry_margin: float = 0.05,
    min_flat_bars: int = 0,
) -> BacktestResult:
    """frame: prepared rows (needs close, entry_price, open_time).

    ``stop_threshold`` is the probability of the *stop* class at which the
    policy steps aside. ``None`` keeps the original behaviour (top class is
    stop). A number makes the rule explicit and, crucially, tunable on the
    validation block: a threshold above what the model usually outputs leaves
    the position alone, which is the right default for an asset that trends up.

    ``reentry_margin`` puts the re-entry line below the exit line, so a
    probability hovering at the boundary cannot churn the position. Each round
    trip costs fees plus slippage on both sides; that is the dominant cost of
    an over-eager stop rule. ``min_flat_bars`` adds a cooldown for the same
    reason.
    """
    if len(frame) == 0:
        return BacktestResult(trades=[], equity_curve=[], metrics={"trades": 0})
    if stop_threshold is not None and not 0.0 <= stop_threshold <= 1.0:
        raise ValueError("stop_threshold must be within [0, 1]")
    if reentry_margin < 0.0:
        raise ValueError("reentry_margin must be >= 0")
    if min_flat_bars < 0:
        raise ValueError("min_flat_bars must be >= 0")

    close = frame["close"].to_numpy(dtype=np.float64)
    entry = frame["entry_price"].to_numpy(dtype=np.float64)
    entry = np.where(np.isfinite(entry), entry, close)
    times = frame["open_time"].to_numpy(dtype=np.int64)

    # probabilities are ordered (-1, 0, +1); class 0 is "stop"
    stop_prob = probabilities[:, 0]
    if stop_threshold is None:
        exit_signal = probabilities.argmax(axis=1) == 0
        enter_signal = ~exit_signal
    else:
        threshold = float(stop_threshold)
        if threshold >= 1.0:
            # 1.0 means "never step aside": enter on the first bar and stay
            # long. A float32 softmax can saturate to exactly 1.0, so ``>=``
            # would still fire an exit and an immediate, fee-paying re-entry.
            exit_signal = np.zeros(len(stop_prob), dtype=bool)
            enter_signal = np.ones(len(stop_prob), dtype=bool)
        else:
            exit_signal = stop_prob >= threshold
            # The re-entry line sits below the exit line, so a probability
            # hovering at the boundary cannot churn the position.
            enter_signal = stop_prob <= max(0.0, threshold - reentry_margin)
    confidence = probabilities.max(axis=1)

    cost = (config.fee_bps + config.slippage_bps) / 10_000.0
    round_trip = 2 * cost

    cash = float(config.initial_equity)
    units = 0.0
    entry_price = 0.0
    entry_time = 0
    entry_index = 0
    entry_cash = 0.0
    trades: list[Trade] = []
    curve: list[dict[str, float]] = []
    exposure = 0
    # The cooldown applies after an exit, not at the start of the block.
    flat_bars = min_flat_bars

    for i in range(len(close)):
        price = float(entry[i])

        if units == 0.0 and enter_signal[i] and flat_bars >= min_flat_bars:
            entry_cash = cash
            units = cash * (1 - cost) / price
            cash = 0.0
            entry_price = price
            entry_time = int(times[i])
            entry_index = i
        elif units > 0.0 and exit_signal[i]:
            proceeds = units * price * (1 - cost)
            gross = price / entry_price - 1.0
            cash = proceeds
            trades.append(
                Trade(
                    entry_time_ms=entry_time,
                    exit_time_ms=int(times[i]),
                    entry_price=entry_price,
                    exit_price=price,
                    gross_return=gross,
                    net_return=proceeds / entry_cash - 1.0,
                    pnl=proceeds - entry_cash,
                    equity_after=proceeds,
                    outcome=1 if price > entry_price else -1,
                    confidence=float(confidence[i]),
                    holding_bars=i - entry_index,
                )
            )
            units = 0.0
            flat_bars = 0

        if units == 0.0:
            flat_bars += 1
        else:
            exposure += 1
        curve.append(
            {"time": int(times[i]) / 1000.0, "equity": round(cash + units * float(close[i]), 2)}
        )

    # Close an open position at the last bar so the return is realised.
    if units > 0.0:
        proceeds = units * float(close[-1]) * (1 - cost)
        trades.append(
            Trade(
                entry_time_ms=entry_time,
                exit_time_ms=int(times[-1]),
                entry_price=entry_price,
                exit_price=float(close[-1]),
                gross_return=float(close[-1]) / entry_price - 1.0,
                net_return=proceeds / entry_cash - 1.0,
                pnl=proceeds - entry_cash,
                equity_after=proceeds,
                outcome=1 if close[-1] > entry_price else -1,
                confidence=float(confidence[-1]),
                holding_bars=len(close) - 1 - entry_index,
            )
        )
        cash = proceeds
        if curve:
            curve[-1]["equity"] = round(proceeds, 2)

    final = curve[-1]["equity"] if curve else float(config.initial_equity)
    peak = float(config.initial_equity)
    max_drawdown = 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        max_drawdown = max(max_drawdown, (peak - point["equity"]) / peak)

    # Same block-bounded formula as run_backtest, so the two policies report
    # excess against the same buy-and-hold number.
    hold_return = (
        float(close[-1]) * (1.0 - cost)
    ) / (float(entry[0]) * (1.0 + cost)) - 1.0
    net_returns = np.array([t.net_return for t in trades], dtype=float)

    start_ms = int(times[0])
    end_ms = int(times[-1])
    return BacktestResult(
        trades=trades,
        equity_curve=curve,
        metrics={
            "trades": len(trades),
            "initial_equity": float(config.initial_equity),
            "final_equity": round(final, 2),
            "total_return": round(final / float(config.initial_equity) - 1.0, 6),
            "buy_hold_return": round(hold_return, 6),
            "excess_return": round(final / float(config.initial_equity) - 1.0 - hold_return, 6),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round(float((net_returns > 0).mean()), 4) if net_returns.size else None,
            "exposure": round(exposure / len(close), 4),
            "round_trip_cost": round(round_trip, 6),
            "fees_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "days": round((end_ms - start_ms) / 86_400_000, 1),
            "policy": "hold",
            "stop_threshold": (
                None if stop_threshold is None else round(float(stop_threshold), 4)
            ),
            "reentry_margin": (
                None if stop_threshold is None else round(float(reentry_margin), 4)
            ),
            "min_flat_bars": int(min_flat_bars),
            "stop_share": round(float(exit_signal.mean()), 4),
            "mean_stop_prob": round(float(stop_prob.mean()), 4),
            "costs_applied": True,
        },
    )
