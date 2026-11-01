"""Walk-forward evaluation over rolling 3-month blocks.

A single 70/15/15 split answers "did this model work on one particular
window", which is a different question from "does it work". For a long-only
strategy on an asset that trends for months at a time, one window measures the
market more than the model — the 4h sweep demonstrated this: validation +41%,
test -4%, same model.

A fold trains on everything before its test block (expanding window), validates
on the 90 days immediately before it, and trades the 90-day test block. Folds
slide forward with non-overlapping test blocks, so per-fold returns form a
continuous path: 24 folds cover six years.

The output that matters is not one number but the distribution: median 3-month
return, worst quarter, share of quarters above +5%, and share beating
buy-and-hold over the same block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .backtest import (
    BacktestConfig,
    open_price_lookup,
    prepare_backtest_frame,
    run_backtest,
)
from .training import TrainConfig, predict_proba, train_model


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]
    test_start_ms: int
    test_end_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "test_start_ms": self.test_start_ms,
            "test_end_ms": self.test_end_ms,
        }


def build_folds(
    frame,
    *,
    test_bars: int,
    val_bars: int,
    min_train_bars: int,
    window: int,
    step_bars: int | None = None,
) -> list[Fold]:
    """Expanding-window folds with non-overlapping test blocks.

    ``min_train_bars`` counts rows, including the feature warm-up that the
    dataset has already dropped. ``step_bars`` defaults to the test length, so
    consecutive test blocks abut instead of overlapping.
    """
    n_rows = len(frame)
    step = int(step_bars or test_bars)
    first = window - 1
    folds: list[Fold] = []

    cursor = int(min_train_bars) + int(val_bars)
    index = 0
    while cursor + test_bars <= n_rows:
        train = (first, cursor - val_bars)
        validation = (cursor - val_bars, cursor)
        test = (cursor, cursor + test_bars)

        if train[1] - train[0] >= window and validation[1] - validation[0] >= window:
            folds.append(
                Fold(
                    index=index,
                    train=train,
                    validation=validation,
                    test=test,
                    test_start_ms=int(frame["open_time"].iloc[test[0]]),
                    test_end_ms=int(frame["open_time"].iloc[test[1] - 1]),
                )
            )
            index += 1

        cursor += step

    return folds


def evaluate_fold(
    frame,
    feature_names: list[str],
    fold: Fold,
    train_config: TrainConfig,
    checkpoint_dir: Path,
    backtest_config: BacktestConfig,
    open_by_time: dict[int, float],
    *,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Train on the fold's train/validation blocks, then trade its test block."""
    result = train_model(
        frame,
        feature_names,
        train_config,
        checkpoint_dir,
        should_stop=should_stop,
        splits={
            "train": fold.train,
            "validation": fold.validation,
            "test": fold.test,
        },
    )

    base = {
        "fold": fold.index,
        "test_start_ms": fold.test_start_ms,
        "test_end_ms": fold.test_end_ms,
        "epochs": result.epochs_done,
        "duration_s": result.duration_s,
    }

    if result.checkpoint_path is None:
        return {**base, "status": result.stop_reason, "note": "no checkpoint"}

    checkpoint = torch.load(
        result.checkpoint_path, map_location="cpu", weights_only=False
    )
    rows = np.arange(fold.test[0], fold.test[1])
    probabilities = predict_proba(checkpoint, frame, rows)
    prepared = prepare_backtest_frame(frame, rows, open_by_time)
    backtest = run_backtest(prepared, probabilities, backtest_config)
    metrics = backtest.metrics

    return {
        **base,
        "status": "ok",
        "return": metrics["total_return"],
        "buy_hold": metrics["buy_hold_return"],
        "excess": metrics["excess_return"],
        "trades": metrics["trades"],
        "win_rate": metrics["win_rate"],
        "max_drawdown": metrics["max_drawdown"],
        "balanced_accuracy": result.metrics.get("balanced_accuracy"),
        "baseline_balanced": result.metrics.get("baseline_balanced_accuracy"),
        "best_epoch": result.metrics.get("best_epoch"),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Distribution across folds — the only honest summary of a strategy."""
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return {"folds": 0}

    returns = np.array([row["return"] for row in ok], dtype=float)
    holds = np.array([row["buy_hold"] for row in ok], dtype=float)
    excess = returns - holds
    trades = np.array([row["trades"] for row in ok], dtype=float)
    drawdowns = np.array([row["max_drawdown"] for row in ok], dtype=float)
    balanced = np.array(
        [row.get("balanced_accuracy") or 0.0 for row in ok], dtype=float
    )

    return {
        "folds": len(ok),
        "median_return": round(float(np.median(returns)), 4),
        "mean_return": round(float(returns.mean()), 4),
        "worst_return": round(float(returns.min()), 4),
        "best_return": round(float(returns.max()), 4),
        "share_over_5pct": round(float((returns >= 0.05).mean()), 3),
        "median_excess": round(float(np.median(excess)), 4),
        "share_beating_hold": round(float((excess > 0).mean()), 3),
        "compounded": round(float(np.prod(1.0 + returns) - 1.0), 4),
        "hold_compounded": round(float(np.prod(1.0 + holds) - 1.0), 4),
        "median_trades": int(np.median(trades)),
        "median_max_drawdown": round(float(np.median(drawdowns)), 4),
        "median_balanced": round(float(np.median(balanced)), 4),
    }
