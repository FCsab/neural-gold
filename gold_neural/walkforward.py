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

import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from .dataset import build_dataset
from .hold_policy import run_hold_policy

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
    policy: str = "signal",
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
    if policy == "hold":
        backtest = run_hold_policy(prepared, probabilities, backtest_config)
    else:
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


# --------------------------------------------------------------------------
# the sweep loop, shared by the API job and the detached CLI runner
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SweepConfig:
    test_bars: int
    val_bars: int
    min_train_bars: int
    epochs: int = 15
    batch_size: int = 256
    lr: float = 3e-4
    minutes_per_fold: float = 3.0
    device: str = "auto"
    policy: str = "signal"


def spec_key(spec: dict[str, Any]) -> str:
    return (
        f"u{spec['k_up']}d{spec['k_down']}"
        f"h{spec['holding']}w{spec['window']}s{spec['seed']}"
    )


def run_sweep(
    *,
    config,
    bars,
    specs: Iterable[dict[str, Any]],
    sweep: SweepConfig,
    backtest_config: BacktestConfig,
    out_dir: Path,
    on_status: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate every spec across every fold, resuming from what is on disk.

    Consumes ``specs`` until it is exhausted or ``should_stop`` returns True.
    Each finished fold is appended to ``results.jsonl`` and skipped on a later
    run, so an interrupted sweep costs one fold rather than the whole night.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    status_path = out / "status.json"
    checkpoint_dir = out / "checkpoint"

    def emit(message: str) -> None:
        if log:
            log(message)

    done: set[str] = set()
    by_spec: dict[str, list[dict[str, Any]]] = {}
    if results_path.exists():
        with results_path.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") != "ok":
                    continue
                done.add(row["key"])
                by_spec.setdefault(spec_key(row["spec"]), []).append(row)

    open_by_time = open_price_lookup(config)
    started = time.monotonic()
    folds_done = 0
    specs_done = 0
    total_specs = len(specs) if hasattr(specs, "__len__") else None
    stopped = False

    def build_status(current: str, state: str = "running") -> dict[str, Any]:
        elapsed = (time.monotonic() - started) / 60.0
        ranked = sorted(
            (
                (key, summarise(rows))
                for key, rows in by_spec.items()
                if len(rows) >= 5
            ),
            key=lambda item: item[1].get("median_return", -9.0),
            reverse=True,
        )
        payload = {
            "status": state,
            "elapsed_minutes": round(elapsed, 1),
            "folds_done_this_run": folds_done,
            "folds_total": len(done),
            "specs_done": specs_done,
            "specs_total": total_specs,
            "folds_per_minute": round(folds_done / elapsed, 2) if elapsed else 0.0,
            "current": current,
            "leaderboard": [{"spec": key, **summary} for key, summary in ranked[:20]],
        }
        status_path.write_text(json.dumps(payload, indent=2))
        if on_status:
            on_status(payload)
        return payload

    build_status("starting")

    try:
        for spec in specs:
            if should_stop and should_stop():
                stopped = True
                break

            label_config = replace(
                config.labels,
                k_up=spec["k_up"],
                k_down=spec["k_down"],
                max_holding_bars=spec["holding"],
            )
            key_base = spec_key(spec)

            try:
                dataset = build_dataset(bars, config.features, label_config)
            except Exception as exc:  # noqa: BLE001 - one bad spec is not the sweep
                emit(f"skip {key_base}: dataset failed: {exc}")
                continue

            frame = dataset.frame.reset_index(drop=True)
            folds = build_folds(
                frame,
                test_bars=sweep.test_bars,
                val_bars=sweep.val_bars,
                min_train_bars=sweep.min_train_bars,
                window=spec["window"],
            )
            if not folds:
                emit(f"skip {key_base}: no folds fit")
                continue

            batch = by_spec.setdefault(key_base, [])
            added = 0
            for fold in folds:
                key = f"{key_base}|f{fold.index}"
                if key in done:
                    continue
                if should_stop and should_stop():
                    stopped = True
                    break

                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                train_config = TrainConfig(
                    window=spec["window"],
                    epochs=sweep.epochs,
                    batch_size=sweep.batch_size,
                    lr=sweep.lr,
                    patience=0,
                    max_minutes=sweep.minutes_per_fold,
                    device=sweep.device,
                    seed=spec["seed"],
                )
                try:
                    row = evaluate_fold(
                        frame,
                        dataset.feature_columns,
                        fold,
                        train_config,
                        checkpoint_dir,
                        backtest_config,
                        open_by_time,
                        policy=sweep.policy,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad fold is not the sweep
                    emit(f"fold failed {key}: {type(exc).__name__}: {exc}")
                    continue

                row["key"] = key
                row["spec"] = spec
                with results_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                done.add(key)
                batch.append(row)
                added += 1
                folds_done += 1

                if folds_done % 5 == 0:
                    build_status(f"{key_base} fold {fold.index + 1}/{len(folds)}")

            specs_done += 1
            if added:
                summary = summarise(batch)
                build_status(f"done {key_base}")
                emit(
                    f"{key_base}: {summary['folds']} folds · "
                    f"median {summary['median_return'] * 100:+.2f}% · "
                    f"worst {summary['worst_return'] * 100:+.2f}% · "
                    f">=+5% {summary['share_over_5pct'] * 100:.0f}% · "
                    f"beats hold {summary['share_beating_hold'] * 100:.0f}%"
                )
            if stopped:
                break
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    final = build_status("stopped" if stopped else "done", "stopped" if stopped else "done")
    emit(f"{'stopped' if stopped else 'finished'} after {folds_done:,} folds this run")
    return final
