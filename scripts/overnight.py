#!/usr/bin/env python3
"""Detached walk-forward sweep (command line wrapper).

The sweep itself lives in :func:`gold_neural.walkforward.run_sweep` and is the
same code the web UI drives. This script only parses arguments, backfills if
the bars are missing, and loops the spec iterator forever so an unattended
machine always has work.

The default policy is ``hold``: stay long, step aside only when the model is
confident about the stop outcome, and let validation decide how confident that
has to be. That is the only formulation that can be compared directly against
buy-and-hold, which is the question this project is trying to answer.

For the browser, use the Walk-forward tab instead.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gold_neural.backtest import BacktestConfig
from gold_neural.config import load_config
from gold_neural.data.backfill import backfill
from gold_neural.data.store import store_from_config
from gold_neural.exchange.rest import interval_to_ms
from gold_neural.walkforward import (
    DEFAULT_STOP_THRESHOLDS,
    SweepConfig,
    run_sweep,
)

STOP = {"requested": False}

K_UP = (2.5, 3.0, 3.5, 4.0)
K_DOWN = (2.0, 2.5, 3.0, 4.0)
HOLDING = (192, 288)
WINDOW = (32, 64, 128)
# Stop probabilities are selected per fold on the validation block. 1.0 means
# "never step aside", so holding is always on the menu and a bad rule cannot
# be forced into the test block.
STOP_THRESHOLDS = DEFAULT_STOP_THRESHOLDS


def _handle_signal(_signum, _frame) -> None:
    STOP["requested"] = True


def spec_stream(k_up=K_UP, k_down=K_DOWN, holding=HOLDING, window=WINDOW, seeds=(7,)):
    """Endless stream of configurations.

    The first passes use the supplied seeds. Once those are exhausted the seed
    advances past the largest one, so the runner never reaches a state where
    every key is already on disk and it spins without doing work.
    """
    base = tuple(int(seed) for seed in seeds) or (7,)
    total = len(k_up) * len(k_down) * len(holding) * len(window)
    index = 0
    while True:
        seed = base[index] if index < len(base) else max(base) + index - len(base) + 1
        pass_index = 0
        for up in k_up:
            for down in k_down:
                for hold in holding:
                    for win in window:
                        pass_index += 1
                        yield {
                            "k_up": up,
                            "k_down": down,
                            "holding": hold,
                            "window": win,
                            "seed": seed,
                            "pass_index": pass_index,
                            "pass_total": total,
                        }
        index += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "data" / "overnight"))
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--val-days", type=int, default=90)
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minutes-per-fold", type=float, default=3.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k-up", default=None, help="comma list, e.g. 3,3.5,4")
    parser.add_argument("--k-down", default=None, help="comma list")
    parser.add_argument("--holding", default=None, help="comma list of bar counts")
    parser.add_argument("--window", default=None, help="comma list of bar counts")
    parser.add_argument("--seeds", default="7", help="comma list of seeds")
    parser.add_argument(
        "--policy",
        choices=("hold", "signal"),
        default="hold",
        help="hold = long by default, step aside on stop (the buy-and-hold test)",
    )
    parser.add_argument(
        "--stop-thresholds",
        default=",".join(str(value) for value in STOP_THRESHOLDS),
        help="comma list of stop probabilities; 1.0 = never step aside",
    )
    parser.add_argument("--reentry-margin", type=float, default=0.05)
    parser.add_argument("--min-flat-bars", type=int, default=1)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def number_list(raw, default, cast=float):
        if raw is None:
            return tuple(default)
        return tuple(cast(part) for part in raw.split(",") if part.strip())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{stamp} {message}"
        print(line, flush=True)
        with (out / "log.txt").open("a") as handle:
            handle.write(line + "\n")

    base = load_config(args.config)
    config = replace(
        base,
        exchange=replace(base.exchange, interval=args.interval),
        data=replace(base.data, root=args.data_root),
    )

    store = store_from_config(config)
    if store.coverage().rows < 5000:
        log(f"backfilling {args.interval} bars for {args.data_root}")
        stats = asyncio.run(backfill(config, years=6.2))
        log(f"backfill: {stats.format()}")

    bars = store.load_bars()
    bars_per_day = 86_400_000 // interval_to_ms(args.interval)

    thresholds = tuple(
        float(part) for part in args.stop_thresholds.split(",") if part.strip()
    )
    up_grid = number_list(args.k_up, K_UP)
    down_grid = number_list(args.k_down, K_DOWN)
    holding_grid = number_list(args.holding, HOLDING, int)
    window_grid = number_list(args.window, WINDOW, int)
    seed_list = number_list(args.seeds, (7,), int)
    grid_size = len(up_grid) * len(down_grid) * len(holding_grid) * len(window_grid)

    log(
        f"start: {args.interval} · {len(bars):,} bars · "
        f"test {bars_per_day * args.test_days} bars · seeds "
        f"{seed_list} · policy {args.policy} · "
        f"stop thresholds {thresholds} · {grid_size} settings/pass"
    )

    stop_file = out / "STOP"
    run_sweep(
        config=config,
        bars=bars,
        specs=spec_stream(
            k_up=up_grid,
            k_down=down_grid,
            holding=holding_grid,
            window=window_grid,
            seeds=seed_list,
        ),
        sweep=SweepConfig(
            test_bars=bars_per_day * args.test_days,
            val_bars=bars_per_day * args.val_days,
            min_train_bars=bars_per_day * args.min_train_days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            minutes_per_fold=args.minutes_per_fold,
            device=args.device,
            policy=args.policy,
            stop_thresholds=thresholds,
            reentry_margin=args.reentry_margin,
            min_flat_bars=args.min_flat_bars,
            specs_per_pass=grid_size,
        ),
        backtest_config=BacktestConfig(
            fee_bps=args.fee_bps, slippage_bps=args.slippage_bps
        ),
        out_dir=out,
        should_stop=lambda: stop_file.exists() or STOP["requested"],
        log=log,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
