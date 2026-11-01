#!/usr/bin/env python3
"""Overnight walk-forward sweep over label settings.

Runs until ``data/overnight/STOP`` exists. Progress is append-only: every
finished fold is one JSON line in ``results.jsonl``, and a restart skips the
keys already present. That makes the job safe to kill, crash, or restart.

Nothing is padded. When the grid is exhausted the seed advances and the sweep
continues, so an unattended machine always has real work rather than a sleep
loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import signal
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gold_neural.backtest import BacktestConfig, open_price_lookup
from gold_neural.config import load_config
from gold_neural.data.backfill import backfill
from gold_neural.data.store import store_from_config
from gold_neural.dataset import build_dataset
from gold_neural.exchange.rest import interval_to_ms
from gold_neural.training import TrainConfig
from gold_neural.walkforward import build_folds, evaluate_fold, summarise

STOP = {"requested": False}

# Ordered so the plausible region is explored first: costs make tight targets
# pointless, and a stop much wider than the target skews the payoff.
K_UP = (3.0, 4.0, 2.5, 3.5, 2.0, 5.0)
K_DOWN = (2.0, 3.0, 1.5, 2.5, 4.0)
HOLDING = (24, 48, 96, 12, 6, 288, 192)
WINDOW = (64, 32, 128)


def _handle_signal(_signum, _frame) -> None:
    STOP["requested"] = True


def spec_stream(k_up=K_UP, k_down=K_DOWN, holding=HOLDING, window=WINDOW, seeds=(7,)):
    """Endless stream of configurations, looping over the given seeds forever.

    The defaults walk the whole grid once per seed, advancing the seed when the
    grid is exhausted so an unattended machine always has work. Narrowing the
    lists is how a region of the grid gets a seed-averaged re-test.
    """
    index = 0
    while True:
        seed = seeds[index % len(seeds)]
        for up in k_up:
            for down in k_down:
                for hold in holding:
                    for win in window:
                        yield {
                            "k_up": up,
                            "k_down": down,
                            "holding": hold,
                            "window": win,
                            "seed": seed,
                        }
        index += 1


def spec_key(spec: dict) -> str:
    return (
        f"u{spec['k_up']}d{spec['k_down']}"
        f"h{spec['holding']}w{spec['window']}s{spec['seed']}"
    )


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
    parser.add_argument(
        "--k-up", default=None, help="comma list, e.g. 3,3.5,4 (default: full grid)"
    )
    parser.add_argument("--k-down", default=None, help="comma list")
    parser.add_argument("--holding", default=None, help="comma list of bar counts")
    parser.add_argument("--window", default=None, help="comma list of bar counts")
    parser.add_argument(
        "--seeds", default="7", help="comma list of seeds to cycle through"
    )
    args = parser.parse_args()

    def number_list(raw, default, cast=float):
        if raw is None:
            return tuple(default)
        return tuple(cast(part) for part in raw.split(",") if part.strip())

    grid = dict(
        k_up=number_list(args.k_up, K_UP),
        k_down=number_list(args.k_down, K_DOWN),
        holding=number_list(args.holding, HOLDING, int),
        window=number_list(args.window, WINDOW, int),
        seeds=number_list(args.seeds, (7,), int),
    )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    status_path = out / "status.json"
    stop_file = out / "STOP"

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
    bar_ms = interval_to_ms(args.interval)
    bars_per_day = 86_400_000 // bar_ms
    test_bars = bars_per_day * args.test_days
    val_bars = bars_per_day * args.val_days
    min_train_bars = bars_per_day * args.min_train_days

    log(
        f"start: {args.interval} · {len(bars):,} bars · test {test_bars} bars "
        f"({args.test_days}d) · val {val_bars} · min train {min_train_bars}"
    )

    # Resume: one row per finished fold, keyed by spec + fold.
    done: set[str] = set()
    by_spec: dict[str, list[dict]] = {}
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
    if done:
        log(f"resuming: {len(done):,} folds already on disk")

    open_by_time = open_price_lookup(config)
    backtest_config = BacktestConfig(
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps
    )
    checkpoint_dir = out / "checkpoint"
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    folds_done = 0

    def write_status(current: str = "") -> None:
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
        status_path.write_text(
            json.dumps(
                {
                    "started_at": started_at,
                    "interval": args.interval,
                    "elapsed_minutes": round(elapsed, 1),
                    "folds_done_this_run": folds_done,
                    "folds_total": len(done),
                    "specs_tracked": len(by_spec),
                    "folds_per_minute": round(folds_done / elapsed, 2) if elapsed else 0,
                    "current": current,
                    "leaderboard": [
                        {"spec": key, **summary} for key, summary in ranked[:10]
                    ],
                },
                indent=2,
            )
        )

    write_status("starting")

    try:
        for spec in spec_stream(**grid):
            if stop_file.exists() or STOP["requested"]:
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
            except Exception as exc:  # noqa: BLE001 - a bad config must not stop the night
                log(f"skip {key_base}: dataset failed: {exc}")
                continue

            frame = dataset.frame.reset_index(drop=True)
            folds = build_folds(
                frame,
                test_bars=test_bars,
                val_bars=val_bars,
                min_train_bars=min_train_bars,
                window=spec["window"],
            )
            if not folds:
                log(f"skip {key_base}: no folds fit")
                continue

            batch = by_spec.setdefault(key_base, [])
            added = 0
            for fold in folds:
                key = f"{key_base}|f{fold.index}"
                if key in done:
                    continue
                if stop_file.exists() or STOP["requested"]:
                    break

                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                train_config = TrainConfig(
                    window=spec["window"],
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=0,
                    max_minutes=args.minutes_per_fold,
                    device=args.device,
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
                    )
                except Exception as exc:  # noqa: BLE001 - one bad fold is not the night
                    log(f"fold failed {key}: {type(exc).__name__}: {exc}")
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
                    write_status(f"{key_base} fold {fold.index + 1}/{len(folds)}")

            if added:
                summary = summarise(batch)
                write_status(f"done {key_base}")
                log(
                    f"{key_base}: {summary['folds']} folds · "
                    f"median {summary['median_return'] * 100:+.2f}% · "
                    f"worst {summary['worst_return'] * 100:+.2f}% · "
                    f">=+5% {summary['share_over_5pct'] * 100:.0f}% · "
                    f"beats hold {summary['share_beating_hold'] * 100:.0f}% · "
                    f"compounded {summary['compounded'] * 100:+.1f}%"
                )
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        write_status("stopped")
        log(f"stopped after {folds_done:,} folds this run ({len(done):,} total)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
