"""Command line entry point: ``python -m gold_neural.cli <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig, load_config
from .data.backfill import backfill, print_progress, verify_bars
from .data.store import store_from_config
from .dataset import build_dataset, walk_forward_split
from .exchange.rest import fetch_exchange_info, interval_to_ms, make_client
from .labels import label_distribution, label_triple_barrier
from .leakage import run_leakage_suite


def _parse_time(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"cannot parse time {text!r}: {exc}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _iso(ms: int | None) -> str:
    if ms is None:
        return "n/a"
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def _load_bars(config: AppConfig) -> pd.DataFrame:
    store = store_from_config(config)
    bars = store.load_bars()
    if bars.empty:
        raise SystemExit(
            "no bars stored yet; run: python -m gold_neural.cli backfill --years 2"
        )
    return bars


def cmd_backfill(args: argparse.Namespace, config: AppConfig) -> int:
    stats = asyncio.run(
        backfill(
            config,
            years=args.years,
            start_ms=_parse_time(args.start),
            end_ms=_parse_time(args.end),
            progress=print_progress,
        )
    )
    print("backfill complete")
    print(f"  {stats.format()}")
    print(f"  range: {_iso(stats.start_ms)} -> {_iso(stats.end_ms)}")

    store = store_from_config(config)
    bars = store.load_bars()
    problems = verify_bars(bars)
    print(f"  stored bars: {len(bars):,}")
    if args.gaps:
        gaps = store.gaps(interval_to_ms(config.exchange.interval))
        print(f"  gaps (>1.5x bar interval): {len(gaps)}")
        for left, right in gaps[:20]:
            print(f"    {_iso(left)} -> {_iso(right)}")
    if problems:
        print("  DATA PROBLEMS:")
        for problem in problems:
            print(f"    - {problem}")
        return 1
    print("  data verification: clean")
    return 0


def cmd_info(args: argparse.Namespace, config: AppConfig) -> int:
    store = store_from_config(config)
    coverage = store.coverage()
    print(f"symbol:      {config.exchange.symbol}")
    print(f"interval:    {config.exchange.interval}")
    print(f"data root:   {config.data_root}")
    print(f"bars dir:    {config.bars_dir}")
    print(f"stored bars: {coverage.rows:,}")
    print(f"range:       {_iso(coverage.start_ms)} -> {_iso(coverage.end_ms)}")
    partitions = store.partitions()
    print(f"partitions:  {len(partitions)}")
    for part in partitions:
        print(
            f"  {part['year']}-{part['month']:02d}  {part['rows']:>6,} bars  "
            f"{_iso(part['start_ms'])} -> {_iso(part['end_ms'])}"
        )
    return 0


def cmd_exchange_info(args: argparse.Namespace, config: AppConfig) -> int:
    async def run() -> dict:
        async with make_client(
            config.exchange.base_url, config.exchange.request_timeout_s
        ) as client:
            return await fetch_exchange_info(
                client,
                symbol=config.exchange.symbol,
                max_retries=config.exchange.max_retries,
            )

    info = asyncio.run(run())
    out = Path(args.out) if args.out else config.data_root / "exchange_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(f"wrote {out}")
    for name, payload in info["filters"].items():
        print(f"  {name}: {payload}")
    return 0


def cmd_dataset(args: argparse.Namespace, config: AppConfig) -> int:
    bars = _load_bars(config)
    dataset = build_dataset(bars, config.features, config.labels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(out, index=False, compression=config.data.compression)

    split = walk_forward_split(dataset)
    counts = dataset.frame["label"].value_counts().sort_index()
    print(f"bars:      {len(bars):,}")
    print(f"rows:      {len(dataset):,}  (dropped {dataset.frame.attrs.get('dropped_rows', 0):,})")
    print(f"features:  {len(dataset.feature_columns)}")
    print(f"labels:    {counts.to_dict()}")
    print(
        "split:     "
        f"train={len(split.train):,} val={len(split.validation):,} test={len(split.test):,}"
    )
    print(f"written:   {out}")
    return 0


def cmd_labels(args: argparse.Namespace, config: AppConfig) -> int:
    bars = _load_bars(config)
    labels = label_triple_barrier(bars, config.labels)
    distribution = label_distribution(labels)
    print(f"bars: {len(bars):,}  labelled: {int(labels['valid'].sum()):,}")
    print(
        f"long={distribution['long']:,} stop={distribution['stop']:,} "
        f"timeout={distribution['flat']:,}"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(out, index=False)
        print(f"written: {out}")
    return 0


def cmd_leakage(args: argparse.Namespace, config: AppConfig) -> int:
    bars = _load_bars(config)
    report = run_leakage_suite(
        bars, config.features, config.labels, samples=args.samples
    )
    print(report.format())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.to_json())
        print(f"\nreport written: {out}")
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gold-neural",
        description="PAXG/USDT neural trading system — data layer",
    )
    parser.add_argument(
        "--config", default=None, help="path to config.yaml (default: ./config.yaml)"
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backfill = sub.add_parser("backfill", help="download historical klines")
    p_backfill.add_argument("--years", type=float, default=2.0)
    p_backfill.add_argument("--start", default=None, help="ISO timestamp (UTC)")
    p_backfill.add_argument("--end", default=None, help="ISO timestamp (UTC)")
    p_backfill.add_argument("--gaps", action="store_true", help="report gaps")
    p_backfill.set_defaults(func=cmd_backfill)

    p_info = sub.add_parser("info", help="show stored coverage")
    p_info.set_defaults(func=cmd_info)

    p_ex = sub.add_parser("exchange-info", help="fetch and cache trading filters")
    p_ex.add_argument("--out", default=None)
    p_ex.set_defaults(func=cmd_exchange_info)

    p_ds = sub.add_parser("dataset", help="build features+labels parquet")
    p_ds.add_argument("--out", default="data/dataset.parquet")
    p_ds.set_defaults(func=cmd_dataset)

    p_labels = sub.add_parser("labels", help="label counts, optional parquet dump")
    p_labels.add_argument("--out", default=None)
    p_labels.set_defaults(func=cmd_labels)

    p_leak = sub.add_parser("leakage", help="run the mandatory leakage suite")
    p_leak.add_argument("--samples", type=int, default=10)
    p_leak.add_argument("--out", default=None)
    p_leak.set_defaults(func=cmd_leakage)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    return int(args.func(args, config))


if __name__ == "__main__":
    sys.exit(main())
