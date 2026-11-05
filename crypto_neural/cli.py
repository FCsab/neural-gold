"""Command line entry point: ``python -m crypto_neural.cli <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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
            "no bars stored yet; run: python -m crypto_neural.cli backfill --years 2"
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


def cmd_risk(args: argparse.Namespace, config: AppConfig) -> int:
    """The drawdown-risk walk-forward on ``risk.interval`` bars.

    Exit code 0 means the run finished, whatever the verdict; read the
    verdict. The report lands in data/reports/risk.json, where the web UI
    finds it.
    """

    from .risk import run_risk_walkforward
    from .server import reports

    from .server.tasks import record_reading, top_up_bars

    risk = config.risk
    local = replace(
        config,
        exchange=replace(config.exchange, symbol=risk.data_symbol, interval=risk.interval),
    )
    if not args.no_refresh:
        top_up_bars(local, print)
    bars = _load_bars(local)
    bars_per_day = 86_400_000 // interval_to_ms(risk.interval)
    report = run_risk_walkforward(
        bars,
        risk,
        config.features,
        bars_per_year=365.0 * bars_per_day,
        progress=lambda fraction, message: print(f"  {fraction:4.0%} {message}"),
    )
    print(report.format())
    payload = report.to_dict()
    path = reports.write_report(config, "risk", payload)
    print(f"\nreport written: {path}")
    if not args.no_record:
        record_reading(config, payload, risk, source="cli", log=print)
    return 0


def cmd_paper(args: argparse.Namespace, config: AppConfig) -> int:
    """Score the predictions log against the bars that followed it."""
    from .predictions import PredictionLog, paper_record, prediction_log_path

    path = prediction_log_path(config.data_root)
    if not path.exists():
        print("no predictions log yet; every `risk` run records one reading")
        return 0
    risk = config.risk
    local = replace(
        config,
        exchange=replace(config.exchange, symbol=risk.data_symbol, interval=risk.interval),
    )
    bars_per_day = 86_400_000 // interval_to_ms(risk.interval)
    report = paper_record(
        PredictionLog(path), _load_bars(local), fee_bps=risk.fee_bps,
        slippage_bps=risk.slippage_bps, bars_per_year=365.0 * bars_per_day,
    )
    print(report.format())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.to_json())
        print(f"\nreport written: {out}")
    return 0


def cmd_funding(args: argparse.Namespace, config: AppConfig) -> int:
    """The pre-registered funding-rate experiment (docs/risk.md, 2026-11-05).

    Two risk walk-forwards on identical bars, folds and ``RiskConfig``
    defaults; the only difference is the four funding columns. The decision
    rule is fixed in ``funding.compare_arms``. It records nothing in the
    predictions log: a funding reading must not enter the paper record.
    """
    from .funding import (
        compare_arms, experiment_bars, funding_builder, load_funding, update_funding,
    )
    from .risk import run_risk_walkforward
    from .server import reports
    from .server.tasks import top_up_bars

    risk = config.risk
    local = replace(
        config,
        exchange=replace(config.exchange, symbol=risk.data_symbol, interval=risk.interval),
    )
    if not args.no_refresh:
        top_up_bars(local, print)
        funding = update_funding(config.data.root, risk.data_symbol)
    else:
        funding = load_funding(config.data.root, risk.data_symbol)
    print(f"funding settlements: {len(funding)}, "
          f"{_iso(int(funding['funding_time'].min()))} -> "
          f"{_iso(int(funding['funding_time'].max()))}")
    bars = experiment_bars(_load_bars(local), funding)
    bars_per_day = 86_400_000 // interval_to_ms(risk.interval)
    arms = {}
    for name, extra in (("base", None), ("funding", funding_builder(funding))):
        print(f"\narm: {name}")
        arms[name] = run_risk_walkforward(
            bars,
            risk,
            config.features,
            bars_per_year=365.0 * bars_per_day,
            progress=lambda fraction, message: print(f"  {fraction:4.0%} {message}"),
            extra_features=extra,
        ).to_dict()
    result = compare_arms(arms["base"], arms["funding"])
    print("\n" + result["text"])
    for row in result["block_auc"]:
        print(f"  {row['start']}  base {row['base']}  funding {row['funding']}  "
              f"falls {row['falls']}")
    path = reports.write_report(config, "funding", {**result, "arms": arms})
    print(f"\nreport written: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto-neural",
        description="BTC/USDC neural trading system — data layer",
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

    p_risk = sub.add_parser(
        "risk", help="drawdown-risk walk-forward against buy-and-hold (4h bars)"
    )
    p_risk.add_argument(
        "--no-refresh", action="store_true", help="do not fetch bars newer than the store"
    )
    p_risk.add_argument(
        "--no-record", action="store_true", help="do not add the reading to the predictions log"
    )
    p_risk.set_defaults(func=cmd_risk)

    p_fund = sub.add_parser(
        "funding", help="run the pre-registered funding-rate experiment once"
    )
    p_fund.add_argument(
        "--no-refresh", action="store_true", help="use stored bars and settlements only"
    )
    p_fund.set_defaults(func=cmd_funding)

    p_paper = sub.add_parser("paper", help="score the predictions log (no orders, ever)")
    p_paper.add_argument("--out", default=None)
    p_paper.set_defaults(func=cmd_paper)

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
