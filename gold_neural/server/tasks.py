"""Job bodies invoked from the web UI.

Each function takes an explicit progress context so the browser can show what
is happening. They are plain synchronous functions: the job manager runs them
in a worker thread, and any async work (exchange calls, backfill) is wrapped
with ``asyncio.run`` inside that thread.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AppConfig, FeatureConfig, LabelConfig
from ..data.backfill import backfill, verify_bars
from ..data.store import store_from_config
from ..dataset import build_dataset, walk_forward_split
from ..exchange.rest import fetch_exchange_info, interval_to_ms, make_client
from ..labels import label_distribution, label_triple_barrier
from ..leakage import run_leakage_suite
from . import reports


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _class_stats(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    grouped = (
        frame.groupby("label")["realized_return"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    out: dict[str, dict[str, Any]] = {}
    for _, row in grouped.iterrows():
        out[str(int(row["label"]))] = {
            "count": int(row["count"]),
            "mean_return": _finite(row["mean"]),
            "std_return": _finite(row["std"]),
        }
    return out


def _verify_payload(config: AppConfig) -> dict[str, Any]:
    store = store_from_config(config)
    bars = store.load_bars()
    interval_ms = interval_to_ms(config.exchange.interval)
    gaps = store.gaps(interval_ms)
    problems = verify_bars(bars)
    return {
        "ok": not problems,
        "rows": int(len(bars)),
        "problems": problems,
        "gap_count": len(gaps),
        "gaps": [
            {"from_ms": int(left), "to_ms": int(right)}
            for left, right in gaps[:200]
        ],
        "interval_ms": interval_ms,
    }


def run_backfill(
    config: AppConfig,
    years: float,
    start_ms: int | None,
    end_ms: int | None,
    ctx,
) -> dict[str, Any]:
    ctx.message(f"backfilling {years:g}y of {config.exchange.interval} bars")
    stats = asyncio.run(
        backfill(
            config,
            years=years,
            start_ms=start_ms,
            end_ms=end_ms,
            progress=lambda s: ctx.message(s.format()),
        )
    )
    ctx.progress(0.85, "verifying stored bars")
    verification = _verify_payload(config)
    reports.write_report(config, "verify", verification)

    result = {
        "pages": stats.pages,
        "bars_fetched": stats.bars_fetched,
        "bars_written": stats.bars_written,
        "unclosed_skipped": stats.bars_skipped_unclosed,
        "elapsed_s": round(stats.elapsed_s, 2),
        "start": _iso(stats.start_ms),
        "end": _iso(stats.end_ms),
        "verification": verification,
    }
    reports.write_report(config, "backfill", result)
    ctx.log(
        f"fetched {stats.bars_fetched:,} bars in {stats.elapsed_s:.1f}s; "
        f"{verification['gap_count']} gaps"
    )
    return result


def run_verify(config: AppConfig, ctx) -> dict[str, Any]:
    ctx.message("verifying bars")
    payload = _verify_payload(config)
    reports.write_report(config, "verify", payload)
    ctx.log(f"{payload['rows']:,} rows, {payload['gap_count']} gaps, {len(payload['problems'])} problems")
    return payload


def run_dataset_build(
    config: AppConfig,
    features_cfg: FeatureConfig,
    labels_cfg: LabelConfig,
    out_path: str,
    ctx,
) -> dict[str, Any]:
    started = time.monotonic()
    store = store_from_config(config)

    ctx.message("loading bars from Parquet")
    bars = store.load_bars()
    if bars.empty:
        raise RuntimeError("no bars stored; run a backfill first")
    ctx.log(f"{len(bars):,} bars loaded")
    ctx.progress(0.15, "building features and labels")

    dataset = build_dataset(bars, features_cfg, labels_cfg)
    ctx.progress(0.7, "writing dataset")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    dataset.frame.to_parquet(tmp, index=False, compression=config.data.compression)
    tmp.replace(out)

    split = walk_forward_split(dataset)
    distribution = label_distribution(dataset.frame[["label"]].assign(valid=True))
    result = {
        "bars": int(len(bars)),
        "rows": int(len(dataset)),
        "dropped": int(dataset.frame.attrs.get("dropped_rows", 0)),
        "features": len(dataset.feature_columns),
        "feature_names": dataset.feature_columns,
        "labels": distribution,
        "by_class": _class_stats(dataset.frame),
        "split": {
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "params": {
            "k_up": labels_cfg.k_up,
            "k_down": labels_cfg.k_down,
            "max_holding_bars": labels_cfg.max_holding_bars,
            "atr_period": labels_cfg.atr_period,
            "ambiguity": labels_cfg.ambiguity,
            "feature_lag": features_cfg.feature_lag,
        },
        "path": str(out),
        "elapsed_s": round(time.monotonic() - started, 2),
    }
    reports.write_report(config, "dataset", result)
    ctx.log(f"wrote {len(dataset):,} rows x {len(dataset.feature_columns)} features to {out}")
    return result


def preview_labels(
    config: AppConfig,
    labels_cfg: LabelConfig,
    ctx,
) -> dict[str, Any]:
    store = store_from_config(config)
    bars = store.load_bars()
    if bars.empty:
        raise RuntimeError("no bars stored; run a backfill first")

    ctx.message("labelling bars")
    labels = label_triple_barrier(bars, labels_cfg)
    valid = labels[labels["valid"]]
    distribution = label_distribution(labels)
    stats = _class_stats(valid[["label", "realized_return"]])
    ctx.log(
        f"long={distribution['long']:,} stop={distribution['stop']:,} "
        f"timeout={distribution['flat']:,}"
    )
    return {
        "bars": int(len(bars)),
        "labelled": int(len(valid)),
        "distribution": distribution,
        "by_class": stats,
        "horizon_bars": labels_cfg.max_holding_bars,
        "params": {
            "k_up": labels_cfg.k_up,
            "k_down": labels_cfg.k_down,
            "max_holding_bars": labels_cfg.max_holding_bars,
            "atr_period": labels_cfg.atr_period,
            "ambiguity": labels_cfg.ambiguity,
        },
    }


def run_leakage(
    config: AppConfig,
    features_cfg: FeatureConfig,
    labels_cfg: LabelConfig,
    samples: int,
    ctx,
) -> dict[str, Any]:
    store = store_from_config(config)
    bars = store.load_bars()
    if bars.empty:
        raise RuntimeError("no bars stored; run a backfill first")

    ctx.message("running leakage checks")
    report = run_leakage_suite(bars, features_cfg, labels_cfg, samples=samples)
    payload = report.to_dict()
    payload["params"] = {
        "k_up": labels_cfg.k_up,
        "k_down": labels_cfg.k_down,
        "max_holding_bars": labels_cfg.max_holding_bars,
        "atr_period": labels_cfg.atr_period,
        "ambiguity": labels_cfg.ambiguity,
        "feature_lag": features_cfg.feature_lag,
    }
    reports.write_report(config, "leakage", payload)
    ctx.log("PASS" if report.passed else "FAIL: " + ", ".join(c.name for c in report.failures))
    return payload


def refresh_exchange_info(config: AppConfig, ctx) -> dict[str, Any]:
    ctx.message("fetching exchangeInfo")

    async def fetch() -> dict[str, Any]:
        async with make_client(
            config.exchange.base_url, config.exchange.request_timeout_s
        ) as client:
            return await fetch_exchange_info(
                client,
                symbol=config.exchange.symbol,
                max_retries=config.exchange.max_retries,
            )

    info = asyncio.run(fetch())
    out = config.data_root / "exchange_info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    reports.write_report(config, "exchange", info)

    filters = info.get("filters", {})
    ctx.log(
        "tickSize={} stepSize={} minNotional={}".format(
            filters.get("PRICE_FILTER", {}).get("tickSize"),
            filters.get("LOT_SIZE", {}).get("stepSize"),
            filters.get("NOTIONAL", {}).get("minNotional"),
        )
    )
    return info
