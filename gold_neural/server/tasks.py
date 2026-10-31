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


def run_training(config: AppConfig, params: dict[str, Any], ctx) -> dict[str, Any]:
    """Train a model in a child process, honouring a time budget and Stop."""
    import multiprocessing
    import queue as queue_module

    import pyarrow.parquet as pq

    from .. import training as training_module
    from ..dataset import META_COLUMNS
    from ..registry import new_id, registry_from_config

    dataset_path = config.data_root / "dataset.parquet"
    if not dataset_path.is_file():
        raise RuntimeError("no dataset on disk; build one from the Dataset tab first")

    report = reports.read_report(config, "dataset") or {}
    report_params = report.get("params") or {}
    feature_names = list(report.get("feature_names") or [])
    if not feature_names:
        schema = pq.read_schema(dataset_path)
        feature_names = [name for name in schema.names if name not in META_COLUMNS]
    if not feature_names:
        raise RuntimeError("dataset has no feature columns")

    window = int(params.get("window", 64))
    train_config = training_module.TrainConfig(
        window=window,
        epochs=int(params.get("epochs", 50)),
        batch_size=int(params.get("batch_size", 256)),
        lr=float(params.get("lr", 3e-4)),
        patience=int(params.get("patience", 5)),
        seed=int(params.get("seed", 7)),
        max_minutes=float(params["minutes"]) if params.get("minutes") else None,
        device=str(params.get("device", "auto")),
    )

    registry = registry_from_config(config)
    model_id = new_id("model")
    model_name = (
        f"{config.exchange.symbol}-{config.exchange.interval}-w{window}-"
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    label_config = {
        key: report_params.get(key)
        for key in ("k_up", "k_down", "max_holding_bars", "atr_period", "ambiguity")
        if report_params.get(key) is not None
    }
    registry.create_model(
        model_id=model_id,
        name=model_name,
        interval=config.exchange.interval,
        architecture="transformer_v1",
        window=window,
        label_config=label_config,
        feature_config={"feature_lag": report_params.get("feature_lag")},
        status="training",
    )

    checkpoint_dir = config.data_root / "checkpoints" / model_id
    started_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    budget = train_config.max_minutes
    budget_text = f"{budget:g} min" if budget else "until stopped"
    ctx.log(
        f"model {model_id} · window {window} · {len(feature_names)} features · "
        f"budget {budget_text}"
    )

    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    progress_queue = context.Queue()
    ctx.on_stop(stop_event.set)

    process = context.Process(
        target=training_module.train_process_entry,
        args=(
            {
                "window": train_config.window,
                "epochs": train_config.epochs,
                "batch_size": train_config.batch_size,
                "lr": train_config.lr,
                "weight_decay": train_config.weight_decay,
                "dropout": train_config.dropout,
                "d_model": train_config.d_model,
                "nhead": train_config.nhead,
                "layers": train_config.layers,
                "dim_feedforward": train_config.dim_feedforward,
                "patience": train_config.patience,
                "max_minutes": train_config.max_minutes,
                "seed": train_config.seed,
                "num_threads": train_config.num_threads,
                "device": train_config.device,
                "train_frac": train_config.train_frac,
                "val_frac": train_config.val_frac,
            },
            str(dataset_path),
            feature_names,
            str(checkpoint_dir),
            progress_queue,
            stop_event,
        ),
        name=f"training-{model_id}",
        daemon=True,
    )
    process.start()

    result_payload: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None
    stop_requested_at: float | None = None

    while True:
        try:
            message = progress_queue.get(timeout=0.4)
        except queue_module.Empty:
            if not process.is_alive():
                break
            if ctx.should_stop() and not stop_event.is_set():
                stop_event.set()
            if stop_event.is_set() and stop_requested_at is None:
                stop_requested_at = time.monotonic()
            if stop_requested_at and time.monotonic() - stop_requested_at > 120:
                ctx.log("training ignored the stop request for 120s; terminating")
                process.terminate()
            continue

        kind = message.get("type")
        if kind == "progress":
            total = max(1, int(message.get("epochs_total", 1)))
            ctx.update(
                message=message.get("message") or "training",
                progress=min(0.99, int(message.get("epoch", 0)) / total),
            )
        elif kind == "result":
            result_payload = message.get("result")
        elif kind == "error":
            error_payload = message

    process.join(timeout=30)
    if process.is_alive():  # pragma: no cover - only on a wedged child
        process.terminate()
        process.join(timeout=10)

    if error_payload is not None:
        registry.update_model(model_id, status="failed")
        registry.record_run(
            model_id=model_id,
            started_at=started_at,
            status="failed",
            config={"window": window, "epochs": train_config.epochs},
            stop_reason="error",
        )
        trace = (error_payload.get("trace") or "").strip().splitlines()
        raise RuntimeError(
            error_payload.get("error", "training failed")
            + ("\n" + "\n".join(trace[-6:]) if trace else "")
        )

    if result_payload is None:
        registry.update_model(model_id, status="failed")
        raise RuntimeError("training process exited without returning a result")

    metrics = result_payload.get("metrics") or {}
    trained = result_payload.get("checkpoint_path") is not None
    status = (
        "candidate"
        if result_payload.get("status") == "ok" and metrics.get("beats_baseline") and trained
        else "draft"
    )
    if trained:
        registry.update_model(
            model_id,
            status=status,
            checkpoint_path=result_payload["checkpoint_path"],
            checkpoint_version=int(result_payload.get("checkpoint_version") or 1),
            metrics=metrics,
        )
    else:
        registry.update_model(model_id, status="draft")

    run_id = registry.record_run(
        model_id=model_id,
        started_at=started_at,
        status=result_payload.get("status", "unknown"),
        config={"window": window, "epochs": train_config.epochs,
                "batch_size": train_config.batch_size, "lr": train_config.lr,
                "max_minutes": train_config.max_minutes, "device": result_payload.get("device")},
        final_metrics=metrics,
        per_class_metrics=metrics.get("per_class"),
        confusion_matrix=metrics.get("confusion_matrix"),
        epochs_done=result_payload.get("epochs_done"),
        stop_reason=result_payload.get("stop_reason"),
        duration_s=result_payload.get("duration_s"),
        device=result_payload.get("device"),
    )

    ctx.log(
        f"{result_payload.get('stop_reason')} after "
        f"{result_payload.get('epochs_done')} epochs in "
        f"{result_payload.get('duration_s')}s on {result_payload.get('device')}"
    )
    ctx.log(
        f"balanced_accuracy {metrics.get('balanced_accuracy')} vs baseline "
        f"{metrics.get('baseline_accuracy')} · signal_edge {metrics.get('signal_edge')}"
    )

    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_status": status,
        "run_id": run_id,
        **result_payload,
    }
