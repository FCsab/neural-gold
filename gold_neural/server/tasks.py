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
from dataclasses import replace
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


def train_child(
    train_config,
    dataset_path: Path,
    feature_names: list[str],
    checkpoint_dir: Path,
    ctx,
    *,
    prefix: str = "",
    manage_progress: bool = True,
) -> dict[str, Any]:
    """Run one training job in a spawned child and return its result payload."""

    import multiprocessing
    import queue as queue_module

    from .. import training as training_module

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
            list(feature_names),
            str(checkpoint_dir),
            progress_queue,
            stop_event,
        ),
        name="training-child",
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
            text = message.get("message") or "training"
            if manage_progress:
                total = int(message.get("epochs_total") or 0)
                if total > 0:
                    ctx.update(
                        message=text,
                        progress=min(0.99, int(message.get("epoch", 0)) / total),
                    )
                else:
                    # Open-ended run: keep the bar in its indeterminate state
                    # rather than parking it at 99% forever.
                    ctx.message(text)
            elif prefix:
                ctx.message(f"{prefix} · {text}")
        elif kind == "result":
            result_payload = message.get("result")
        elif kind == "error":
            error_payload = message

    process.join(timeout=30)
    if process.is_alive():  # pragma: no cover - only on a wedged child
        process.terminate()
        process.join(timeout=10)

    if error_payload is not None:
        trace = (error_payload.get("trace") or "").strip().splitlines()
        raise RuntimeError(
            error_payload.get("error", "training failed")
            + ("\n" + "\n".join(trace[-6:]) if trace else "")
        )
    if result_payload is None:
        raise RuntimeError("training process exited without returning a result")
    return result_payload


def run_training(config: AppConfig, params: dict[str, Any], ctx) -> dict[str, Any]:
    """Train a model in a child process, honouring a time budget and Stop."""
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
        epochs=int(params.get("epochs", 0)),
        batch_size=int(params.get("batch_size", 256)),
        lr=float(params.get("lr", 3e-4)),
        dropout=float(params.get("dropout", 0.1)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        patience=int(params.get("patience", 0)),
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

    try:
        result_payload = train_child(
            train_config,
            dataset_path,
            feature_names,
            checkpoint_dir,
            ctx,
            prefix=f"model {model_id}",
        )
    except Exception:
        registry.update_model(model_id, status="failed")
        registry.record_run(
            model_id=model_id,
            started_at=started_at,
            status="failed",
            config={"window": window, "epochs": train_config.epochs},
            stop_reason="error",
        )
        raise

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
        f"balanced_accuracy {metrics.get('balanced_accuracy')} vs constant guess "
        f"{metrics.get('baseline_balanced_accuracy')} "
        f"(plain accuracy {metrics.get('accuracy')} vs "
        f"{metrics.get('baseline_accuracy')}) · signal_edge {metrics.get('signal_edge')}"
    )

    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_status": status,
        "run_id": run_id,
        **result_payload,
    }


def run_backtest(config: AppConfig, params: dict[str, Any], ctx) -> dict[str, Any]:
    """Score the test slice with a trained checkpoint and simulate the trades."""
    import numpy as np

    from .. import training as training_module
    from ..backtest import (
        BacktestConfig,
        open_price_lookup,
        prepare_backtest_frame,
        run_backtest as simulate,
        verdict,
    )
    from ..registry import registry_from_config

    registry = registry_from_config(config)
    model_id = params.get("model_id")
    if model_id:
        model = registry.get_model(model_id)
    else:
        model = next(
            (m for m in registry.list_models(limit=50) if m.get("checkpoint_path")),
            None,
        )
    if model is None:
        raise RuntimeError("no trained model with a checkpoint; train one first")

    checkpoint_path = Path(model["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise RuntimeError(f"checkpoint for {model['name']} is missing on disk")

    dataset_path = config.data_root / "dataset.parquet"
    if not dataset_path.is_file():
        raise RuntimeError("no dataset on disk; build one from the Labels tab")

    ctx.message("loading checkpoint and dataset")
    checkpoint = training_module.load_checkpoint(checkpoint_path)
    frame = pd.read_parquet(dataset_path)
    train_config = checkpoint.get("config", {})
    window = int(checkpoint.get("window", 64))
    test_start = int(
        len(frame)
        * (float(train_config.get("train_frac", 0.70)) + float(train_config.get("val_frac", 0.15)))
    )
    rows = np.arange(max(test_start, window - 1), len(frame))
    if len(rows) < 50:
        raise RuntimeError("the test slice is too small to backtest")

    ctx.progress(0.25, f"scoring {len(rows):,} rows with {model['name']}")
    probabilities = training_module.predict_proba(checkpoint, frame, rows)
    prepared = prepare_backtest_frame(frame, rows, open_price_lookup(config))

    backtest_config = BacktestConfig(
        initial_equity=float(params.get("initial_equity", 10_000.0)),
        fee_bps=float(params.get("fee_bps", 10.0)),
        slippage_bps=float(params.get("slippage_bps", 2.0)),
        min_confidence=float(params.get("min_confidence", 0.0)),
    )

    ctx.progress(0.7, "simulating trades")
    result = simulate(prepared, probabilities, backtest_config)
    payload = result.to_dict()
    payload["verdict"] = verdict(result.metrics)
    payload["model"] = {
        "id": model["id"],
        "name": model["name"],
        "status": model["status"],
    }
    payload["params"] = {
        "initial_equity": backtest_config.initial_equity,
        "fee_bps": backtest_config.fee_bps,
        "slippage_bps": backtest_config.slippage_bps,
        "min_confidence": backtest_config.min_confidence,
    }
    payload["slice"] = {
        "rows": int(len(rows)),
        "first_ms": int(prepared["open_time"].iloc[0]),
        "last_ms": int(prepared["open_time"].iloc[-1]),
    }
    reports.write_report(config, "backtest", payload)

    metrics = result.metrics
    ctx.log(
        f"{metrics['trades']} trades · {metrics['total_return'] * 100:+.2f}% vs "
        f"buy-and-hold {metrics['buy_hold_return'] * 100:+.2f}%"
    )
    return payload


def _downsample(points: list[dict[str, float]], limit: int) -> list[dict[str, float]]:
    if len(points) <= limit:
        return points
    step = max(1, len(points) // limit)
    return points[::step] + [points[-1]]


def run_search(config: AppConfig, params: dict[str, Any], ctx) -> dict[str, Any]:
    """Sweep label settings, train each one, rank them on the validation slice."""
    import numpy as np

    from .. import training as training_module
    from ..backtest import (
        BacktestConfig,
        open_price_lookup,
        prepare_backtest_frame,
        run_backtest as simulate,
        verdict,
    )
    from ..dataset import build_dataset
    from ..indicators import atr as atr_indicator
    from ..registry import new_id, registry_from_config
    from ..search import (
        SearchConfig,
        generate_candidates,
        refine_candidates,
        score,
    )

    search_config = SearchConfig(
        total_minutes=float(params.get("total_minutes", 180.0)),
        per_candidate_minutes=float(params.get("per_candidate_minutes", 4.0)),
        max_candidates=int(params.get("max_candidates", 40)),
        epochs=int(params.get("epochs", 400)),
        batch_size=int(params.get("batch_size", 512)),
        lr=float(params.get("lr", 3e-4)),
        device=str(params.get("device", "auto")),
        fee_bps=float(params.get("fee_bps", 10.0)),
        slippage_bps=float(params.get("slippage_bps", 2.0)),
        refine_top=int(params.get("refine_top", 2)),
    )

    store = store_from_config(config)
    bars = store.load_bars()
    if bars.empty:
        raise RuntimeError("no bars stored; run a backfill first")

    atr_series = atr_indicator(bars["high"], bars["low"], bars["close"], 14)
    atr_bps = float((atr_series / bars["close"]).dropna().median() * 1e4)

    candidates = generate_candidates(atr_bps, search_config)
    if not candidates:
        raise RuntimeError(
            "no label setting survives the cost filter: the profit target must be at "
            f"least {search_config.min_target_ratio:g}x the "
            f"{search_config.round_trip_cost_bps:.0f} bps round trip"
        )

    registry = registry_from_config(config)
    search_id = new_id("search")
    open_by_time = open_price_lookup(config)
    started = time.monotonic()
    deadline = started + search_config.total_minutes * 60.0
    started_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    backtest_config = BacktestConfig(
        fee_bps=search_config.fee_bps,
        slippage_bps=search_config.slippage_bps,
        initial_equity=float(params.get("initial_equity", 10_000.0)),
    )
    results: list[dict[str, Any]] = []
    # Grows when the refinement pass adds candidates, so the UI never shows
    # "5 / 2 candidates".
    total_planned = len(candidates)

    def json_row(row: dict[str, Any]) -> dict[str, Any]:
        out = {key: value for key, value in row.items() if key != "_curve"}
        score_value = out.get("score")
        # A candidate below the trade floor scores -inf, which is not valid
        # JSON and would break the browser. Rank on the float, publish null.
        if score_value is not None and not math.isfinite(score_value):
            out["score"] = None
        return out

    def publish(status: str = "running") -> dict[str, Any]:
        ranked = sorted(
            [row for row in results if row.get("status") == "ok"],
            key=lambda row: row["score"],
            reverse=True,
        )
        for index, row in enumerate(ranked, start=1):
            row["rank"] = index
        best = ranked[0] if ranked else None
        payload = {
            "search_id": search_id,
            "status": status,
            "started_at": started_at,
            "finished_at": (
                datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
                if status != "running"
                else None
            ),
            "atr_bps": round(atr_bps, 2),
            "round_trip_cost_bps": search_config.round_trip_cost_bps,
            "elapsed_minutes": round((time.monotonic() - started) / 60.0, 1),
            "candidates_total": total_planned,
            "candidates_done": len(results),
            "rows": [json_row(row) for row in results],
            "leaderboard": [json_row(row) for row in ranked[:10]],
            "best": json_row(best) if best else None,
            "best_curve": _downsample(best.get("_curve", []), 400) if best else [],
            "params": {
                "total_minutes": search_config.total_minutes,
                "per_candidate_minutes": search_config.per_candidate_minutes,
                "max_candidates": search_config.max_candidates,
                "fee_bps": search_config.fee_bps,
                "slippage_bps": search_config.slippage_bps,
                "device": search_config.device,
            },
        }
        reports.write_report(config, "search", payload)
        return payload

    def run_candidate(candidate, index: int, total: int) -> dict[str, Any]:
        label_config = replace(
            config.labels,
            k_up=candidate.k_up,
            k_down=candidate.k_down,
            max_holding_bars=candidate.holding,
        )
        prefix = f"c{index}/{total} · {candidate.key()}"
        ctx.message(f"candidate {index}/{total} · {candidate.label()} · labelling")

        try:
            dataset = build_dataset(bars, config.features, label_config)
        except Exception as exc:  # noqa: BLE001 - a bad label config must not kill the sweep
            return {
                "status": "failed",
                "candidate": candidate.to_dict(),
                "error": f"dataset: {exc}",
                "score": float("-inf"),
            }

        dataset_path = config.data_root / "search" / "candidate.parquet"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.frame.to_parquet(
            dataset_path, index=False, compression=config.data.compression
        )

        model_id = new_id("model")
        model_name = f"search-{search_id[-6:]}-{candidate.key()}"
        checkpoint_dir = config.data_root / "checkpoints" / model_id
        registry.create_model(
            model_id=model_id,
            name=model_name,
            interval=config.exchange.interval,
            architecture="transformer_v1",
            window=candidate.window,
            label_config={
                "k_up": candidate.k_up,
                "k_down": candidate.k_down,
                "max_holding_bars": candidate.holding,
                "atr_period": config.labels.atr_period,
                "ambiguity": config.labels.ambiguity,
            },
            feature_config={"feature_lag": config.features.feature_lag},
            status="training",
        )
        run_started = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        train_config = training_module.TrainConfig(
            window=candidate.window,
            epochs=search_config.epochs,
            batch_size=search_config.batch_size,
            lr=search_config.lr,
            patience=search_config.patience,
            max_minutes=search_config.per_candidate_minutes,
            device=search_config.device,
            seed=search_config.seed,
        )

        try:
            payload = train_child(
                train_config,
                dataset_path,
                dataset.feature_columns,
                checkpoint_dir,
                ctx,
                prefix=prefix,
                manage_progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            registry.update_model(model_id, status="failed")
            registry.record_run(
                model_id=model_id,
                started_at=run_started,
                status="failed",
                config={"window": candidate.window},
                stop_reason="error",
            )
            return {
                "status": "failed",
                "candidate": candidate.to_dict(),
                "model_id": model_id,
                "error": str(exc)[:300],
                "score": float("-inf"),
            }

        if not payload.get("checkpoint_path"):
            registry.update_model(model_id, status="draft")
            return {
                "status": "skipped",
                "candidate": candidate.to_dict(),
                "model_id": model_id,
                "error": "no checkpoint: stopped before the first epoch",
                "score": float("-inf"),
            }

        ctx.message(f"candidate {index}/{total} · {candidate.key()} · backtesting")
        checkpoint = training_module.load_checkpoint(payload["checkpoint_path"])
        frame = dataset.frame.reset_index(drop=True)
        total_rows = len(frame)
        first = candidate.window - 1
        val_start = int(total_rows * train_config.train_frac)
        val_end = int(total_rows * (train_config.train_frac + train_config.val_frac))
        val_rows = np.arange(max(val_start, first), val_end)
        test_rows = np.arange(max(val_end, first), total_rows)

        validation = simulate(
            prepare_backtest_frame(frame, val_rows, open_by_time),
            training_module.predict_proba(checkpoint, frame, val_rows),
            backtest_config,
        )
        test = simulate(
            prepare_backtest_frame(frame, test_rows, open_by_time),
            training_module.predict_proba(checkpoint, frame, test_rows),
            backtest_config,
        )

        row = {
            "status": "ok",
            "candidate": candidate.to_dict(),
            "model_id": model_id,
            "model_name": model_name,
            "score": score({"validation": {"metrics": validation.metrics}}, search_config.min_trades),
            "train": {
                "epochs": payload.get("epochs_done"),
                "stop_reason": payload.get("stop_reason"),
                "duration_s": payload.get("duration_s"),
                "device": payload.get("device"),
            },
            "validation": {
                "metrics": validation.metrics,
                "verdict": verdict(validation.metrics),
            },
            "test": {"metrics": test.metrics, "verdict": verdict(test.metrics)},
            "_curve": validation.equity_curve,
        }

        metrics = payload.get("metrics") or {}
        status = (
            "candidate"
            if row["score"] > float("-inf") and metrics.get("beats_baseline")
            else "draft"
        )
        registry.update_model(
            model_id,
            status=status,
            checkpoint_path=payload["checkpoint_path"],
            checkpoint_version=int(payload.get("checkpoint_version") or 1),
            metrics=metrics,
        )
        registry.record_run(
            model_id=model_id,
            started_at=run_started,
            status="ok",
            config={
                "window": candidate.window,
                "k_up": candidate.k_up,
                "k_down": candidate.k_down,
                "holding": candidate.holding,
                "max_minutes": search_config.per_candidate_minutes,
            },
            final_metrics=metrics,
            per_class_metrics=metrics.get("per_class"),
            confusion_matrix=metrics.get("confusion_matrix"),
            epochs_done=payload.get("epochs_done"),
            stop_reason=payload.get("stop_reason"),
            duration_s=payload.get("duration_s"),
            device=payload.get("device"),
        )
        return row

    def run_batch(batch, total: int, offset: int = 0) -> bool:
        for index, candidate in enumerate(batch, start=offset + 1):
            if ctx.should_stop():
                ctx.log("stop requested")
                return False
            if time.monotonic() > deadline:
                ctx.log("time budget exhausted")
                return False
            results.append(run_candidate(candidate, index, total))
            publish()
            ctx.update(
                progress=min(
                    0.99, (time.monotonic() - started) / (search_config.total_minutes * 60.0)
                )
            )
        return True

    ctx.log(
        f"{len(candidates)} candidates · ATR {atr_bps:.1f} bps · "
        f"round trip {search_config.round_trip_cost_bps:.0f} bps · "
        f"budget {search_config.total_minutes:g} min"
    )
    publish()

    completed = run_batch(candidates, len(candidates))

    if completed and search_config.refine_top > 0:
        ranked = sorted(
            [row for row in results if row.get("status") == "ok"],
            key=lambda row: row["score"],
            reverse=True,
        )
        extra = refine_candidates([{"candidate": row["candidate"]} for row in ranked], search_config)
        remaining = deadline - time.monotonic()
        # Only refine when there is real time left; on a short sweep this would
        # otherwise double the run past the budget the user asked for.
        if extra and remaining > 2 * search_config.per_candidate_minutes * 60:
            total_planned += len(extra)
            ctx.log(f"refining {len(extra)} candidates with different windows")
            run_batch(extra, total_planned, offset=len(candidates))
        elif extra:
            ctx.log("skipping the window refinement: not enough time left")

    final = publish("done" if completed and not ctx.should_stop() else "stopped")
    best = final.get("best")
    if best:
        ctx.log(
            f"best: {best['candidate']['key']} · val {best['validation']['metrics']['total_return'] * 100:+.2f}% "
            f"· test {best['test']['metrics']['total_return'] * 100:+.2f}%"
        )
    else:
        ctx.log("no candidate produced a usable model")
    return final


def run_walkforward(config: AppConfig, params: dict[str, Any], ctx) -> dict[str, Any]:
    """Run the walk-forward sweep as a web job.

    Overrides the interval locally rather than requiring a restart: the 4h
    bars live beside the 5m ones, so the browser can ask for either.
    """
    from ..backtest import BacktestConfig
    from ..walkforward import DEFAULT_STOP_THRESHOLDS, SweepConfig, run_sweep

    interval = str(params.get("interval") or config.exchange.interval)
    local = replace(config, exchange=replace(config.exchange, interval=interval))

    store = store_from_config(local)
    if store.coverage().rows < 2000:
        ctx.log(f"no {interval} bars stored; downloading 6 years")
        stats = asyncio.run(backfill(local, years=float(params.get("years", 6.2))))
        ctx.log(f"backfill: {stats.format()}")

    bars = store.load_bars()
    if bars.empty:
        raise RuntimeError(f"no {interval} bars available")

    bars_per_day = 86_400_000 // interval_to_ms(interval)
    specs = [
        {
            "k_up": float(k_up),
            "k_down": float(k_down),
            "holding": int(holding),
            "window": int(window),
            "seed": int(seed),
        }
        for seed in params["seeds"]
        for k_up in params["k_up"]
        for k_down in params["k_down"]
        for holding in params["holding"]
        for window in params["window"]
    ]
    if not specs:
        raise RuntimeError("the sweep has no configurations")

    ctx.log(
        f"{len(specs)} settings · {interval} · test {params['test_days']}d "
        f"· seeds {params['seeds']}"
    )

    def on_status(status: dict[str, Any]) -> None:
        total = max(1, status.get("specs_total") or 1)
        done = status.get("specs_done") or 0
        ctx.update(
            message=status.get("current") or "running",
            progress=min(0.99, done / total),
        )

    out = config.data_root / "walkforward"
    final = run_sweep(
        config=local,
        bars=bars,
        specs=specs,
        sweep=SweepConfig(
            test_bars=bars_per_day * int(params["test_days"]),
            val_bars=bars_per_day * int(params["val_days"]),
            min_train_bars=bars_per_day * int(params["min_train_days"]),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            lr=float(params["lr"]),
            minutes_per_fold=float(params["minutes_per_fold"]),
            device=str(params["device"]),
            policy=str(params.get("policy", "signal")),
            stop_thresholds=(
                DEFAULT_STOP_THRESHOLDS
                if params.get("policy") == "hold"
                else ()
            ),
            reentry_margin=0.05,
            min_flat_bars=1,
        ),
        backtest_config=BacktestConfig(
            fee_bps=float(params["fee_bps"]),
            slippage_bps=float(params["slippage_bps"]),
        ),
        out_dir=out,
        on_status=on_status,
        should_stop=ctx.should_stop,
        log=ctx.log,
    )

    best = (final.get("leaderboard") or [None])[0]
    if best:
        ctx.log(
            f"best {best['spec']}: median {best['median_return'] * 100:+.2f}% "
            f"· worst {best['worst_return'] * 100:+.2f}% "
            f"· beats hold {best['share_beating_hold'] * 100:.0f}%"
        )
    return final
