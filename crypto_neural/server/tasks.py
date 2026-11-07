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
from ..data.backfill import backfill, top_up, verify_bars
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


def _no_training_bars(market: AppConfig) -> str:
    symbol = market.exchange.symbol
    return (
        f"no {symbol} {market.exchange.interval} bars stored; download them once with: "
        f"CRYPTONEURAL_EXCHANGE_SYMBOL={symbol} crypto-neural backfill --start 2017-08-17"
    )


def run_dataset_build(
    config: AppConfig,
    features_cfg: FeatureConfig,
    labels_cfg: LabelConfig,
    out_path: str,
    ctx,
) -> dict[str, Any]:
    started = time.monotonic()
    market = config.training_market()
    if market.exchange.symbol != config.exchange.symbol:
        # Nothing else keeps the training series current: "Update data"
        # downloads the traded pair only.
        ctx.message(f"fetching the newest {market.exchange.symbol} bars")
        top_up_bars(market, ctx.log)
    store = store_from_config(market)

    ctx.message("loading bars from Parquet")
    bars = store.load_bars()
    if bars.empty:
        raise RuntimeError(_no_training_bars(market))
    ctx.log(f"{len(bars):,} {market.exchange.symbol} bars loaded")
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
        "symbol": market.exchange.symbol,
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
    market = config.training_market()
    bars = store_from_config(market).load_bars()
    if bars.empty:
        raise RuntimeError(_no_training_bars(market))

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
    # The suite checks the series the model actually learns from.
    market = config.training_market()
    bars = store_from_config(market).load_bars()
    if bars.empty:
        raise RuntimeError(_no_training_bars(market))

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


# What the Train tab's live chart needs from each epoch; the rest of the
# progress message is text for the job bar.
SERIES_KEYS = (
    "epoch", "train_loss", "val_loss", "val_balanced_accuracy",
    "baseline_balanced_accuracy", "best_epoch", "elapsed_s",
)
# Epochs without a new validation low before an attempt is given up and a
# fresh one starts (``restarts``). Matches the live card's "memorising" line.
RESTART_PATIENCE = 10


def training_patience(params: dict[str, Any]) -> int:
    """Patience 0 means "never stall"; with restarts on, an attempt that
    never stalls would never be replaced, so the restart patience applies."""
    patience = int(params.get("patience") or 0)
    return patience or (RESTART_PATIENCE if params.get("restarts") else 0)


def train_child(
    train_config,
    dataset_path: Path,
    feature_names: list[str],
    checkpoint_dir: Path,
    ctx,
    *,
    prefix: str = "",
    manage_progress: bool = True,
    attempt: int = 0,
    epoch_offset: int = 0,
) -> dict[str, Any]:
    """Run one training job in a spawned child and return its result payload.

    ``attempt`` and ``epoch_offset`` place a restarted attempt after the
    earlier ones in the job's live series, whose epochs must keep rising.
    """

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
            point = {key: message.get(key) for key in SERIES_KEYS}
            if attempt:
                text = f"attempt {attempt} · {text}"
                point["attempt"] = attempt
                for key in ("epoch", "best_epoch"):
                    if point.get(key) is not None:
                        point[key] = float(point[key]) + epoch_offset
            ctx.point(point)
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


def _train_with_restarts(
    train_config, dataset_path: Path, feature_names: list[str], checkpoint_dir: Path, ctx
) -> dict[str, Any]:
    """Train, and start over with a new seed whenever an attempt stalls.

    An attempt ends when its validation loss has not reached a new low for
    ``train_config.patience`` epochs. While time is left and nobody pressed
    Stop, a fresh model with the next seed takes over. The attempt kept is the
    one with the lowest *validation* loss: the test slice plays no part in the
    choice, and only the kept attempt's test metrics are ever reported, so
    the backtest that follows is still on data nothing was chosen by.

    Choosing the best of many seeds on one validation slice flatters that
    slice a little; the attempt count is recorded in the metrics so a reader
    can weigh it.
    """
    import shutil

    budget = train_config.max_minutes
    started = time.monotonic()
    best: dict[str, Any] | None = None
    best_dir: Path | None = None
    chosen = 0
    summary: list[dict[str, Any]] = []
    offset = 0
    attempt = 0
    while True:
        attempt += 1
        remaining = None
        if budget:
            remaining = budget - (time.monotonic() - started) / 60.0
            if remaining < 0.1:
                break
        attempt_dir = checkpoint_dir / f"attempt-{attempt}"
        config = replace(
            train_config, seed=train_config.seed + attempt - 1, max_minutes=remaining
        )
        payload = train_child(
            config, dataset_path, feature_names, attempt_dir, ctx,
            attempt=attempt, epoch_offset=offset,
        )
        offset += int(payload.get("epochs_done") or 0)
        val_loss = (payload.get("metrics") or {}).get("best_val_loss")
        summary.append({
            "attempt": attempt,
            "seed": config.seed,
            "epochs": int(payload.get("epochs_done") or 0),
            "best_val_loss": val_loss,
            "stop_reason": payload.get("stop_reason"),
        })
        ctx.log(
            f"attempt {attempt} (seed {config.seed}): {payload.get('stop_reason')} after "
            f"{payload.get('epochs_done')} epochs, best validation loss {val_loss}"
        )
        usable = payload.get("checkpoint_path") is not None and val_loss is not None
        best_val = (best or {}).get("metrics", {}).get("best_val_loss")
        if usable and (best is None or val_loss < best_val):
            if best_dir is not None:
                shutil.rmtree(best_dir, ignore_errors=True)
            best, best_dir, chosen = payload, attempt_dir, attempt
        else:
            shutil.rmtree(attempt_dir, ignore_errors=True)
        if payload.get("stop_reason") != "early_stopping" or ctx.should_stop():
            break

    if best is None:
        if summary:
            # Nothing finished an epoch; hand back the last attempt's
            # "no checkpoint" result so the caller records a draft.
            return {**payload, "attempts": summary}
        raise RuntimeError("the time budget ran out before the first attempt started")
    ctx.log(f"kept attempt {chosen} of {len(summary)}, chosen by validation loss only")
    metrics = {**best["metrics"], "attempts": len(summary), "chosen_attempt": chosen}
    return {
        **best,
        "metrics": metrics,
        "attempts": summary,
        "epochs_done": offset,
        "duration_s": round(time.monotonic() - started, 2),
        # The run as a whole ended for the last attempt's reason (time, Stop);
        # the status is the kept model's own: did *it* run to convergence.
        "stop_reason": payload.get("stop_reason"),
        "status": best.get("status"),
    }


def run_training(config: AppConfig, params: dict[str, Any], ctx) -> dict[str, Any]:
    """Train a model in a child process, honouring a time budget and Stop."""
    import pyarrow.parquet as pq

    from .. import training as training_module
    from ..dataset import META_COLUMNS
    from ..registry import new_id, registry_from_config

    dataset_path = config.data_root / "dataset.parquet"
    if not dataset_path.is_file():
        raise RuntimeError("no dataset on disk; press “Train and test” in the web UI, or run `cli dataset`")

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
        patience=training_patience(params),
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
        if params.get("restarts"):
            result_payload = _train_with_restarts(
                train_config, dataset_path, feature_names, checkpoint_dir, ctx
            )
        else:
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
    # Entries fill at the next open of the series the dataset was built from.
    prepared = prepare_backtest_frame(frame, rows, open_price_lookup(config.training_market()))

    backtest_config = BacktestConfig(
        initial_equity=float(params.get("initial_equity", 10_000.0)),
        fee_bps=float(params.get("fee_bps", 10.0)),
        slippage_bps=float(params.get("slippage_bps", 2.0)),
        min_confidence=float(params.get("min_confidence", 0.0)),
        entry_rule=str(params.get("entry_rule", "expected_value")),
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
        "entry_rule": backtest_config.entry_rule,
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


def _risk_in_child(bars, risk_cfg, features, bars_per_year: float, ctx):
    """Run the walk-forward in a spawned process; relay progress and Stop."""
    import multiprocessing
    import queue as queue_module

    from ..risk import RiskReport, risk_process_entry

    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    messages = context.Queue()
    ctx.on_stop(stop_event.set)
    process = context.Process(
        target=risk_process_entry,
        args=(bars, risk_cfg, features, bars_per_year, messages, stop_event),
        name="risk-child",
        daemon=True,
    )
    process.start()
    outcome: dict[str, Any] | None = None
    while outcome is None:
        try:
            message = messages.get(timeout=0.5)
        except queue_module.Empty:
            if not process.is_alive():
                break
            continue
        if message["type"] == "progress":
            ctx.update(message=message["message"], progress=message["fraction"])
        else:
            outcome = message
    process.join(timeout=30)
    if process.is_alive():  # pragma: no cover - only on a wedged child
        process.terminate()

    if outcome is None:
        raise RuntimeError("the risk process exited without a result")
    if outcome["type"] == "stopped":
        return None
    if outcome["type"] == "error":
        trace = (outcome.get("trace") or "").strip().splitlines()
        raise RuntimeError(outcome["error"] + ("\n" + "\n".join(trace[-6:]) if trace else ""))
    return RiskReport(**outcome["report"])


def run_risk(config: AppConfig, params: dict[str, Any], ctx) -> dict[str, Any]:
    """The crash-risk walk-forward as a web job.

    Learns from ``risk.data_symbol`` bars at ``risk.interval`` (BTCUSDT 4h by
    default: continuous since 2017), downloading them on first use. Models
    with a neural part run in a child process.
    """
    from ..risk import Stopped, run_risk_walkforward

    risk_cfg = config.risk
    bars_per_day = 86_400_000 // interval_to_ms(risk_cfg.interval)
    if params.get("drawdown") is not None:
        risk_cfg = replace(risk_cfg, drawdown=float(params["drawdown"]))
    if params.get("horizon_days") is not None:
        risk_cfg = replace(risk_cfg, horizon_bars=int(params["horizon_days"]) * bars_per_day)

    local = replace(
        config,
        exchange=replace(
            config.exchange, symbol=risk_cfg.data_symbol, interval=risk_cfg.interval
        ),
    )
    store = store_from_config(local)
    if store.coverage().rows < risk_cfg.min_train_bars + 4 * risk_cfg.test_bars:
        ctx.log(f"not enough {risk_cfg.data_symbol} {risk_cfg.interval} bars; downloading 9.5 years")
        stats = asyncio.run(backfill(local, years=9.5))
        ctx.log(f"backfill: {stats.format()}")
    elif params.get("refresh", True):
        top_up_bars(local, ctx.log)
    bars = store.load_bars()
    if bars.empty:
        raise RuntimeError(f"no {risk_cfg.data_symbol} {risk_cfg.interval} bars available")

    ctx.log(
        f"{len(bars):,} {risk_cfg.data_symbol} {risk_cfg.interval} bars · model "
        f"{risk_cfg.model} · watching for a {risk_cfg.drawdown:.0%} fall within "
        f"{risk_cfg.horizon_bars / bars_per_day:g} days"
    )
    bars_per_year = 365.0 * bars_per_day
    if risk_cfg.model == "lgbm":
        try:
            report = run_risk_walkforward(
                bars,
                risk_cfg,
                config.features,
                bars_per_year=bars_per_year,
                progress=lambda fraction, message: ctx.update(message=message, progress=fraction),
                should_stop=ctx.should_stop,
            )
        except Stopped:
            report = None
    else:
        report = _risk_in_child(bars, risk_cfg, config.features, bars_per_year, ctx)
    if report is None:
        ctx.log("stopped before the last block; no report written")
        return {"stopped": True}

    payload = report.to_dict()
    reports.write_report(config, "risk", payload)
    for line in report.format().splitlines():
        ctx.log(line)
    # Only the configured model goes in the log. A one-off run with another
    # drawdown is an experiment; splicing it into the series would score a
    # predictor that was never run forward.
    if risk_cfg == config.risk:
        recorded = record_reading(config, payload, risk_cfg, source="server", log=ctx.log)
    else:
        recorded = False
        ctx.log("settings differ from config: reading not added to the predictions log")
    return {
        "status": report.status, "verdict": report.verdict, "today": report.today,
        "recorded": recorded,
    }


def top_up_bars(config: AppConfig, log) -> None:
    """Fetch bars newer than the store's last one; on failure, say so and
    carry on with what is stored (the reading's bar time shows its age)."""
    try:
        stats = asyncio.run(top_up(config))
    except Exception as exc:  # noqa: BLE001 - network boundary: degrade, don't fail the job
        log(f"could not fetch the newest bars ({type(exc).__name__}: {exc}); using stored bars")
        return
    if stats is None:
        log("stored bars are too old to top up; download history on the Overview page")
    elif stats.bars_written:
        log(f"fetched {stats.bars_written} new bar(s)")


def record_reading(
    config: AppConfig, payload: dict[str, Any], risk_cfg, *, source: str, log
) -> bool:
    """Append the next-bar reading to the predictions log (first one wins)."""
    from ..predictions import PredictionLog, prediction_log_path, reading_from_report

    reading = reading_from_report(
        payload, symbol=risk_cfg.data_symbol, interval=risk_cfg.interval, source=source
    )
    new = PredictionLog(prediction_log_path(config.data_root)).record(reading)
    stamp = datetime.fromtimestamp(reading["for_bar_ms"] / 1000, tz=timezone.utc)
    log(
        f"predictions log: {'recorded' if new else 'already had'} the reading for "
        f"{stamp:%Y-%m-%d %H:%M} UTC"
    )
    return new


def paper_payload(config: AppConfig) -> dict[str, Any]:
    """The paper record for the newest risk configuration in the log."""
    from ..predictions import PredictionLog, paper_record, prediction_log_path

    path = prediction_log_path(config.data_root)
    risk_cfg = config.risk
    if not path.exists():
        return {"readings": 0, "verdict": "No readings yet. Each risk run records one."}
    local = replace(
        config,
        exchange=replace(config.exchange, symbol=risk_cfg.data_symbol, interval=risk_cfg.interval),
    )
    bars = store_from_config(local).load_bars()
    bars_per_day = 86_400_000 // interval_to_ms(risk_cfg.interval)
    report = paper_record(
        PredictionLog(path), bars, fee_bps=risk_cfg.fee_bps,
        slippage_bps=risk_cfg.slippage_bps, bars_per_year=365.0 * bars_per_day,
    )
    return report.to_dict()
