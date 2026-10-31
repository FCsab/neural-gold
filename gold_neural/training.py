"""Training loop: walk-forward windows, a time budget, and a cooperative stop.

Two things make this usable from a browser:

* ``max_minutes`` caps wall-clock time. When it expires the loop stops at the
  next batch boundary, evaluates the test slice, and writes a checkpoint. A
  time-boxed run is useful, not a crash.
* ``should_stop`` is polled between batches. Pressing Stop in the UI sets the
  event, the loop exits cleanly, and whatever was learned so far is saved.

Training runs in its own process (see :func:`train_process_entry`). PyTorch
holds the GIL for long stretches in Python-level ops, and a process also gives
a hard boundary if the GPU driver misbehaves.
"""

from __future__ import annotations

import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model import WindowTransformer, probe_device

CLASSES = (-1, 0, 1)


@dataclass(frozen=True)
class TrainConfig:
    window: int = 64
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 64
    nhead: int = 4
    layers: int = 2
    dim_feedforward: int = 256
    patience: int = 5
    # None or 0 = no wall-clock limit (run until Stop or max epochs).
    max_minutes: float | None = 10.0
    seed: int = 7
    num_threads: int = 0  # 0 = let torch decide
    device: str = "auto"
    train_frac: float = 0.70
    val_frac: float = 0.15

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if not 0.3 <= self.train_frac <= 0.9:
            raise ValueError("train_frac must be within 0.3..0.9")
        if not 0.0 <= self.val_frac <= 0.5:
            raise ValueError("val_frac must be within 0..0.5")
        if self.train_frac + self.val_frac >= 0.99:
            raise ValueError("train_frac + val_frac must leave room for the test slice")


@dataclass
class TrainResult:
    status: str
    stop_reason: str
    epochs_done: int
    duration_s: float
    history: list[dict[str, float]]
    metrics: dict[str, Any]
    checkpoint_path: str | None
    checkpoint_version: int
    device: str
    device_note: str
    parameters: int
    config: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------
def _scaled(values: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise with statistics from the training slice only."""
    train = values[:train_end]
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return (values - mean) / std, mean, std


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray,
) -> dict[str, Any]:
    confusion = [
        [int(((y_true == actual) & (y_pred == predicted)).sum()) for predicted in CLASSES]
        for actual in CLASSES
    ]

    per_class: dict[str, dict[str, float]] = {}
    recalls: list[float] = []
    for i, label in enumerate(CLASSES):
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted = sum(confusion[j][i] for j in range(len(CLASSES)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
            "predicted": int(predicted),
        }
        if support:
            recalls.append(recall)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    balanced = float(np.mean(recalls)) if recalls else 0.0
    support_counts = np.bincount((y_true + 1).astype(int), minlength=3)
    baseline = float(support_counts.max() / len(y_true)) if len(y_true) else 0.0

    # A constant predictor that always says the majority class gets that class
    # right every time and the others never, so its *balanced* accuracy is
    # 1 / (number of classes present). Comparing balanced accuracy against the
    # plain majority share would be comparing two different scales.
    present = int((support_counts > 0).sum())
    baseline_balanced = 1.0 / present if present else 0.0

    long_mask = y_pred == 1
    stop_mask = y_pred == -1
    long_return = float(returns[long_mask].mean()) if long_mask.any() else None
    stop_return = float(returns[stop_mask].mean()) if stop_mask.any() else None
    edge = (
        long_return - stop_return
        if long_return is not None and stop_return is not None
        else None
    )

    return {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced, 4),
        "baseline_accuracy": round(baseline, 4),
        "baseline_balanced_accuracy": round(baseline_balanced, 4),
        "beats_baseline": bool(balanced > baseline_balanced + 0.05),
        "expected_return_long": None if long_return is None else round(long_return, 6),
        "expected_return_stop": None if stop_return is None else round(stop_return, 6),
        "signal_edge": None if edge is None else round(edge, 6),
        "long_coverage": round(float(long_mask.mean()), 4) if len(y_pred) else 0.0,
        "test_rows": int(len(y_true)),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
def train_model(
    frame: pd.DataFrame,
    feature_names: list[str],
    config: TrainConfig,
    checkpoint_dir: Path,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainResult:
    started = time.monotonic()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.num_threads:
        torch.set_num_threads(int(config.num_threads))

    probe = probe_device(config.device)
    device = torch.device(probe.device)

    values = frame[feature_names].to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy(dtype=np.int64)
    returns = (
        frame["realized_return"].to_numpy(dtype=np.float64)
        if "realized_return" in frame
        else np.zeros(len(frame))
    )

    n_rows = len(frame)
    window = int(config.window)
    train_end = int(n_rows * config.train_frac)
    val_end = int(n_rows * (config.train_frac + config.val_frac))
    first = window - 1

    train_rows = np.arange(first, train_end)
    val_rows = np.arange(max(train_end, first), val_end)
    test_rows = np.arange(max(val_end, first), n_rows)

    if len(train_rows) < 200 or len(val_rows) < 50 or len(test_rows) < 50:
        raise ValueError(
            f"dataset too small for window={window}: "
            f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )

    scaled, mean, std = _scaled(values, train_end)
    windows = np.lib.stride_tricks.sliding_window_view(scaled, window, axis=0)

    def batch_tensor(row_indices: np.ndarray) -> torch.Tensor:
        # windows[i - (window - 1)] is the window ending at row i
        chunk = windows[row_indices - first].transpose(0, 2, 1)
        return torch.from_numpy(np.ascontiguousarray(chunk)).to(device)

    label_index = (labels + 1).astype(np.int64)  # -1,0,1 -> 0,1,2

    counts = np.bincount(label_index[train_rows], minlength=3).astype(np.float32)
    weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = WindowTransformer(
        n_features=len(feature_names),
        window=window,
        d_model=config.d_model,
        nhead=config.nhead,
        layers=config.layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = F.cross_entropy
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    deadline = (
        started + float(config.max_minutes) * 60.0
        if config.max_minutes
        else None
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    stop_reason = "completed"
    epochs_done = 0

    def evaluate(rows: np.ndarray) -> tuple[np.ndarray, float]:
        model.eval()
        predictions: list[np.ndarray] = []
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(rows), config.batch_size):
                chunk = rows[start : start + config.batch_size]
                inputs = batch_tensor(chunk)
                targets = torch.from_numpy(label_index[chunk]).to(device)
                logits = model(inputs)
                total += float(criterion(logits, targets, weight=class_weights)) * len(chunk)
                predictions.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions), total / len(rows)

    for epoch in range(1, config.epochs + 1):
        if should_stop and should_stop():
            stop_reason = "stopped"
            break
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "time_budget"
            break

        model.train()
        order = np.random.permutation(train_rows)
        epoch_loss = 0.0
        seen = 0
        interrupted = False

        for start in range(0, len(order), config.batch_size):
            if should_stop and should_stop():
                stop_reason = "stopped"
                interrupted = True
                break
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = "time_budget"
                interrupted = True
                break

            chunk = order[start : start + config.batch_size]
            inputs = batch_tensor(chunk)
            targets = torch.from_numpy(label_index[chunk]).to(device)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(inputs)
                    loss = criterion(logits, targets, weight=class_weights)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = criterion(logits, targets, weight=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += float(loss.detach()) * len(chunk)
            seen += len(chunk)

        if interrupted or seen == 0:
            break

        train_loss = epoch_loss / seen
        val_predictions, val_loss = evaluate(val_rows)
        val_accuracy = float(
            (val_predictions - 1 == labels[val_rows]).mean()
        )
        epochs_done = epoch

        record = {
            "epoch": float(epoch),
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 4),
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        history.append(record)

        if progress is not None:
            progress(
                {
                    **record,
                    "epochs_total": config.epochs,
                    "message": (
                        f"epoch {epoch}/{config.epochs} · "
                        f"val_loss {val_loss:.4f} · val_acc {val_accuracy:.3f}"
                    ),
                }
            )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                stop_reason = "early_stopping"
                break

    if epochs_done == 0:
        return TrainResult(
            status="stopped",
            stop_reason=stop_reason,
            epochs_done=0,
            duration_s=round(time.monotonic() - started, 2),
            history=history,
            metrics={
                "note": "stopped before the first epoch completed; no checkpoint written",
                "test_rows": int(len(test_rows)),
            },
            checkpoint_path=None,
            checkpoint_version=0,
            device=probe.device,
            device_note=probe.note,
            parameters=model.parameter_count(),
            config=asdict(config),
        )

    # Evaluate the best weights (or the current ones if nothing improved yet).
    if best_state is not None:
        model.load_state_dict(best_state)
    test_predictions, test_loss = evaluate(test_rows)
    predicted_labels = test_predictions - 1
    metrics = _metrics(
        labels[test_rows], predicted_labels, returns[test_rows]
    )
    metrics["test_loss"] = round(test_loss, 6)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_loss"] = None if math.isinf(best_val) else round(best_val, 6)
    metrics["val_accuracy"] = round(
        float(
            (evaluate(val_rows)[0] - 1 == labels[val_rows]).mean()
        ),
        4,
    )

    version = max(epochs_done, 1)
    checkpoint_path = checkpoint_dir / f"v{version}.pt"
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "feature_names": list(feature_names),
        "window": window,
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "label_order": list(CLASSES),
        "metrics": metrics,
        "epoch": int(best_epoch or epochs_done),
        "torch_version": torch.__version__,
        "device": probe.device,
    }
    _save_checkpoint(checkpoint_path, payload)

    duration = time.monotonic() - started
    status = "ok" if stop_reason in {"completed", "early_stopping"} else "stopped"

    return TrainResult(
        status=status,
        stop_reason=stop_reason,
        epochs_done=epochs_done,
        duration_s=round(duration, 2),
        history=history,
        metrics=metrics,
        checkpoint_path=str(checkpoint_path),
        checkpoint_version=version,
        device=probe.device,
        device_note=probe.note,
        parameters=model.parameter_count(),
        config=asdict(config),
    )


# --------------------------------------------------------------------------
# process entry point
# --------------------------------------------------------------------------
def train_process_entry(
    config_dict: dict[str, Any],
    dataset_path: str,
    feature_names: list[str],
    checkpoint_dir: str,
    output_queue,
    stop_event,
) -> None:
    """Run inside a spawned process; speak only through the queue."""
    try:
        config = TrainConfig(**config_dict)
        frame = pd.read_parquet(dataset_path)

        def progress(payload: dict[str, Any]) -> None:
            output_queue.put({"type": "progress", **payload})

        result = train_model(
            frame,
            feature_names,
            config,
            Path(checkpoint_dir),
            progress=progress,
            should_stop=stop_event.is_set,
        )
        output_queue.put({"type": "result", "result": asdict(result)})
    except Exception as exc:  # noqa: BLE001 - the parent turns this into a job error
        output_queue.put(
            {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            }
        )
    finally:
        try:
            output_queue.close()
            output_queue.join_thread()
        except Exception:  # pragma: no cover - best effort
            pass


# --------------------------------------------------------------------------
# inference (used by the backtest)
# --------------------------------------------------------------------------
def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def predict_proba(
    checkpoint: dict[str, Any],
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Probabilities for ``row_indices`` of ``frame``, in the given order.

    Applies the scaler stored in the checkpoint, so the numbers match what the
    model saw during training. Rows must have a full window behind them.
    """
    feature_names = list(checkpoint["feature_names"])
    window = int(checkpoint["window"])
    cfg = checkpoint["config"]
    mean = np.asarray(checkpoint["scaler"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["scaler"]["std"], dtype=np.float32)

    if len(row_indices) and int(row_indices.min()) < window - 1:
        raise ValueError("row_indices must all have a full window behind them")

    values = frame[feature_names].to_numpy(dtype=np.float32)
    scaled = (values - mean) / std

    model = WindowTransformer(
        n_features=len(feature_names),
        window=window,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        layers=int(cfg["layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    view = np.lib.stride_tricks.sliding_window_view(scaled, window, axis=0)
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(row_indices), batch_size):
            chunk = row_indices[start : start + batch_size]
            batch = np.ascontiguousarray(
                view[chunk - (window - 1)].transpose(0, 2, 1)
            )
            logits = model(torch.from_numpy(batch))
            batches.append(torch.softmax(logits, dim=1).numpy())

    if not batches:
        return np.zeros((0, len(CLASSES)), dtype=np.float32)
    return np.concatenate(batches)
