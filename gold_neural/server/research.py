"""Detached overnight research campaign: start, stop, and status.

The campaign is deliberately *not* a server job. It is a separate process tree
(``scripts/overnight_ctl.sh``) so it survives the browser tab closing and the
server restarting. The server only writes a pidfile and reads the campaign's
own files; it never holds the training loop in its own process.

Everything here is behind the bearer token at the route layer. Commands are
built as an argument list, never through a shell.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "overnight_ctl.sh"
MAX_TAIL_BYTES = 4_000_000


def campaign_dir(config: AppConfig) -> Path:
    return config.data_root / "overnight"


def _pid_path(config: AppConfig) -> Path:
    return campaign_dir(config) / "campaign.pid"


def _pid_alive(pid: int) -> bool:
    # A campaign launched from this server is our child. Once it exits it stays
    # a zombie until reaped, and kill(pid, 0) succeeds on a zombie, so the UI
    # would report "running" and refuse a new start until the server restarts.
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # not our child (started from a shell): fall through to kill
    else:
        if reaped == pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_pid(config: AppConfig) -> int | None:
    """PID of the campaign supervisor, or None when it is not running."""
    path = _pid_path(config)
    if not path.is_file():
        return None
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _csv(values: Any) -> str:
    return ",".join(str(value) for value in values)


def build_command(config: AppConfig, params: dict[str, Any]) -> list[str]:
    """Argument list for the detached launcher. No shell, no user path."""
    out = campaign_dir(config)
    return [
        "bash",
        str(LAUNCHER),
        "--interval",
        str(params.get("interval", "4h")),
        "--data-root",
        str(config.data_root),
        "--out",
        str(out),
        "--policy",
        str(params.get("policy", "hold")),
        "--k-up",
        _csv(params.get("k_up", [2.5, 3.0, 3.5, 4.0])),
        "--k-down",
        _csv(params.get("k_down", [2.0, 2.5, 3.0, 4.0])),
        "--holding",
        _csv(params.get("holding", [192, 288])),
        "--window",
        _csv(params.get("window", [32, 64, 128])),
        "--seeds",
        _csv(params.get("seeds", [7])),
        "--epochs",
        str(params.get("epochs", 15)),
        "--minutes-per-fold",
        str(params.get("minutes_per_fold", 3.0)),
        "--test-days",
        str(params.get("test_days", 90)),
        "--val-days",
        str(params.get("val_days", 90)),
        "--min-train-days",
        str(params.get("min_train_days", 365)),
        "--fee-bps",
        str(params.get("fee_bps", 10.0)),
        "--slippage-bps",
        str(params.get("slippage_bps", 2.0)),
        "--device",
        str(params.get("device", "auto")),
        "--stop-thresholds",
        _csv(params.get("stop_thresholds", [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0])),
        "--reentry-margin",
        str(params.get("reentry_margin", 0.05)),
        "--min-flat-bars",
        str(params.get("min_flat_bars", 1)),
    ]


def start_campaign(
    config: AppConfig,
    params: dict[str, Any],
    *,
    popen: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    """Launch the detached campaign and record the supervisor PID."""
    out = campaign_dir(config)
    out.mkdir(parents=True, exist_ok=True)

    existing = running_pid(config)
    if existing is not None:
        raise RuntimeError(f"a campaign is already running (pid {existing})")
    if not LAUNCHER.is_file():
        raise RuntimeError(f"launcher not found: {LAUNCHER}")

    stop = out / "STOP"
    if stop.exists():
        stop.unlink()

    command = build_command(config, params)
    log_path = out / "web-launch.log"
    with log_path.open("a") as log:
        log.write(f"\n--- start {datetime.now(timezone.utc).isoformat()} ---\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        process = popen(
            command,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    _pid_path(config).write_text(str(process.pid))
    return {"started": True, "pid": int(process.pid), "out": str(out)}


def stop_campaign(config: AppConfig) -> dict[str, Any]:
    """Ask the supervisor to stop after the fold in flight."""
    out = campaign_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "STOP").touch()
    return {"stopped": True, "running": running_pid(config) is not None}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_text(path: Path, max_bytes: int = MAX_TAIL_BYTES) -> str:
    """Last ``max_bytes`` of a text file, without reading the whole thing."""
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            data = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    if size > max_bytes and "\n" in data:
        data = data.split("\n", 1)[1]
    return data


def _tail_lines(path: Path, count: int) -> list[str]:
    return _tail_text(path).splitlines()[-count:]


def _clean_spec(spec: Any) -> str | None:
    """Strip a legacy policy suffix from a leaderboard spec key.

    Status files written before the leaderboard grouping fix carry
    ``u2.5d3.0h288w128s7|hold``; fold keys are built from the clean spec.
    """
    text = str(spec or "").split("|", 1)[0].strip()
    return text or None


def _fold_prefix(spec: str, policy: str) -> str:
    # New hold campaigns key folds as "<spec>|hold|fN"; signal runs keep the
    # legacy "<spec>|fN" so old results still resume.
    return spec if policy == "signal" else f"{spec}|{policy}"


FOLD_FIELDS = (
    "fold",
    "test_start_ms",
    "test_end_ms",
    "return",
    "buy_hold",
    "excess",
    "trades",
    "stop_threshold",
    "validation_excess",
    "balanced_accuracy",
)


def _fold_view(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in FOLD_FIELDS}


def _read_folds(
    path: Path, spec: str | None, policy: str, limit: int
) -> list[dict[str, Any]]:
    if not path.is_file() or not spec:
        return []

    prefix = _fold_prefix(spec, policy)
    by_fold: dict[int, dict[str, Any]] = {}
    for line in _tail_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("status") != "ok":
            continue
        if row.get("policy", "signal") != policy:
            continue
        key = str(row.get("key", ""))
        if not key.startswith(prefix + "|f"):
            continue
        fold = int(row.get("fold", -1))
        by_fold[fold] = _fold_view(row)

    return [by_fold[fold] for fold in sorted(by_fold)][-limit:]


def campaign_status(
    config: AppConfig,
    *,
    spec: str | None = None,
    include_folds: bool = True,
) -> dict[str, Any]:
    """Everything the Research tab needs, from files the campaign owns."""
    out = campaign_dir(config)
    status = _read_json(out / "status.json")
    board = [
        dict(item)
        for item in (status.get("leaderboard") or [])
        if isinstance(item, dict)
    ]
    max_folds = max((int(item.get("folds") or 0) for item in board), default=0)
    for item in board:
        item["spec"] = _clean_spec(item.get("spec"))
        wanted = int(item.get("expected_folds") or max_folds or 0)
        item["expected_folds"] = wanted
        item["complete"] = bool(wanted and int(item.get("folds") or 0) >= wanted)
    if board:
        status["leaderboard"] = board

    requested = _clean_spec(spec)
    best = requested or (_clean_spec(board[0].get("spec")) if board else None)
    # Status files written before the hold campaign have no policy field; they
    # are the signal policy. New files always write it.
    policy = str(status.get("policy") or "signal")
    if status:
        status["policy"] = policy
    pid = running_pid(config)

    folds: list[dict[str, Any]] = []
    if include_folds and best:
        top_spec = _clean_spec(board[0].get("spec")) if board else None
        cached = status.get("best_folds") if (not requested or requested == top_spec) else None
        if isinstance(cached, list) and cached:
            folds = [
                _fold_view(row) for row in cached if isinstance(row, dict)
            ][-240:]
        else:
            folds = _read_folds(out / "results.jsonl", best, policy, limit=240)

    log_tail = _tail_lines(out / "log.txt", 8)
    if not log_tail:
        # A launcher that failed before the runner started leaves its reason
        # in web-launch.log; show that instead of an empty pane.
        log_tail = _tail_lines(out / "web-launch.log", 8)

    return {
        "running": pid is not None,
        "pid": pid,
        "stop_requested": (out / "STOP").exists(),
        "dir": str(out),
        "status": status,
        "best_spec": best,
        "folds": folds,
        "log_tail": log_tail,
    }
