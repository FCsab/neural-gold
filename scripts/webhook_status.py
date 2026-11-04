#!/usr/bin/env python3
"""Post a compact gold-neural campaign status to a Discord webhook.

Reads the campaign's own files under ``data/overnight`` and the webhook URL
from ``GOLDNEURAL_DISCORD_WEBHOOK`` or ``~/.config/gold-neural/discord-webhook``.
The URL is never printed or logged.

Intended to run from cron every 10 minutes:

    */10 * * * * cd ... && .venv/bin/python scripts/webhook_status.py \\
        >> data/overnight/webhook.log 2>&1

``--dry-run`` prints the message without sending or updating the delta state.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "overnight"
STATUS = OUT / "status.json"
LOG = OUT / "log.txt"
RESULTS = OUT / "results.jsonl"
PIDFILE = OUT / "campaign.pid"
STOP = OUT / "STOP"
STATE = OUT / "webhook-state.json"
WEBHOOK_FILE = Path.home() / ".config" / "gold-neural" / "discord-webhook"
MAX_CHARS = 1900


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def age_seconds(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def campaign_running() -> bool:
    if PIDFILE.is_file():
        try:
            if _pid_alive(int(PIDFILE.read_text().strip())):
                return True
        except ValueError:
            pass
    # Fallback for a manually started runner without the supervisor pidfile.
    try:
        result = subprocess.run(
            ["pgrep", "-af", "scripts/overnight.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    marker = str(OUT)
    relative = "data/overnight"
    return any(
        "overnight.py" in line and (marker in line or relative in line)
        for line in result.stdout.splitlines()
    )


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value) * 100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "–"


def points(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value) * 100:+.{digits}f} pts"
    except (TypeError, ValueError):
        return "–"


def count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "–"


def best_rows(board: list[Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    rows = [row for row in board if isinstance(row, dict)]
    complete = [row for row in rows if row.get("complete") is not False]
    pool = complete or rows
    if not pool:
        return None, None, 0

    by_wins = sorted(
        pool,
        key=lambda row: (
            float(row.get("share_beating_hold") or 0),
            float(row.get("compounded") or 0),
            int(row.get("folds") or 0),
        ),
        reverse=True,
    )
    by_compounded = sorted(
        pool, key=lambda row: float(row.get("compounded") or -9), reverse=True
    )
    ahead = [
        row
        for row in pool
        if row.get("hold_compounded") is not None
        and float(row.get("compounded") or 0)
        > float(row.get("hold_compounded") or 0) + 1e-9
    ]
    return by_wins[0], by_compounded[0], len(ahead)


def build_message(
    status: dict[str, Any], state: dict[str, Any], running: bool, stop: bool
) -> str:
    board = status.get("leaderboard") or []
    best, best_compounded, ahead = best_rows(board)
    folds = int(status.get("folds_total") or 0)
    specs = int(status.get("specs_done") or status.get("specs_tracked") or 0)

    previous_folds = int(state.get("folds_total") or 0) if state else None
    previous_specs = int(state.get("specs_done") or 0) if state else None
    fold_delta = None if previous_folds is None else folds - previous_folds
    spec_delta = None if previous_specs is None else specs - previous_specs

    lines: list[str] = [f"**gold-neural status** · {time.strftime('%H:%M')}"]
    state_word = "Running" if running else ("Stopping" if stop else "Not running")
    bits = [state_word]
    for key in ("interval", "policy"):
        if status.get(key):
            bits.append(str(status[key]))
    if status.get("folds_per_minute") is not None:
        try:
            bits.append(f"{float(status['folds_per_minute']):.1f} folds/min")
        except (TypeError, ValueError):
            pass
    bits.append(f"{count(folds)} folds")
    if fold_delta is not None:
        bits.append(f"{fold_delta:+d} this interval")
    lines.append(" · ".join(bits))

    lines.append(f"Current: `{status.get('current') or '—'}`")
    if spec_delta is not None:
        lines.append(f"Settings finished this interval: {spec_delta}")

    if best:
        wins = round(
            float(best.get("share_beating_hold") or 0) * int(best.get("folds") or 0)
        )
        compounded = best.get("compounded")
        hold = best.get("hold_compounded")
        lines.append(
            f"Best by wins: `{best.get('spec')}` — {wins}/{best.get('folds')} wins · "
            f"median {pct(best.get('median_return'))} · "
            f"{pct(compounded)} vs hold {pct(hold)} "
            f"({points((compounded or 0) - (hold or 0))})"
        )
    if best_compounded and (
        not best or best_compounded.get("spec") != best.get("spec")
    ):
        lines.append(
            f"Best compounded: `{best_compounded.get('spec')}` — "
            f"{pct(best_compounded.get('compounded'))} vs hold "
            f"{pct(best_compounded.get('hold_compounded'))}"
        )
    lines.append(
        f"Ahead of holding: {ahead} setting{'s' if ahead != 1 else ''}"
        if ahead
        else "Ahead of holding: none yet"
    )

    warnings: list[str] = []
    if stop:
        warnings.append("STOP file present — finishing the current fold")
    if not running and not stop:
        warnings.append("campaign process is not running")

    results_age = age_seconds(RESULTS)
    if running and results_age is not None and results_age > 600:
        warnings.append(f"no fold written in {results_age / 60:.0f} min")

    status_age = age_seconds(STATUS)
    if running and status_age is not None and status_age > 1800:
        warnings.append(f"status.json is {status_age / 60:.0f} min old")

    try:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        if free_gb < 1.0:
            warnings.append(f"disk free {free_gb:.1f} GB")
    except OSError:
        pass

    try:
        tail = LOG.read_text().splitlines()[-5:]
        bad = [
            line
            for line in tail
            if "Traceback" in line or "failed" in line.lower()
        ]
        if bad:
            warnings.append("recent log: " + bad[-1][:140])
    except OSError:
        pass

    if warnings:
        lines.append("")
        lines.extend("⚠ " + warning for warning in warnings)

    return "\n".join(lines)[:MAX_CHARS]


def webhook_url() -> str:
    from_env = os.environ.get("GOLDNEURAL_DISCORD_WEBHOOK", "").strip()
    if from_env:
        return from_env
    try:
        return WEBHOOK_FILE.read_text().strip()
    except OSError:
        return ""


def post(url: str, text: str) -> int:
    payload = json.dumps({"content": text}).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "gold-neural-status/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    status = read_json(STATUS)
    state = read_json(STATE)
    running = campaign_running()
    stop = STOP.exists()
    text = build_message(status, state, running, stop)

    if args.dry_run:
        print(text)
        return 0

    folds = int(status.get("folds_total") or 0)
    specs = int(status.get("specs_done") or status.get("specs_tracked") or 0)

    url = webhook_url()
    if not url:
        print(f"{time.strftime('%FT%T')} no webhook configured")
        return 1

    try:
        code = post(url, text)
    except urllib.error.HTTPError as exc:
        print(f"{time.strftime('%FT%T')} webhook HTTP {exc.code}")
        return 1
    except Exception as exc:  # noqa: BLE001 - cron boundary, log the type only
        print(f"{time.strftime('%FT%T')} webhook error {type(exc).__name__}")
        return 1

    print(f"{time.strftime('%FT%T')} sent {code}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        json.dumps(
            {
                "folds_total": folds,
                "specs_done": specs,
                "checked_at": time.time(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
