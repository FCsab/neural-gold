#!/usr/bin/env bash
# One-screen status for the overnight walk-forward sweep.
#
#   ./scripts/status.sh            # reads data/overnight
#   ./scripts/status.sh path/to/out
#
# Safe to run at any time: it only reads files and asks pgrep for pids.

set -u
cd "$(dirname "$0")/.." || exit 1

OUT="${1:-data/overnight}"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ ! -d "$OUT" ]; then
  echo "no sweep directory at $OUT"
  echo "start one with: nohup bash scripts/overnight_ctl.sh --interval 4h >/dev/null 2>&1 &"
  exit 1
fi

"$PY" - "$OUT" <<'PY'
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
status_path = out / "status.json"
results_path = out / "results.jsonl"
stop_path = out / "STOP"
log_path = out / "log.txt"

BOLD = "\033[1m" if sys.stdout.isatty() else ""
DIM = "\033[2m" if sys.stdout.isatty() else ""
RESET = "\033[0m" if sys.stdout.isatty() else ""


def pids(pattern: str) -> list[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [line for line in result.stdout.split() if line.isdigit()]


def pct(value):
    if value is None:
        return "     -"
    return f"{value * 100:+6.2f}%"


def share(value):
    if value is None:
        return "   -"
    return f"{value * 100:3.0f}%"


supervisors = pids("overnight_ctl")
runners = pids("overnight\\.py")

print(f"{BOLD}gold-neural · overnight walk-forward{RESET}")
print("─" * 74)

if supervisors or runners:
    parts = []
    if supervisors:
        parts.append(f"supervisor {supervisors[0]}")
    if runners:
        parts.append(f"runner {runners[0]}")
    print(f"state      RUNNING  ({', '.join(parts)})")
else:
    print("state      STOPPED")

if stop_path.exists():
    print(f"           {BOLD}STOP file present{RESET} — finishing the current fold, then exiting")

status = None
if status_path.exists():
    try:
        status = json.loads(status_path.read_text())
    except json.JSONDecodeError:
        status = None

if status is None:
    print("progress   no status.json yet — the first fold has not finished")
else:
    elapsed = status.get("elapsed_minutes") or 0.0
    hours = elapsed / 60
    print(
        f"elapsed    {hours:5.1f} h  ·  {status.get('folds_per_minute', 0)} folds/min"
        f"  ·  interval {status.get('interval', '?')}"
    )
    print(
        f"progress   {status.get('folds_total', 0):,} folds on disk"
        f"  ·  {status.get('specs_tracked', 0)} settings tracked"
    )
    print(f"current    {status.get('current', '-')}")

rows = 0
if results_path.exists():
    with results_path.open() as handle:
        rows = sum(1 for _ in handle)
if status is not None and rows and rows != status.get("folds_total"):
    print(f"{DIM}           ({rows:,} result lines on disk){RESET}")

board = (status or {}).get("leaderboard") or []
if board:
    print()
    print(f"{BOLD}best settings so far, by median 3-month return{RESET}")
    header = (
        f"  {'setting':<22}{'folds':>6}{'median':>9}{'worst':>9}"
        f"{'>=+5%':>7}{'beats hold':>12}{'compounded':>12}"
    )
    print(f"{DIM}{header}{RESET}")
    for entry in board[:8]:
        name = entry.get("spec", "?")
        print(
            f"  {name:<22}{entry.get('folds', 0):>6}"
            f"{pct(entry.get('median_return')):>9}"
            f"{pct(entry.get('worst_return')):>9}"
            f"{share(entry.get('share_over_5pct')):>7}"
            f"{share(entry.get('share_beating_hold')):>12}"
            f"{pct(entry.get('compounded')):>12}"
        )
    print()
    print(
        f"{DIM}  median = typical quarter · worst = the quarter that pays for the rest\n"
        f"  >=+5% = share of quarters that cleared the goal · beats hold = share the\n"
        f"  model beat buy-and-hold over the same block{RESET}"
    )
else:
    print()
    print("no setting has finished all of its folds yet — nothing to rank")

if log_path.exists():
    lines = [line.rstrip() for line in log_path.read_text().splitlines() if line.strip()]
    if lines:
        print()
        print(f"{BOLD}recent log{RESET}")
        for line in lines[-4:]:
            print(f"  {DIM}{line}{RESET}")

print()
print(f"{DIM}stop with:  touch {out}/STOP{RESET}")
PY
