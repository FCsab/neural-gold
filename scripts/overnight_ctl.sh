#!/usr/bin/env bash
# Keep the overnight sweep alive until <out>/STOP exists.
#
#   nohup bash scripts/overnight_ctl.sh --interval 4h >/dev/null 2>&1 &
#
# The runner is resumable, so a restart costs at most the fold that was in
# flight. Create <out>/STOP to end it cleanly; the supervisor stops after the
# current fold and does not relaunch.
#
# The supervisor owns <out>/campaign.pid. The web UI reads it to tell whether a
# campaign is already running, whether it was started here or from the browser.

set -u
cd "$(dirname "$0")/.." || exit 1

OUT="data/overnight"
previous=""
for argument in "$@"; do
  if [ "$previous" = "--out" ]; then
    OUT="$argument"
  fi
  previous="$argument"
done

mkdir -p "$OUT"
PIDFILE="$OUT/campaign.pid"
printf '%s\n' "$$" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
trap 'exit 0' INT TERM

while [ ! -f "$OUT/STOP" ]; do
  .venv/bin/python scripts/overnight.py "$@" >> "$OUT/supervisor.log" 2>&1
  code=$?
  if [ -f "$OUT/STOP" ]; then
    break
  fi
  printf '%s runner exited with %s; restarting in 15s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code" >> "$OUT/supervisor.log"
  sleep 15
done

printf '%s supervisor stopped\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$OUT/supervisor.log"
