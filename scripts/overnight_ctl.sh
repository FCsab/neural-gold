#!/usr/bin/env bash
# Keep the overnight sweep alive until data/overnight/STOP exists.
#
#   nohup bash scripts/overnight_ctl.sh --interval 4h >/dev/null 2>&1 &
#
# The runner is resumable, so a restart costs at most the fold that was in
# flight. Create data/overnight/STOP to end it cleanly; the supervisor stops
# after the current fold and does not relaunch.

set -u
cd "$(dirname "$0")/.." || exit 1

OUT="data/overnight"
mkdir -p "$OUT"

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
