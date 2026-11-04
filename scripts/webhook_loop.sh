#!/usr/bin/env bash
# Post a campaign status to the Discord webhook every 10 minutes.
#
#   nohup setsid bash scripts/webhook_loop.sh >/dev/null 2>&1 &
#
# No cron. The loop runs as a detached process and survives closing this TUI.
# Stop it with:
#   touch data/overnight/webhook.stop
# or:
#   kill "$(cat data/overnight/webhook.pid)"

set -u
cd "$(dirname "$0")/.." || exit 1

OUT="data/overnight"
mkdir -p "$OUT"

PIDFILE="$OUT/webhook.pid"
STOPFILE="$OUT/webhook.stop"

printf '%s\n' "$$" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
trap 'exit 0' INT TERM

# The first update is immediate; then every ten minutes.
while [ ! -f "$STOPFILE" ]; do
  .venv/bin/python scripts/webhook_status.py >> "$OUT/webhook.log" 2>&1
  sleep 600
done
