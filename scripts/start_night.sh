#!/usr/bin/env bash
# Start the unattended hold-policy campaign and walk away.
#
#   ./scripts/start_night.sh                # data/overnight, default grid
#   ./scripts/start_night.sh /tmp/night --seeds 7,8,9
#
# The runner is resumable: re-running continues from the folds already on
# disk. Create <out>/STOP to end it cleanly after the current fold.

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

OUT="data/overnight"
# A leading flag means "use the default directory"; a path means "write here".
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  OUT="$1"
  shift
fi

if [ ! -x .venv/bin/python ]; then
  echo "error: .venv/bin/python is missing; see README.md setup" >&2
  exit 1
fi

if ! .venv/bin/python -c "import torch" >/dev/null 2>&1; then
  echo "error: torch is not installed in .venv (training needs it)" >&2
  echo "  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
  exit 1
fi

if pgrep -f "scripts/overnight.py" >/dev/null 2>&1; then
  echo "error: a sweep is already running; stop it with touch ${OUT}/STOP" >&2
  exit 1
fi

mkdir -p "$OUT"
rm -f "$OUT/STOP"

nohup bash scripts/overnight_ctl.sh --interval 4h --out "$OUT" "$@" \
  >/dev/null 2>&1 &

sleep 2
echo "started  $OUT"
echo "status   ./scripts/status.sh $OUT"
echo "stop     touch $OUT/STOP"
echo "log      $OUT/log.txt"
