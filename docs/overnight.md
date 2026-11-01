# Overnight walk-forward sweep

`scripts/overnight_ctl.sh` keeps `scripts/overnight.py` alive until a stop file
appears. The runner evaluates every label setting across rolling 3-month
blocks and never idles: when the grid is exhausted the seed advances.

## Start

```bash
nohup bash scripts/overnight_ctl.sh --interval 4h >/dev/null 2>&1 &
```

## Stop

```bash
touch data/overnight/STOP
```

The runner finishes the fold it is on, writes its result, and exits. The
supervisor sees the file and does not relaunch. Nothing is lost — every
finished fold is already on disk.

## What a fold is

```
train  [ first row .. T-90d )     expanding window, starts at the first row
val    [ T-90d .. T )             90 days, used only for checkpoint selection
test   [ T .. T+90d )             90 days, traded
```

Folds slide forward 90 days, so test blocks abut and the per-fold returns form
a continuous path. At 4h bars over 6.1 years that is 19 folds per setting.

Each fold trains a fresh model, picks its best epoch by validation loss, then
trades the test block through the normal backtest (10 bps fee, 2 bps slippage,
entry at the next bar's open).

## What is swept

| Setting | Values |
|---|---|
| `k_up` | 3.0, 4.0, 2.5, 3.5, 2.0, 5.0 |
| `k_down` | 2.0, 3.0, 1.5, 2.5, 4.0 |
| holding | 24, 48, 96, 12, 6 bars |
| window | 64, 32, 128 |

450 configurations per seed, then the seed advances so the night continues.
At roughly 11 folds per minute each setting takes under two minutes, so one
full pass is about 13 hours of work.

## Check on it

```bash
./scripts/status.sh
```

One screen: whether the supervisor and runner are alive, elapsed time, folds
per minute, the setting in progress, and the best settings so far ranked by
median 3-month return. It only reads files, so running it never disturbs the
sweep.

Sample:

```
state      RUNNING  (supervisor 1224520, runner 1224523)
elapsed      3.7 h  ·  9.4 folds/min  ·  interval 4h
progress   2,180 folds on disk  ·  31 settings tracked
current    u3.0d2.0h48w64s7 fold 7/19

best settings so far, by median 3-month return
  setting                folds   median    worst  >=+5%  beats hold  compounded
  u3.0d2.0h48w64s7          19   +2.10%   -6.40%    38%         68%     +12.40%
```

## Reading the output

- `data/overnight/status.json` — live heartbeat: folds done, rate, current
  setting, and a leaderboard of the ten best settings by median 3-month return.
- `data/overnight/log.txt` — one line per finished setting:
  `median`, `worst`, share of quarters `>= +5%`, share `beating hold`,
  and the compounded return.
- `data/overnight/results.jsonl` — one line per fold. The raw data; every
  conclusion should come from this file, not from the leaderboard.

The numbers that matter, given a goal of +5% per quarter:

- **median_return** — a typical 3-month block. Positive and above holding is
  the bar.
- **share_over_5pct** — how often a quarter actually clears the goal.
- **share_beating_hold** — whether the model adds timing value at all, in
  up quarters and down ones.
- **worst_return** — the quarter that pays for everything else.

## Re-testing one region

The full grid is the right first pass; a focused re-test is the right second
one. Narrow the search space and give it several seeds:

```bash
nohup bash scripts/overnight_ctl.sh --interval 4h \
  --k-up 3.5,4 --k-down 2.5,3,4 --holding 192,288 --window 32,64 \
  --seeds 7,8,9 >/dev/null 2>&1 &
```

Because results are keyed by setting *and* seed, reusing `data/overnight`
skips the seed-7 work already done and only runs the new seeds.

This is how a promising cell gets separated from a lucky initialisation: a
single seed can show a +6% median quarter, while the same cell across three
seeds settles to its real mean.

## Resume

The runner reads `results.jsonl` at startup and skips any fold already
recorded. Killing it, rebooting, or a crash costs at most the fold in flight.
Re-running the same command continues where it stopped.

Deleting `results.jsonl` starts from scratch; deleting `data/overnight`
removes everything.
