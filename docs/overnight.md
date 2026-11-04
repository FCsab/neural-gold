# Overnight walk-forward sweep

`scripts/overnight_ctl.sh` keeps `scripts/overnight.py` alive until a stop file
appears. The runner evaluates label settings across rolling 3-month blocks and
never idles: when the grid is exhausted the seed advances.

The default policy is **hold**: stay long, step aside only when the model is
confident about the stop outcome. That is the only formulation that can be
compared directly against buy-and-hold, which is the question this project is
trying to answer. The signal policy is still available with
`--policy signal`, but it has to time entries as well as exits and has not
beaten holding in the recorded runs.

## Start

```bash
nohup bash scripts/overnight_ctl.sh --interval 4h >/dev/null 2>&1 &
```

That starts the default hold campaign: `k_up` 2.5–4, `k_down` 2–4, holding
192/288 bars, window 32/64/128, and a stop-threshold grid selected per fold on
validation. It is 96 settings × 19 folds ≈ 2.2 hours per pass at the measured
rate; when the pass ends the seed advances and the night continues.

To start from a clean slate, remove the old output directory first:

```bash
rm -rf data/overnight && nohup bash scripts/overnight_ctl.sh --interval 4h \
  >/dev/null 2>&1 &
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
val    [ T-90d .. T )             90 days: checkpoint choice and threshold choice
test   [ T .. T+90d )             90 days, traded
```

Folds slide forward 90 days, so test blocks abut. At 4h bars over 6.1 years
that is 19 folds per setting.

Each fold trains a fresh model, picks its best epoch by validation loss, then
trades the test block through the normal backtest (10 bps fee, 2 bps slippage,
entry at the next bar's open). The hold policy force-closes at the last bar of
the block, so a fold cannot borrow returns from the next one.

## How the hold policy decides

The model outputs three probabilities: stop (`-1`), timeout (`0`), target
(`+1`). The policy ignores the target class:

1. Start long unless the stop probability is already above the entry line.
2. Step aside when `p_stop >= stop_threshold`.
3. Re-enter when `p_stop <= stop_threshold - reentry_margin`, and only after
   `min_flat_bars` bars flat.

The margin and the cooldown are there because every round trip costs 24 bps at
the default fees. A rule that flips on a probability hovering at the line pays
that cost repeatedly for no information.

`stop_threshold` is chosen **per fold on the validation block**, from
`0.30 … 0.60` and `1.0`. Candidates are scored by net excess return against
buy-and-hold; ties go to fewer trades. `1.0` means "never step aside". If no
candidate beats holding on validation, the policy stays long for the test
block. Doing nothing is a decision; shipping a rule that lost money on the only
data available to choose it is not.

The choice is recorded in every result line as `stop_threshold`, next to the
validation excess that justified it. The test block is never used to pick it.

## What is swept

| Setting | Values |
|---|---|
| `k_up` | 2.5, 3.0, 3.5, 4.0 |
| `k_down` | 2.0, 2.5, 3.0, 4.0 |
| holding | 192, 288 bars (32, 48 days at 4h) |
| window | 32, 64, 128 bars |
| stop threshold | 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0 |

Long holdings dominate the earlier signal-policy results: 192 and 288 bars were
the only regions that came close to holding. The grid is deliberately centred
there instead of spending the night on 6-bar settings that cannot clear costs.

## Check on it

```bash
./scripts/status.sh
```

One screen: whether the supervisor and runner are alive, elapsed time, folds
per minute, the setting in progress, and the best settings so far. For the hold
policy the leaderboard is ranked by **median excess return vs holding**, not by
raw return — a strategy that makes 5% while the asset makes 8% has not achieved
the goal.

Sample:

```
state      RUNNING  (supervisor 1224520, runner 1224523)
elapsed      3.7 h  ·  9.4 folds/min  ·  interval 4h
progress   2,180 folds on disk  ·  31 settings tracked
current    u3.0d3.0h288w64s7 fold 7/19

best settings so far, by median excess vs hold
  setting                folds   median    worst  >=+5%  beats hold  compounded
  u2.5d3.0h288w128s7        19   +3.20%   -6.40%    38%         68%     +12.40%
```

## Reading the output

- `data/overnight/status.json` — live heartbeat: folds done, rate, current
  setting, policy, and a leaderboard of the best settings.
- `data/overnight/log.txt` — one line per finished setting: `median`, `worst`,
  share of quarters `>= +5%`, share `beating hold`, and the compounded return.
- `data/overnight/results.jsonl` — one line per fold. The raw data; every
  conclusion should come from this file, not from the leaderboard.

The numbers that matter, given a goal of beating holding:

- **median_excess** — the typical quarter's lead over buy-and-hold. This is the
  selection criterion for the hold policy.
- **share_beating_hold** — how often the policy actually wins, in up quarters
  and down ones.
- **worst_return** — the quarter that pays for everything else.
- **median_trades** — churn. A high count with a negative median excess means
  the signal is paying fees for noise.
- **stop_threshold** — the value chosen on validation. A cluster near `1.0`
  means the model rarely earned the right to step aside.

## Re-testing one region

The first pass is exploration; a focused re-test is the second. Narrow the
search space and give it several seeds:

```bash
nohup bash scripts/overnight_ctl.sh --interval 4h \
  --k-up 3.0,3.5 --k-down 2.5,3.0 --holding 288 --window 64,128 \
  --seeds 7,8,9 >/dev/null 2>&1 &
```

Because results are keyed by setting *and* seed *and* policy, reusing
`data/overnight` skips the work already done and only runs the new
combinations. Results from the old signal-policy campaign stay in the file but
are ignored by a hold-policy leaderboard.

This is how a promising cell gets separated from a lucky initialisation: a
single seed can show a +6% median quarter, while the same cell across three
seeds settles to its real mean.

## Resume

The runner reads `results.jsonl` at startup and skips any fold already
recorded. Killing it, rebooting, or a crash costs at most the fold in flight.
Re-running the same command continues where it stopped.

Deleting `results.jsonl` starts from scratch; deleting `data/overnight`
removes everything.
