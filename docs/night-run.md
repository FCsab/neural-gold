# Night run: handoff

Everything below is ready to start and leave. The runner needs no AI process, no
browser, and no network after the bars are on disk.

## Launch

```bash
./scripts/start_night.sh
```

Or open the **Research** tab in the web UI and press **Start campaign**; it
launches the same detached supervisor, with live progress and charts. Both
paths write to `data/overnight` and are resumable.

The shell command starts the default hold-policy campaign in the background and
prints the status and stop commands. To give it several seeds:

```bash
./scripts/start_night.sh data/overnight --seeds 7,8,9
```

To start from a clean slate:

```bash
rm -rf data/overnight && ./scripts/start_night.sh
```

## While it runs

```bash
./scripts/status.sh                 # one screen, read-only
tail -f data/overnight/log.txt      # one line per finished setting
```

## Stop

```bash
touch data/overnight/STOP
```

The runner finishes the fold in flight, writes it, and exits. Re-running the
same command resumes from disk.

## What the night actually tests

Each fold trains on everything before a 90-day block, validates on the 90 days
before it, and trades the test block. The policy stays long by default, steps
aside when the model's stop probability crosses a threshold, and re-enters below
that threshold minus a margin.

The threshold is chosen **on validation** from `0.30 … 0.60` and `1.0`. If no
threshold beats buy-and-hold on validation, it picks `1.0` and simply holds for
the test block. The test block never chooses the rule; it only reports what the
rule did.

The grid is 96 label settings × 19 folds ≈ 2.2 hours per seed at the measured
rate. When a pass ends the seed advances (7 → 8 → 9, or past the supplied list),
so the night keeps producing new work instead of re-walking finished keys.

## What to look for in the morning

Read `data/overnight/status.json` first. The leaderboard is ranked by
`median_excess` — the typical quarter's lead over buy-and-hold — because a
strategy that makes 5% while the asset makes 8% has not achieved the goal.

The four numbers that decide whether a setting is real:

- `median_excess` > 0 and `share_beating_hold` comfortably above 0.5.
- `worst_return` not much worse than holding's worst quarter.
- `median_trades` low: every round trip costs 24 bps. A high trade count with a
  negative median excess is the signal paying fees for noise.
- `stop_threshold` not clustered at `1.0`. A cluster there means the model
  rarely earned the right to step aside on validation; the policy correctly did
  nothing.

One pass is a screen, not a verdict. A cell with `share_beating_hold` near 0.6
on one seed is a candidate for a three-seed re-test, not a promotion.

## Evidence before the night run

The primary configuration was walked forward end to end on 4h bars, 19 folds,
with the final code and the cost-matched baseline:

| setting | median excess | share beating hold | worst quarter | compounded vs hold |
|---|---|---|---|---|
| `u2.5d3.0h288w128` | 0.00% | 26% | -5.21% | +121.9% vs +132.2% |

That is the honest current state: the policy matches holding in the median
quarter, cuts the worst quarter to -5.2%, and trails terminal wealth by about
10 points over 4.75 years (roughly 2 points a year). The policy stays long in
8 of 19 quarters because validation offered no edge; those folds match holding
exactly and do not count as wins. Of the 11 quarters where it did step aside,
it beat holding in 5. The losses from premature exits were larger than the
gains from avoided drawdowns.

Two exploratory runs are not in the table because their share numbers were
computed before the baseline was cost-matched:

- closer stop (`u2.5d2.0h288w64`): lower terminal wealth, worse tail.
- 200-bar EMA feature added: higher median quarter, but a -16.0% worst quarter
  and lower terminal wealth. Reverted.

The negative feature result is recorded in the section below so the next attempt
does not repeat it without new evidence.

The model's median balanced accuracy on the first configuration was 0.52
against a 0.33 baseline, so the classifier is learning something. The gap is in
turning that into timing value, not in fitting the training data.

## Feature experiment: reverted

`ema_dist_200` (distance from a 200-bar EMA, ≈33 days at 4h) was added, tested
on one setting, and reverted. It did what it was supposed to do — more active
rules, a higher median quarter — but the tail got worse: a -16.0% worst quarter
against -5.3% without it, and lower terminal wealth. The leakage suite passed
with it, so this was a signal-quality result, not a correctness bug.

The feature set is back to `(20, 50)` in `gold_neural/config.py` and
`config.example.yaml`. Do not re-add it without a paired walk-forward that shows
a better worst quarter, not just a better median.

## Definition of done for this handoff

- `pytest`: 137 collected, green.
- `cli leakage`: PASS, exit 0 (21 features; the 22-feature variant also passed
  before it was reverted).
- `docs/overnight.md`: updated for the hold campaign, threshold rule, and
  ranking.
- `./scripts/status.sh`: renders policy, interval and setting count, and ranks
  by median excess.
- `./scripts/start_night.sh`: preflight, launch, status and stop instructions.
- `scripts/smoke.py` against a live server: 12/12 (including the leakage job).
