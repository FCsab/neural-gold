# Auto-search

The question this answers is not "what is the correct `k_up`". There is no
correct value. It is: **which label settings produce trades big enough to
survive costs, and do they still work on data the model never saw?**

## How a sweep works

For each candidate `(k_up, k_down, holding, window)`:

1. Build the dataset with those barriers (features are unchanged).
2. Train a model for `per_candidate_minutes`, in its own child process.
3. Backtest it on the **validation** slice and on the **test** slice.
4. Record it as a model in the registry and append it to the leaderboard.

Then it moves to the next candidate until the time budget or the candidate cap
runs out. Every result is written to `data/reports/search.json` the moment it
finishes, so a crash costs at most the candidate that was in flight.

## Two rules that keep it honest

**Selection happens on validation.** The winner is the candidate with the best
validation return, not the best test return. Test is then reported for the
winner and is the only number that has not influenced the choice. Reporting
the best *test* number across 40 attempts would be selection bias wearing a
lab coat.

**Hopeless candidates never get trained.** Two cheap filters run before any
time is spent:

- *Cost filter.* The profit target (`k_up × ATR`) must be at least 2.5× the
  round-trip cost. Below that, a perfect oracle loses money — this is the
  failure the first backtest on this repo hit.
- *Reachability filter.* A driftless walk covers roughly `ATR × √holding`, so a
  target above `1.5 × √holding` ATR is not going to be reached before the
  timeout. Training it would only produce timeouts.

Both filters read the ATR from the stored bars, so they adapt if you change
the pair or the timeframe.

## Budget knobs

| Setting | Meaning |
|---|---|
| total hours | wall clock for the whole sweep; it stops wherever it is |
| minutes per candidate | training time per setting; more time = better models, fewer settings |
| max candidates | hard cap on the sweep size |
| fees / slippage | used for the cost filter *and* the backtests |

Defaults: 3 hours, 4 minutes per candidate, 40 candidates. On 210k five-minute
bars a candidate takes roughly 4 minutes, so the default fills the budget.

## The second pass

After the label sweep, the top `refine_top` candidates are retrained with
different windows (32 and 128 by default) to check whether the window matters
at all. It only runs if time remains.

## Reading the leaderboard

- **score** — the ranking key, shown as a percentage:
  `validation return − ¼ × max drawdown`. Two candidates with the same return
  are not equal if one got there through a deeper drawdown. It is a ranking
  number, not a P&L.
- **validation** — the return that feeds the score. Optimistic, because the
  settings were chosen looking at it.
- **trades** — how many round trips produced that number. Fewer than 20 and
  the candidate is unranked (shown as `—`), whatever the return says.
- **test** — data the model never trained on and that did not influence the
  choice. This is the number to believe.

A positive validation with a negative test means the settings do not
generalise. That is the normal outcome on a near-random-walk target, and the
sweep will say so rather than hiding it.

## What it does not do

- Walk-forward re-fitting across folds. It reuses the single 70/15/15 split.
- Any search over features or architecture. Only the labels and the window
  change; the feature set and the transformer stay fixed so the comparison is
  like-for-like.
- Paper trading. A positive sweep is a reason to build the execution engine,
  not a reason to trade.

## This is meant to run unattended

Start it, close the laptop lid, come back later. The progress panel updates
every few seconds while it runs, the job survives browser refreshes, and
`data/reports/search.json` always holds the latest leaderboard. `Stop` finishes
the current candidate and saves it rather than discarding the work.
