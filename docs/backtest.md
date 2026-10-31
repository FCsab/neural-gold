# Backtest

Classification metrics answer "does it beat guessing". The backtest answers
"would it have made money", which is a different question and often a less
comfortable answer.

## What it does

1. Loads the model's checkpoint and rebuilds the exact split it was trained on.
2. Scores the most recent 15% of the data — rows the model never saw.
3. Turns every long signal into a trade:
   - **entry** at the next bar's open (you cannot trade at a close that has
     already printed),
   - **exit** at the price the label recorded: the barrier that was hit, or the
     timeout close,
   - **fees** (10 bps) and **slippage** (2 bps) charged on both sides.
4. Compounds the result into an equity curve and compares it with simply
   holding PAXG over the same window.

One position at a time, long only, all-in by default. Costs are configurable
in the UI, so the sensitivity is visible rather than assumed.

## Measured on this machine

Model `PAXGUSDT-5m-w64-20261031-203150`, test window 2026-07-14 → 2026-10-31
(109.5 days, 31,550 scored rows), 10 bps fee + 2 bps slippage per side:

```
initial / final      10,000.00 → 2.15 USDT
total return         -99.98%
buy and hold         -3.23%
trades               3,520
win rate             0.8%
max drawdown         99.98%
sharpe (daily)       -51.6
```

The verdict the UI shows:

> Even the trades that went the right way captured only 8 bps on average,
> while a round trip costs 24 bps in fees and slippage. No model can be
> profitable at these label settings.

## Why -100% is the right answer here, not a bug

The label configuration from the Dataset tab is `k_up = 2.0`, `k_down = 1.5`
with a 12-bar hold. On 5-minute PAXG bars the mean ATR is **8.1 bps**, so:

| | |
|---|---|
| profit target (2.0 × ATR) | 16.2 bps |
| stop loss (1.5 × ATR) | 12.2 bps |
| round trip at 10 + 2 bps per side | **24 bps** |

The cost of taking a trade is larger than the entire profit target. A perfect
oracle would still lose about 8 bps per winning trade. The 0.8% win rate is
not the model being uniquely awful; it is what "every trade is net negative"
looks like when you measure wins after costs.

This is exactly the failure mode the plan warned about, arriving early and
cheaply instead of after real money.

## What to change

The lever is the label configuration, not the network:

- **Widen the barriers.** Target at least ~4× the round-trip cost. With 24 bps
  of cost that means roughly 100 bps of target, so `k_up` in the 8-12 range on
  this ATR.
- **Give it time.** A 12-bar hold (one hour) rarely reaches a 1% move; 96-288
  bars (8 hours to a day) is more plausible.
- Then rebuild the dataset, retrain, and re-run the backtest. Changing the
  label config invalidates both the dataset and the model.

The same rule applies before any live trading: if the target is not several
times the cost, no amount of model tuning helps.

## What is not modelled

- Partial fills, queue position, and book depth (the plan's Phase 2 gate).
- Funding (irrelevant: spot only) and borrow (long-only).
- Intrabar ordering: the exit assumes the `ambiguity` setting from the label
  config, which books same-bar ties pessimistically.
- Walk-forward re-fitting: one 85/15 split, not rolling folds.
- Paper-vs-simulation divergence: that is build-order step 5 and needs the
  execution engine.

So treat a positive backtest as a reason to continue, never as a profit
forecast.
