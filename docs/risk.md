# The drawdown-risk model

The project now trades BTC/USDC. The PAXG/USDT (gold) research comes first,
as the record of how the model came to look the way it does; the BTC
sections at the end are the current results. **Current default: a neural
network + gradient-boosted ensemble predicting a ≥10% fall within 5 days,
learned and tested on BTC/USDT 4h history from 2017.**

`crypto_neural/risk.py`, run with `cli risk`, `POST /api/risk/run`, or the
**Risk model** view in the web UI. About ten seconds on six years of 4h bars.

## Why the project moved here (PAXG research)

Measured 2026-11-04, on PAXG/USDT 4h bars from 2020-08 to 2026-10, with
purged expanding walk-forward folds and a heavily regularised gradient-boosted
model (the strongest learner tried):

- **Direction is not predictable from this data.** Every direction target
  scored worse out of sample than guessing the class frequencies, with AUC
  ≈ 0.5: the triple-barrier races the overnight sweep used, and up/down over
  5, 20 and 60 days. Adding EUR/USDT (a dollar proxy) and BTC/USDT features
  did not change that.
- **The sweep's labels were coin flips.** At the setting it ranked best
  (`k_up 2.5, k_down 2.0, holding 192`), a label was a race between +1.4% and
  −1.2% that resolved in a median of 13 bars; 99.8% hit a barrier long before
  the 32-day cap, so the holding grid did not matter. About 40 GPU hours went
  into searching that space.
- **A neural network is the wrong tool for this data.** On the same target and
  folds, a regularised MLP (5 seeds, early stopping) scored AUC 0.514; the
  gradient-boosted model scored 0.718. About ten thousand rows whose labels
  overlap for months carry only a few dozen independent episodes.
- **Only drawdown risk showed signal.** Volatility clusters, so the size of
  the next fall is somewhat predictable even when its direction is not.

## What it does

Label: 1 if the lowest low within `horizon_bars` (default 360 = 60 days) is
at least `drawdown` (default 8%) below the close. Features: the 21 standard
ones plus long-horizon momentum, momentum in units of volatility, distance
from moving averages, drawdown from the high, and volatility (the set fixed
before the first out-of-sample test; see `RiskConfig`).

Each 90-day block gets a fresh seed-ensemble trained only on rows whose label
window closed before the block opened. The exit line is a quantile of that
model's training predictions; re-entry needs risk below 90% of the line and
30 bars flat. Which quantile a block uses is chosen by Sharpe over the earlier
blocks only, and the first two blocks simply hold. Every switch pays 10 bps
fee + 2 bps slippage. It is compared with buy-and-hold and with a no-model
trend rule (long above the 200-day average), and with a placebo: the same
exits shifted to random dates, 200 times.

A rule is a `candidate` only if it beats holding on Sharpe, has a shallower
worst drop, *and* fewer than 5% of the random-date placebos do as well.
Otherwise it is a `draft`.

## PAXG result with the PAXG defaults (8% within 60 days)

Out of sample 2022-05-05 → 2026-10-11 (4.4 years, 18 blocks):

| | per year | worst drop | Sharpe |
|---|---|---|---|
| risk model | +16.0% | 20% | 1.01 |
| buy-and-hold | +19.8% | 29% | 1.05 |
| trend rule | +14.1% | 31% | 0.89 |

Random exits did as well 15% of the time. Prediction AUC 0.52. **Status:
draft.** It cuts the worst drop the way being out of the market 18% of the
time would, and gives up return for it; the timing is not distinguishable
from luck.

## The PAXG signal is fragile

The earlier research figure (AUC 0.72; 21.8% a year against 23.1% for
holding, worst drop 21% against 29%) came from one specific setup. Two small
choices move the result a lot:

| fall measured from | first test block | AUC | model / hold per year | Sharpe model / hold |
|---|---|---|---|---|
| low (default) | 2022-05 | 0.52 | 16.0% / 19.8% | 1.01 / 1.05 |
| close | 2022-05 | 0.59 | 18.9% / 19.8% | 1.10 / 1.05 |
| close | 2022-08 | 0.72 | 22.0% / 23.1% | 1.24 / 1.18 |
| low | 2022-08 | 0.62 | 19.5% / 23.1% | 1.19 / 1.18 |

A real edge would not depend on whether the first three months of 2022 are
included. The defaults (pessimistic low-based fall, one year of training)
were fixed before this table was made and have not been changed after it;
choosing the best row would be selecting on the test period.

## What would change the answer

Only information this project does not have yet: data that actually drives
gold (the dollar index, real yields, futures positioning) rather than gold's
own price history. That widens the scope beyond Binance and is a decision for
the owner, not a tuning job.

## BTC/USDC only, first attempt (2026-11-04, superseded)

**Settings fixed before any BTC result was seen.** The fall to watch for was
scaled from PAXG's 8% by the ratio of volatilities, measured on the first year
after the relisting (training data only): BTC 42.8% a year against PAXG
12.5%, ratio 3.41, so 27.3%, rounded to **25% within 60 days**. Everything
else is unchanged.

**Data.** Binance delisted BTCUSDC from 2022-09-29 to 2023-03-12. A walk
across that gap would mix two regimes in every rolling feature and book five
months of price change as one bar, so `risk.py` starts after the last gap
longer than a day. With 250 days of feature warm-up and a year of training,
that leaves 7 test blocks.

Out of sample 2024-11-16 → 2026-08-08 (1.7 years, 7 blocks):

| | per year | worst drop | Sharpe |
|---|---|---|---|
| risk model | −26.7% | 56% | −0.59 |
| buy-and-hold | −17.9% | 54% | −0.22 |
| trend rule | −8.9% | 37% | −0.15 |

Random exits did as well 76% of the time; prediction AUC 0.42. **Draft.**
Seven blocks in one falling market cannot establish anything either way.
BTC/USDT has continuous history from 2017 at prices within basis points of
BTC/USDC, and would give the test about four times as many blocks; using it as
the research series is a decision for the owner.

## BTC, nine years (BTC/USDT history, 2026-11-04)

**Why BTC/USDT.** BTC/USDC trades within basis points of it and is the pair
the owner can use, but its history has a five-month hole; BTC/USDT is
continuous from 2017-08 (20,184 4h bars, a few exchange-maintenance gaps of
hours). `RiskConfig.data_symbol` selects the series; set it to `BTCUSDC` to
learn only from the traded pair.

**Which targets carry signal** (purged expanding walk-forward, LightGBM vs
guessing the class frequencies; the drawdown thresholds are PAXG's scaled by
the 3.41 volatility ratio, fixed before running):

| target | features | information gain | AUC |
|---|---|---|---|
| triple-barrier races (4 settings) | either | ≤ 0 | 0.43–0.52 |
| up / down over 5, 20, 60 days | either | ≤ −2% | 0.42–0.51 |
| **≥10% fall within 5 days** | **standard** | **+7.4%** | **0.647** |
| ≥10% fall within 5 days | + regime | −1.8% | 0.615 |
| ≥17% fall within 20 days | standard | −2.2% | 0.542 |
| ≥25% fall within 60 days | standard | −8.4% | 0.496 |

Same picture as PAXG: direction is not predictable; the long-horizon regime
features make things worse (patterns from earlier regimes reverse); the one
signal is short-horizon crash risk, which also won on PAXG (3% in 5 days,
AUC 0.66). The 5-day target was picked from this table, i.e. after seeing it;
what makes that defensible is the replication across two assets, and the
economic test below was then run once with the defaults fixed.

**Which model** (same target and folds, 33 blocks):

| model | AUC |
|---|---|
| LightGBM | 0.652 |
| MLP (64-32, dropout 0.3, early stopping on a purged slice) | 0.650 |
| windowed transformer (the Train view's architecture, 32 bars) | 0.600 |
| **LightGBM + MLP, averaged** | **0.664** |

With ~17,000 rows and a 5-day horizon (far less label overlap than the 60-day
target) the neural network matches the trees, and the two are only 0.62
correlated, so averaging helps. The transformer is the weakest. Hence
`model = "ensemble"`.

**Economics**, out of sample 2018-08-27 → 2026-10-15 (8.1 years, 33 blocks,
10 + 2 bps per side):

| | per year | worst drop | Sharpe |
|---|---|---|---|
| risk model (exit rule) | +26.5% | 76% | 0.71 |
| buy-and-hold | +35.9% | 77% | 0.81 |
| **trend rule (no model)** | **+37.9%** | **63%** | **0.95** |

Random exits did as well 68% of the time. **Draft.** The model does predict
10% falls (AUC 0.65), but stepping aside on that prediction loses money: a
high crash risk mostly means high volatility, and BTC's largest up days
cluster with its largest down days, so the rule misses both.

**One pre-declared follow-up, then stop.** The textbook use of a risk
forecast is sizing rather than exiting: exposure = min(1, base rate ÷
predicted risk), rebalanced when it moves by more than 10 points. It made
+22.6% a year (worst drop 72%, Sharpe 0.69; random placebo 44%) against
+35.9% for holding. Also a loss. No further variants were tried on this
period, because each one is another chance to fit noise.

**What this means.** The engine is now as good a crash-risk predictor as
this data allows with these features, and it is honest about the fact that,
on BTC, that prediction does not turn into a better strategy than holding.
The simplest rule tested, a 200-day trend filter, is the best risk-adjusted
result, and it needs no model.

**But the trend rule has not passed the luck test either** (added
2026-11-05). The placebo that judges the model was not being applied to the
trend rule. Applied the same way (its position path shifted to 200 random
dates), random exits at the same exposure and frequency did as well as the
trend rule **13% of the time** (p = 0.13; the model's is 0.68). It is the
best of the three, but its edge over holding on these eight years is not
established; it would be a `draft` under the model's own rule. This does not
change any default; it changes how strongly the README may recommend it.

## Next-bar reading (2026-11-05)

"Today's reading" used to score the newest *stored* row. With
`feature_lag = 1` that row's features stop one bar earlier, so the newest
closed 4h bar was ignored and the reading was for a bar that had already
opened. It now scores the row of the bar that opens next
(`risk.next_bar_features`): a placeholder bar is appended and the row is
built twice with different placeholder prices, and the reading is refused
unless both agree. A test checks that this row equals the one the
walk-forward computes once that bar exists. The walk-forward numbers are
unchanged (the reading is not part of them). The risk job also fetches the
bars newer than the store before it runs (`data.top_up`), so the reading is
for the actual next bar rather than for whenever the store was last filled.

## Paper record

The plan's gate is that paper P&L and the walk-forward agree within about
20% over the same dates before any real order. The walk-forward cannot
measure that against itself, so every risk run with the configured settings
now writes its next-bar reading to `data/predictions.sqlite`
(`crypto_neural/predictions.py`), stamped with the time it was made:

- **The first reading for a bar wins.** Rerunning cannot replace it, so
  re-rolling until the answer looks right is impossible rather than merely
  discouraged. The key is (bar, fingerprint of the risk settings); runs with
  other settings from the UI are experiments and are not logged.
- **Late readings trade late.** A reading recorded more than ten minutes
  after its bar opened is filled at the first bar that opens after it was
  recorded. The walk-forward always fills at the bar's open; that difference
  is part of what the gate measures.
- **Between readings the position is held**, as a person following the log
  would hold it on days nothing ran.

`cli paper` and the Risk model page score the log: realised outcomes with
the walk-forward's own label, forecast error (Brier) against a constant that
knows the realised rate, and paper returns for the risk rule, the trend rule
and holding after 10 + 2 bps per side. For on-time readings set
`server.record_readings: true`; the server then runs the job about 45 seconds
after each 4h bar opens. Nothing in this path can place an order.

The first reading was recorded on 2026-11-05 at 21:44 UTC for the 20:00
bar: late, as manual runs are.

### The gate, measured (added 2026-11-07)

`score_paper` now reports `gate`: the paper path against a simulation of the
same readings over the same dates, each filled at its own bar's open as the
walk-forward fills it. The difference is what execution cost (late readings,
runs that did not happen), which is the part a walk-forward cannot see. The
thresholds were fixed in `predictions.py` before any record was long enough
to judge, and must not move after one is:

- at least **90 days** of record;
- an on-time reading for at least **90% of the bars** in that window;
- total return of paper and simulation within **20%**, measured against the
  simulated P&L or 1% of capital, whichever is larger (a ratio of two
  near-zero numbers means nothing).

The gate is reported for the risk rule and, separately, for the trend rule.
Passing it says the paper record matches the simulation, not that the rule
beats holding; the walk-forward already says neither has an edge that is
distinguishable from luck.

## Funding rates as new information: pre-registration (2026-11-05)

Written before any funding data was downloaded or looked at.

**Why this and not another variant.** Every test so far used BTC's own
price. The memory of this project says the only open avenue is information
that drives BTC from outside its own price history. Perpetual-futures
funding is the cheapest such source: it is what leveraged longs pay shorts
every eight hours, so it measures how crowded the long side is, and a
crowded long side is the mechanism behind 10%-in-5-days liquidation
cascades. Binance publishes it without keys from 2019-09-10.

**Design, fixed now.**

- Data: BTCUSDT USDⓈ-M perpetual funding settlements, 8-hourly.
- Four features, as of each bar, from settlements strictly before the bar
  opens: the last rate, the mean of the last 9 (3 days), the mean of the last
  90 (30 days), and (3-day mean − 30-day mean) ÷ the standard deviation of
  the last 90.
- Both arms run on identical bars (BTCUSDT 4h from the first bar with 90
  settlements behind it), identical folds and the unchanged `RiskConfig`
  defaults (ensemble, 10% within 5 days, one year of training, 90-day
  blocks, 10 + 2 bps). The only difference is the four extra columns.
- Primary measure: pooled out-of-sample AUC on the same rows. Secondary:
  the share of blocks (with both outcomes present) in which the funding arm
  has the higher AUC, and the economic result against holding.

**Decision rule.** Funding counts as adding information only if the pooled
AUC rises by at least 0.010 *and* the funding arm wins at least 60% of
blocks. If it does, the defaults still do not change in this session: the
result goes to the owner, and the predictions log is the forward test. If it
does not, no variants (other windows, open interest, basis) are tried on
this period.

**Result (run once, 2026-11-05 evening, `cli funding`).** Both look-ahead
checks passed, including the one that hides later settlements. Bars:
BTCUSDT 4h from 2019-10-10 (the first with 90 settlements behind it);
out of sample 2020-10-17 → 2026-09-16, 5.9 years, 24 blocks, the same rows in
both arms. The report is `data/reports/funding.json`.

| | without funding | with funding | rule |
|---|---|---|---|
| pooled out-of-sample AUC | 0.657 | 0.674 (+0.017) | ≥ +0.010 |
| blocks with the higher AUC | | 15 of 22 (68%) | ≥ 60% |

By the rule written before the data was seen, **funding adds information.**
As pre-registered, the defaults do not change. That is the owner's call, and
the forward test is the predictions log.

What the result does *not* say:

- **It does not make money yet.** Costs are 10 + 2 bps per side, and holding
  made +37.8%/yr (worst drop 77%, Sharpe 0.84). The exit rule without funding
  made +38.0%/yr (76%, 0.86, placebo p = 0.26). With funding it made
  +41.7%/yr (79%, 0.90, placebo p = 0.10). Neither has timing that is
  distinguishable from luck at 5%, and the funding arm's worst drop is larger
  than holding's.
- **Per-block skill is weak in both arms.** Several blocks are below 0.5 in
  both (2021-07 → 2021-12, 2023), so much of the pooled AUC comes from
  separating calm periods from turbulent ones, not from timing within a
  block. The largest single gain (2024-09: 0.33 → 0.75) was on only 24 falls.
- **It is one test on one history.** The margins (0.017 against 0.010, and
  68% against 60%) are modest. No variants (other windows, open interest,
  basis) are to be tried on this period; doing so now would be choosing
  after seeing.

## Volatility sizing: pre-registration (2026-11-07)

Written before any code for it ran on real data, and committed separately
so the history shows that.

**Why this.** On 5-minute bars a tree model matches the transformer, and
both predict *whether* price moves (barrier hit vs timeout, AUC 0.62), not
*which way* (AUC 0.52; `docs/training.md`). On 4h bars the crash-risk model
predicts large falls (AUC 0.65), and much of that is telling calm periods
from turbulent ones. Volatility is the one thing the models here demonstrably
forecast. A long-only holder can use it without forecasting direction: hold
less when the next day is expected to be turbulent. Volatility-managed
portfolios are known to improve risk-adjusted returns on other assets. On BTC
it is untested here.

**Design, fixed now.**

- Data: BTCUSDT 4h bars after the last gap longer than 7 days. The same
  expanding folds as the risk model: one year (2,190 bars) before the first
  90-day (540-bar) test block, abutting blocks, `RiskConfig` defaults.
- Target: log realised volatility of the returns a position decided at bar
  *t* earns over the next day: `log(sqrt(sum r²))` over the six open-to-open
  log returns from `open[t]` to `open[t+6]`. The label at *t* is decided at
  bar *t+6*. Training rows whose label reaches into the test block are purged.
- Features: the risk model's default feature set (lag 1, look-ahead checked).
- Model: LightGBM regression on the target, with the risk model's
  regularised tree settings, averaged over its 5 seeds.
- No-model forecast: EWMA (span 30 bars) of squared 4h log returns through
  bar *t−1*, scaled to six bars: `log(sqrt(6 * ewma_var))`.
- Sizing, the same for both forecasts: weight `min(1, target / forecast)`,
  with no leverage and no shorting. `target` for each block is the median
  forecast over that block's training rows, so it comes from the past only.
  A trade happens only if the weight moves by at least 0.10 from the current
  position. Every change in position pays 10 + 2 bps on the amount traded.
- Arms: hold (weight 1), EWMA-sized, model-sized. All three run over the same
  out-of-sample bars.

**Decision rule.** Volatility sizing with the model counts as working only if
all four hold over the whole out-of-sample period:

1. The model forecasts better than EWMA: lower mean squared error of log
   volatility on the pooled test rows, *and* lower in at least 60% of blocks.
2. Model-sized Sharpe is above both hold's and EWMA-sized.
3. Model-sized worst drop is smaller than hold's.
4. Model-sized timing beats chance: the shifted-path placebo p ≤ 0.05.

If 2 to 4 hold for EWMA-sized but 1 fails, the result is that sizing works
and the model adds nothing over a formula. That still counts as a result,
reported as such. It runs once. No other spans, horizons, bands or targets
get tried on this period.

### Volatility sizing: result (run once, 2026-11-07)

`cli volsize`, BTCUSDT 4h, out of sample 2018-08-27 → 2026-10-15 (8.1 years,
33 blocks), 10 + 2 bps per side. Report: `data/reports/volsizing.json`.

| | per year | worst drop | Sharpe | exposure | placebo p |
|---|---|---|---|---|---|
| hold | +35.9% | 77.0% | 0.81 | 100% | |
| EWMA-sized | +27.3% | 74.2% | 0.74 | 89% | 0.74 |
| model-sized | +29.1% | 74.8% | 0.77 | 90% | 0.52 |

| criterion | result |
|---|---|
| 1. model forecasts better than EWMA | **yes**: MSE 0.329 vs 0.409, better in 32 of 33 blocks |
| 2. Sharpe above hold and EWMA | no (0.77 vs 0.81) |
| 3. worst drop below hold | yes, by 2 points |
| 4. timing beats chance | no (p = 0.52) |

**By the pre-registered rule it does not work.** What it does show:

- **The volatility forecast is real skill.** LightGBM beats the EWMA formula
  almost every quarter for eight years. This is the strongest out-of-sample
  result in the project, and it is a forecast of magnitude, not direction.
- **Turning it into sizing does not beat holding BTC.** Cutting exposure
  before turbulent days gave up more return than it saved in drawdown. On
  BTC the turbulent stretches include the large rallies, so holding less
  when volatility is high misses them as often as it misses the crashes.
  The "volatility-managed portfolio" effect seen in equities did not show up
  here.
- Nothing changes in the defaults. As registered, no other spans, horizons,
  bands or targets will be tried on this period.
