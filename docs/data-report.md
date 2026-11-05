# Data layer report

> **Measured on PAXG/USDT**, before the project switched to BTC/USDC on
> 2026-11-04 (PAXG is not available in Hungary). The method still applies;
> the numbers are PAXG's and have not been re-measured on BTC.

Result of the step-1 verification run against live Binance spot data on
2026-10-31. Regenerate everything with the commands in `README.md`; the
artifacts themselves (`data/`) are git-ignored.

## Bars

| | |
|---|---|
| Symbol / interval | PAXGUSDT / 5m |
| Bars stored | 210,384 |
| Range | 2024-10-31 05:50 UTC → 2026-10-31 17:45 UTC |
| Span | 730 days |
| REST pages | 211 (1,000 klines each) |
| Unclosed bars skipped | 1 |
| Backfill wall time | 103.7 s at 0.15 s request spacing |
| Partitions | 25 (year=YYYY / month=MM) |
| Gaps > 1.5 × bar interval | 0 |

`verify_bars` found no duplicate timestamps, no non-monotonic ordering, no
OHLC ordering violations, no non-positive prices, and no `close_time <=
open_time` rows.

## Dataset

| | |
|---|---|
| Feature rows | 210,322 |
| Rows dropped (warm-up) | 50 |
| Features | 21 |
| Train / validation / test | 147,225 / 31,548 / 31,549 |

Label distribution (k_up = 2.0, k_down = 1.5, H = 12, ATR period 14,
`ambiguity = stop_priority`):

| Label | Rows | Share |
|---|---|---|
| +1 target | 71,454 | 34.0% |
| 0 timeout | 42,521 | 20.2% |
| -1 stop | 96,347 | 45.8% |

The stop share is a sanity check, not a tuning result: for a driftless
series the probability of touching the nearer barrier first is
`k_up / (k_up + k_down) = 2.0 / 3.5 = 57.1%`. Observed among resolved
(non-timeout) rows: `96,347 / 167,801 = 57.4%`. A labeller with a
directional bias, an off-by-one horizon, or a leaky ATR would not land on
that number.

## Leakage suite

```
LEAKAGE SUITE: PASS
  bars=210384 rows=210322 labels={'long': 71454, 'flat': 42521, 'stop': 96347}

  [PASS] features_prefix_invariant
         20 feature rows recomputed from truncated history match the
         full-history values exactly
  [PASS] labels_depend_on_future
         10 sampled labels flipped to +1 when a future bar was forced
         through the upper barrier
  [PASS] labels_ignore_far_future
         10 labels survived truncation past their forward window
  [PASS] dataset_invariants
         all structural invariants hold
  [PASS] shuffled_label_correlation
         max |corr(feature, shuffled label)| = 0.00335 (range_pct),
         threshold 0.00872 = 4/sqrt(n=210322)
  [PASS] feature_label_mutual_information
         informational: range_ma=0.0258, atr_pct=0.0255, rv_48=0.0191,
         ewma_vol_48=0.0185, ewma_vol_12=0.0161
```

The mutual-information leaders are all volatility features. That is the
expected shape of real signal here: a volatility regime changes the odds
of touching a 2×ATR barrier inside 12 bars versus timing out. It is
reported, never gated — a gate on MI would reject genuine edge.

## Exchange filters cached from `/api/v3/exchangeInfo`

Fetched 2026-10-31, stored in `data/exchange_info.json`. These must be
re-read hourly at runtime; nothing downstream may hardcode them.

| Filter | Value |
|---|---|
| `PRICE_FILTER.tickSize` | 0.01 |
| `LOT_SIZE.stepSize` | 0.0001 |
| `LOT_SIZE.minQty` / `maxQty` | 0.0001 / 9000 |
| `NOTIONAL.minNotional` | 5.0 USDT |
| `NOTIONAL.maxNotional` | 9,000,000 USDT |
| `MAX_NUM_ORDERS` | 200 |

## Test suite

42 tests, all passing, no network access required:

```
.venv/bin/pytest -q
42 passed
```
