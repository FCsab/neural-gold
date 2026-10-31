# gold-neural

PAXG/USDT neural trading system. Binance spot, long-only, triple-barrier
labels, 5-minute bars by default.

**Current state: build-order step 1 (data layer) is complete and verified.**
No training, no execution engine, no orders. Nothing here can move money.

## What step 1 delivers

- Paginated historical backfill of PAXG/USDT klines from Binance spot REST.
- Parquet storage partitioned by year/month, with a SQLite index of the
  stored ranges and gap diagnostics.
- A leakage-safe feature builder (returns, EWMA/realised volatility, RSI,
  MACD, ATR, volume z-score, range features, EMA distances).
- A vectorised triple-barrier labeller that also returns the realised
  return and exit price, including gap-through fills.
- A mandatory leakage suite that fails the build on temporal leakage.

## Quickstart

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"

cp config.example.yaml config.yaml      # optional; defaults are sane

.venv/bin/python -m gold_neural.cli backfill --years 2 --gaps
.venv/bin/python -m gold_neural.cli info
.venv/bin/python -m gold_neural.cli dataset --out data/dataset.parquet
.venv/bin/python -m gold_neural.cli leakage --out data/leakage-report.json
.venv/bin/pytest
```

`leakage` exits non-zero when any check fails, so it can gate CI or a
training run without a wrapper script.

## Layout

```
gold_neural/
  config.py           YAML + GOLDNEURAL_* env config, strict key checking
  exchange/rest.py    async Binance REST: klines, exchangeInfo filters
  data/store.py       Parquet partitions + SQLite range index
  data/backfill.py    pagination, resume, bar verification
  indicators.py       causal ATR / RSI / EMA / z-score
  features.py         feature frame, feature_lag contract
  labels.py           vectorised triple barrier
  dataset.py          time-aligned feature+label frame, walk-forward split
  leakage.py          the mandatory leakage suite
  cli.py              backfill / info / exchange-info / dataset / labels / leakage
tests/                pure-synthetic tests, no network required
```

## Design decisions worth knowing

**Precision is never hardcoded.** `exchange-info` fetches
`PRICE_FILTER.tickSize`, `LOT_SIZE.stepSize` / `minQty` / `maxQty` and
`MIN_NOTIONAL.minNotional` and caches them. The execution layer in step 4+
must round to those values and re-fetch hourly; a mid-run change has to
cancel resting orders, because a tick-size change can invalidate them.

**`feature_lag = 1` by default.** The plan says a feature may not use the
current bar's close. Rather than argue the point, the default shifts every
feature back one bar: the row paired with bar `t`'s label only sees data
through `t-1`. Set `features.feature_lag: 0` to use bar `t`'s own OHLCV —
still leak-free, and the prefix-invariance check proves it either way. The
cost of the default is one bar of information; the benefit is a contract
nobody has to re-derive.

**Prefix invariance is the real leak detector, not label shuffling.**
`check_prefix_invariance` recomputes the feature frame on `bars[:i+1]` and
requires row `i` to match the full-history value exactly. A feature that
peeks at the future cannot pass that. Shuffling labels, as specified in the
plan, is still run — but it proves the *harness* is wired correctly, not
that the features are clean: leakage and signal both vanish when you
shuffle. The docstring says so, so nobody later trusts the weaker check.

**Same-bar barrier ambiguity is resolved pessimistically.** If a bar's
high touches the target and its low touches the stop, the order is
unknowable without tick data. `ambiguity: stop_priority` books it as a
loss. Optimistic tie-breaks manufacture edge that does not exist.

**Gap fills are modelled.** A target touched at a gap fills at the open,
not at the barrier; a stop that gaps through fills at the open. The label
and `realized_return` are therefore consistent, which matters because step
2 reports expected return per signal class.

**Warm-up rows are `valid=False`, not `0`.** A label fabricated as "timeout"
because ATR was still warming up would quietly teach the model that flat
is common. They are excluded instead.

## Verified run

The checks in this repo were exercised against live Binance data:

- ~2 years of 5-minute PAXG/USDT bars backfilled and verified (OHLC
  ordering, monotonic timestamps, no duplicates, no non-positive prices).
- Leakage suite: PASS on the real series.
- `pytest`: 42 tests, all passing, no network access required.

Numbers from the actual run are in `docs/data-report.md`.

## Not built yet

Build-order steps 2-8: training workbench, model registry, GPU probe,
paper execution, fill-simulation gate, live orders, drift loop, dashboard.
The plan's rule stands: no live order until paper P&L and the walk-forward
simulation agree within ~20% on the same date range.
