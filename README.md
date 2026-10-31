# gold-neural

PAXG/USDT neural trading system. Binance spot, long-only, triple-barrier
labels, 5-minute bars by default.

`python run.py` starts one HTTPS server that serves the UI and the API.
Backfill, label previews, dataset builds, the leakage suite, and the
exchange filters are all operated from the browser.

**Current state: build-order step 1 (data layer) is complete and verified,
with the web workbench on top.** No training, no execution engine, no
orders. Nothing here can move money.

## Run it

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python run.py
```

The first boot generates a self-signed certificate and an API token, prints
both, and serves `https://<lan-ip>:8443/`. Paste the token once; the browser
remembers it.

```
  gold-neural 0.1.0
  UI          https://192.168.1.20:8443/
  symbol      PAXGUSDT 5m
  data        /home/csabi/projects/gold-neural/data
  transport   HTTPS (self-signed)
```

From there:

- **Data** — candlestick chart, coverage, partitions, gaps, backfill, verify.
- **Dataset** — triple-barrier sliders, label preview, build + write Parquet.
- **Leakage** — run the mandatory suite, see every check and its metrics.
- **Exchange** — fetch and cache `exchangeInfo` filters.
- **Console** — every job, live progress, full log.
- **Training** — deliberately empty until step 2 lands.

See `docs/web-ui.md` for the API surface, security posture, and the reasoning
behind the no-build-step frontend.

## Headless

The same operations exist as a CLI, which is what CI and cron should use:

```bash
.venv/bin/python -m gold_neural.cli backfill --years 2 --gaps
.venv/bin/python -m gold_neural.cli info
.venv/bin/python -m gold_neural.cli dataset --out data/dataset.parquet
.venv/bin/python -m gold_neural.cli leakage --out data/leakage-report.json
.venv/bin/pytest
```

`leakage` exits non-zero when any check fails, so it gates a build without a
wrapper script. `scripts/smoke.py` checks a running server end to end.

## What step 1 delivers

- Paginated historical backfill of PAXG/USDT klines from Binance spot REST.
- Parquet storage partitioned by year/month, with a SQLite index of the
  stored ranges and gap diagnostics.
- A leakage-safe feature builder (returns, EWMA/realised volatility, RSI,
  MACD, ATR, volume z-score, range features, EMA distances).
- A vectorised triple-barrier labeller that also returns the realised
  return and exit price, including gap-through fills.
- A mandatory leakage suite that fails the build on temporal leakage.
- A single-entrypoint web workbench over all of the above.

## Layout

```
run.py                 entrypoint: TLS bootstrap, token, uvicorn
gold_neural/
  config.py            YAML + GOLDNEURAL_* env config, strict key checking
  exchange/rest.py     async Binance REST: klines, exchangeInfo filters
  data/store.py        Parquet partitions + SQLite range index
  data/backfill.py     pagination, resume, bar verification
  indicators.py        causal ATR / RSI / EMA / z-score
  features.py          feature frame, feature_lag contract
  labels.py            vectorised triple barrier
  dataset.py           time-aligned feature+label frame, walk-forward split
  leakage.py           the mandatory leakage suite
  cli.py               headless backfill / dataset / labels / leakage
  server/              FastAPI app, job queue, auth, TLS, reports
web/                   single-page UI (plain ES2020, vendored chart lib)
scripts/smoke.py       post-deploy check of a running server
tests/                 pure-synthetic tests, no network required
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

**Jobs run one at a time.** Backfill, dataset builds and leakage all touch
the same Parquet store and the single SQLite writer. Serialising them costs
nothing at this scale and removes a class of partial-state bugs.

**The UI has no build step.** `python run.py` is the whole deployment. A Vite
toolchain would add `npm install` plus a `dist/` artefact that has to be
rebuilt whenever the UI changes. See `docs/web-ui.md` for the trade-off.

## Verified

Against live Binance data on 2026-10-31:

- 210,384 five-minute bars over 730 days, zero gaps, OHLC verification clean.
- 210,322 feature rows × 21 features; walk-forward split 147k / 31.5k / 31.5k.
- Leakage suite: PASS on the real series.
- `pytest`: 59 tests, no network required.
- `scripts/smoke.py` against the running server: 12/12 checks.
- Headless Chromium: chart renders, all six tabs work, a label preview runs
  from the browser and renders its per-class table, zero console errors.

Numbers from the data run are in `docs/data-report.md`.

## Not built yet

Build-order steps 2-8: training workbench, model registry, GPU probe,
paper execution, fill-simulation gate, live orders, drift loop, dashboard
polish. The plan's rule stands: no live order until paper P&L and the
walk-forward simulation agree within ~20% on the same date range.
