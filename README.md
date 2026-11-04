# gold-neural

PAXG/USDT neural trading system. Binance spot, long-only, triple-barrier
labels, 5-minute bars by default.

`python run.py` starts one HTTPS server that serves the UI and the API.
Backfill, label previews, dataset builds, the leakage suite, the exchange
filters, and training are all operated from the browser.

**Current state: step 1 (data layer) is complete, and step 2 is partly in
(one model trains end to end, with a registry).** No execution engine, no
orders. Nothing here can move money.

## Run it

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python run.py
```

The torch line is only needed for training. On an AMD GPU use
`--index-url https://download.pytorch.org/whl/rocm6.4` instead; the device is
probed per run either way.

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

The page has three tabs — **Data** (market data, labels, look-ahead, exchange
rules), **Model** (training, walk-forward, research), and **Backtest** — plus
an Activity log. A status dot per tab and a “next step” banner always name the
one thing left to do. Every control has a plain-English help line, and results
open with a verdict in words before the numbers.

- **Market data** — candlestick chart, what is stored, data check, backfill.
- **Labels & dataset** — barrier settings, label preview, build Parquet.
- **Look-ahead check** — the mandatory leakage suite, with metrics.
- **Training** — run for N minutes or until you press Stop.
- **Backtest** — replay the unseen test slice as trades with fees and
  slippage, equity curve against buy-and-hold.
- **Walk-forward** — trains a model on everything before each 90-day block,
  trades that block, and slides forward across six years. Two policies: trade
  the barrier signal, or stay long by default and step aside when the model
  says stop — the direct test against just holding. You get the
  distribution of quarters, not one lucky split: median, worst, how often it
  clears +5%, and how often it beats simply holding. Runs from the browser;
  `scripts/overnight.py` is the same engine for an unattended machine.
- **Research** — starts that campaign from the browser as a detached,
  resumable process. Progress, log, leaderboard, and two charts (quarter by
  quarter, cumulative against holding) update live; it keeps running if the
  tab or the server closes.
- **Exchange rules** — fetch and cache `exchangeInfo` filters.
- **Activity** — every job, live progress, full log.

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
./scripts/start_night.sh                    # unattended hold-policy sweep
```

`scripts/start_night.sh` launches the same walk-forward engine in the
background, with the hold policy as the default, and ranks settings by excess
return over buy-and-hold. It resumes from the folds already on disk; see
`docs/overnight.md` for the grid, the stop-threshold rule, and how to read the
output.

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
  model.py             window transformer + mandatory GPU probe
  training.py          training loop: time budget, cooperative stop, metrics
  backtest.py          trade simulation: fees, slippage, equity, buy-and-hold
  search.py            candidate grid, cost pruning, validation scoring
  registry.py          SQLite model + training-run registry
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
- `pytest`: 110 tests, no network required.
- `scripts/smoke.py` against the running server: 12/12 checks.
- Headless Chromium: chart renders, all six tabs work, a label preview and a
  training run both start and finish from the browser, zero console errors.
- Guidance checked against both an empty data root and the populated one: the
  next-step banner, tab dots, intro card and empty states all follow state.
- Auto-search: a real 3-candidate sweep and a live 2-candidate sweep, watched
  in the browser — the leaderboard gained a row while the job was still
  running, and the final state read `done` with 2/2 candidates.
- Training on the RX 7600 XT (ROCm): 10 epochs over 210k rows in 121 s,
  stopped exactly at the 2-minute budget, checkpoint written, balanced
  accuracy 0.403 against a 0.485 majority baseline — recorded as `draft`,
  because it did not earn `candidate`.
- Backtest of that model over 109 days of unseen data: 3,520 trades,
  10,000 → 2.15 USDT. Not a model failure first — the profit target is 16 bps
  while a round trip costs 24 bps, so no model can win at those label
  settings. The UI says so instead of reporting a bare percentage.

Numbers from the data run are in `docs/data-report.md`; the training run and
its honest result are in `docs/training.md`.

## Not built yet

The rest of step 2 (compare view, shadow/live promotion, predictions log),
then steps 4-8: paper execution, fill-simulation gate, live orders, drift
loop, dashboard polish. The plan's rule stands: no live order until paper
P&L and the walk-forward simulation agree within ~20% on the same date
range.
