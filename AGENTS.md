# AGENTS.md

Guidance for coding agents working in this repository.

## What this is

BTC/USDC trading research system on Binance spot. Long-only. Two models: a
crash-risk model (`risk.py`: MLP + LightGBM ensemble, 4h bars, learned from
BTC/USDT history) and the original triple-barrier transformer (5-minute
bars). One FastAPI HTTPS server serves
both a JSON API and a no-build-step web UI.

The symbol is BTC/USDC (PAXG is not available to the owner, in Hungary).
Read `docs/risk.md` before proposing model work: on PAXG, direction was
measured as unpredictable, a neural network lost to gradient-boosted trees,
and the drawdown signal was fragile. A sweep over more label settings is not
new work. On BTC the same held: direction unpredictable, short-horizon crash
risk predictable (AUC 0.65), an MLP equal to LightGBM and their average
better, and still no exit or sizing rule beat holding. BTCUSDC has a 164-day
listing gap (2022-09 → 2023-03), so the risk model learns from BTCUSDT
(`risk.data_symbol`); the 5-minute transformer does the same through
`data.training_symbol` (default BTCUSDT), and its backtest buys only when the
expected gain covers the round-trip cost. The risk job imports PyTorch only in a child process;
keep it out of the server process.

**Current scope: step 1 (data layer) is complete; step 2 is partial** — one
model trains end to end and a registry records it. There is no execution
engine, no order code, and nothing that can move money. Do not add any.

The governing rule from the plan: **no live order until paper P&L and the
walk-forward simulation agree within ~20% on the same date range.** Work that
skips earlier steps toward that gate is out of scope.

## Setup and commands

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # training only
.venv/bin/python run.py                    # HTTP UI + API on https://<lan-ip>:8443/
.venv/bin/pytest                           # 185 tests, no network
```

Headless equivalents (use these in CI, cron, and agent verification):

```bash
.venv/bin/python -m crypto_neural.cli backfill --years 2 --gaps
.venv/bin/python -m crypto_neural.cli info
.venv/bin/python -m crypto_neural.cli dataset --out data/dataset.parquet
.venv/bin/python -m crypto_neural.cli leakage --out data/leakage-report.json
.venv/bin/python -m crypto_neural.cli risk   # needs 4h bars; logs the next-bar reading
.venv/bin/python -m crypto_neural.cli paper  # score the predictions log
.venv/bin/python -m crypto_neural.cli funding  # pre-registered funding test; run once
.venv/bin/python -m crypto_neural.cli volsize  # pre-registered volatility-sizing test; run once
.venv/bin/python scripts/smoke.py          # against a running server
```

`leakage` exits non-zero on any failure; that is the build gate, not a warning.
The console script `crypto-neural` maps to the same CLI.

`run.py` is the only entrypoint needed for the UI: first boot generates a
self-signed cert in `certs/` and serves the page. There is no login. `--no-tls`, `--host`, `--port` exist for bootstrap
and tests.

## Map of the code

```
run.py                 TLS bootstrap, uvicorn
crypto_neural/
  config.py            frozen dataclasses; strict key checking; env overrides
  exchange/rest.py     async Binance REST (klines, exchangeInfo filters)
  data/store.py        Parquet partitions by year/month + SQLite range index
  data/backfill.py     pagination, resume, bar verification
  indicators.py        causal ATR / RSI / EMA / z-score
  features.py          feature frame + the feature_lag contract
  labels.py            vectorised triple barrier; drawdown label
  dataset.py           feature/label merge, walk-forward split
  leakage.py           the mandatory leakage suite
  model.py             window transformer + real GPU probe
  training.py          child-process training loop: budget, stop, metrics
  backtest.py          trade simulation with fees and slippage
  volsizing.py         pre-registered volatility-sizing walk-forward
  risk.py              drawdown-risk walk-forward: purged folds, past-only
                       rule choice, placebo, baselines, next-bar reading
  predictions.py       predictions log (first reading per bar wins) and the
                       paper record scored from it; never places orders
  registry.py          SQLite models + training_runs
  cli.py               headless backfill / dataset / labels / leakage
  server/              FastAPI app, job queue, IP allowlist, TLS, reports, reading scheduler
web/                   one page, plain ES2020, vendored chart library
tests/                 synthetic fixtures only, no network
```

## Invariants — do not break these

**Look-ahead safety.** `features.feature_lag` defaults to `1`: the feature row
at bar `t` sees data through `t-1` only. `leakage.check_prefix_invariance`
recomputes features on `bars[:i+1]` and requires row `i` to match the
full-history value exactly. That check is the real leak detector and must
never be relaxed, skipped, or sampled down to nothing.

**Alignment is by time, never by position.** `dataset.build_dataset` merges on
`open_time` with `validate="one_to_one"`. A positional join is the classic
silent-lookahead bug here; if you touch the merge, keep the validation and keep
the test that proves one-row misalignment is rejected.

**Feature column order is defined once.** `features.feature_names(cfg)` is the
source of truth, and `required_history(cfg)` defines the warm-up gap. Adding a
feature means updating both, the tests, and anything that indexes by column
position.

**Ambiguous bars are resolved pessimistically.** A bar touching both barriers is
booked as a loss under the default `ambiguity: stop_priority`. Gaps fill at the
open, not at the barrier. Do not add optimistic tie-breaks to make a backtest
look better.

**Warm-up rows are excluded, not fabricated.** Incomplete labels are dropped or
marked invalid; never defaulted to class `0` / "timeout".

**Costs are real.** Any performance claim must include `fee_bps` +
`slippage_bps`. If the profit target is smaller than the round-trip cost, no
model can win — say so plainly instead of reporting the percentage.

**Selection never uses the block being reported.** `risk.py` chooses each
block's exit rule from the earlier blocks only, and the transformer's early
stopping uses validation. Choosing on test would manufacture the number; so
would changing `RiskConfig` defaults after seeing results.

**Purge overlapping labels.** A label is decided by prices up to its exit.
Training rows whose exit falls inside the next block are dropped
(`training.purge_overlap`, `risk.risk_folds`). Do not remove the purge.

**The predictions log is append-only and first-wins.** A reading is keyed by
its bar and the risk-settings fingerprint; never update, delete, or let a
rerun replace one, and never fill a late reading at its own bar's open.
That log is the paper side of the gate.

**One writer, one job at a time.** Backfill, dataset builds, and leakage share
one Parquet store and one SQLite writer. Jobs are serialised by design. Only
the server process writes SQLite; the training child writes checkpoint files
only.

**No hardcoded exchange precision.** `exchange-info` caches `tickSize`,
`stepSize` / `minQty` / `maxQty`, and `minNotional`. Rounding is done from
those values, and any future execution layer must re-fetch hourly.

## Conventions

- Python 3.11+, `from __future__ import annotations` at the top of each module,
  full type hints on public functions.
- Config is a graph of frozen dataclasses built by `config.load_config`.
  Precedence: dataclass defaults < `config.yaml` < `CRYPTONEURAL_<SECTION>_<KEY>`.
  Unknown keys raise `ConfigError` on purpose — a typo like `k_up: 20` instead
  of `2.0` must fail loudly, not silently change what the model predicts.
  Tuple-typed fields stay tuples (hashability and equality depend on it).
- One module, one concern. The exchange layer is the only async part; analysis
  code is pure pandas/numpy and testable without I/O.
- Results are dataclasses with `to_dict()` / `to_json()` / `format()`, and
  machine-readable artefacts land in `data/reports/`. Reuse that shape.
- Docstrings explain *why* and *what fails otherwise*, not what the code
  obviously does. Existing modules set the tone; match it.
- House style is British (`serialise`, `vectorised`, `labeller`, `behaviour`),
  ~88 columns, 4-space indent, double quotes.
- Errors are explicit (`ConfigError`, `LeakageError`, `ValueError`); avoid
  `except Exception` except at process boundaries where a fallback is
  deliberate and logged.
- The UI has no build step: edit `web/app.js`, `web/index.html`,
  `web/styles.css` directly. Do not introduce npm, a bundler, or a `dist/`.
- UI text is plain English: every control gets a help line, results open with a
  verdict in words before numbers.

## Tests

- `pytest` only, `tests/` mirrors module names (`features.py` → `test_features.py`).
- Tests are synthetic and offline. `conftest.synthetic_bars()` and
  `conftest.flat_bars()` are the standard fixtures; never call Binance from a
  test and never require `data/`.
- Server tests use `fastapi.testclient.TestClient` with
  `ServerConfig(tls_enabled=False)` and a `tmp_path` data root; `wait_for_job`
  polls a job to a terminal state.
- When changing the feature builder or labeller, add or update a prefix-
  invariance style test: recompute on a truncated series and assert equality on
  the overlapping row.
- Keep the suite fast; a new test that sleeps or downloads is a redesign, not a
  test.

## Security and secrets

- There is no API token, by the owner's choice: the server runs on their own
  machine. The only gate is the optional `server.allowed_ips` allowlist
  (empty = any source). The server binds `0.0.0.0`, so anyone on the LAN can
  reach it; that is acceptable only because nothing here can move money.
  OpenAPI/docs stay disabled. Certificates live in `certs/`.
- `.gitignore` already excludes `data/`, `certs/`, `.token`, `*.pt`, `*.parquet`,
  `*.sqlite*`. Do not add tracked fixtures there, and do not "helpfully" commit
  a generated report.

## Editing rules

- Keep changes inside the working set; do not refactor unrelated modules while
  fixing one.
- Never loosen a leakage check or a dataset invariant to make a test pass.
  If a check fails, the pipeline is wrong.
- Do not change `feature_lag`, `ambiguity`, or cost defaults without saying so
  explicitly in the change: each is a documented contract.
- No execution, order placement, key handling, or withdrawal code. Out of scope
  until the paper-vs-walk-forward gate is met.
- Web changes: no framework, no build, no CDN at runtime (the chart library is
  vendored).

## Definition of done

1. `.venv/bin/pytest` is green.
2. If you touched features, labels, dataset, or leakage: `cli leakage` reports
   PASS and exits 0.
3. If you touched the server or the UI: `scripts/smoke.py` against a running
   `run.py` reports all 10 checks (12 with `--leakage`), or you state which
   check you could not run and why.
4. Metrics are reported as measured, including the baseline they are compared
   against. A model that loses to "always guess the majority class" is recorded
   as `draft`, not promoted.
