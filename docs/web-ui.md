# Web UI

`python run.py` starts one HTTPS server that serves the UI and the JSON API.
Everything the workbench does is reachable from the browser: there is no
required CLI step after the server starts.

## Boot

```bash
.venv/bin/python run.py
```

First boot does two things on its own:

1. Generates a self-signed certificate in `certs/` (825 days, SANs for
   `localhost`, `127.0.0.1` and the bound address).
2. Opens the SQLite index, creates `data/` and `data/reports/`.

Then open `https://<lan-ip>:8443/`. There is no login.

The certificate is self-signed, so the browser shows a warning once per
device. For anything beyond a trusted LAN, point `server.cert_path` and
`server.key_path` at a real certificate.

## Four views: Overview, Train, Risk model, Activity

The page used to expose nine panels and about sixty inputs. It now shows only
the choices that change the answer; every other knob is still on the API and
the CLI.

**Overview** — a hero row with the last price, the 24-hour change (arrow and
sign, not colour alone), the 24-hour high and low, and today's risk-model
reading (marked "draft model, not advice" when the rule has not beaten
holding; click it to open the Risk model view). Below it, the candlestick
chart (1 day / 1 week / 2 weeks) with a readout that shows open, high, low,
close and the candle's move under the cursor, then one sentence on what is
stored and three ticks: bars stored, data check, look-ahead check.
*Download latest bars* fetches only what is missing (it asks for the history
already held, so nothing is trimmed). *Check the data* runs the bar check and
then the look-ahead suite.

**Train** — three label settings (profit target, stop loss, give up after)
and a training time. One button runs the whole chain in order: build the
dataset → look-ahead check → train → backtest on the unseen newest 15%. A
failed look-ahead check ends the chain before training. Stopping training is
not a failure: the best weights are kept and the backtest still runs. The
result opens with the backtest verdict in words, then whether the model beats
guessing, then four numbers (model, just holding, trades, worst drop) and the
equity curve against holding. Past models sit in a collapsed list.

While a model trains, a **Training, live** card polls the job every two
seconds and draws two charts by epoch: the error on training data against the
error on held-out data (the gap is memorising), and the held-out balanced
accuracy against always guessing the same answer, on the same scale as the
final verdict. Its opening sentence says the best epoch so far and whether the
held-out error has stopped improving. The curves of the last run stay after it
finishes, until the server restarts.

The chain is driven by the page. If the tab closes mid-chain, the job in
flight finishes but the next step is not started.

**Risk model** — runs the drawdown-risk walk-forward (`POST /api/risk/run`,
about two and a half minutes with the default neural + trees ensemble) with
two choices: the size of fall to watch for and the window. Opens with the
verdict against holding, the reading for the next 4h bar (and the trend
rule's), four numbers (per-year return of the model and of holding, worst
drops, how often random exits did as well), an equity chart against holding
and the trend rule, and a block-by-block table. Below it, the **paper
record** (`GET /api/paper`): readings logged before their outcome, scored
against the bars that followed, with late readings marked. See
`docs/risk.md`.

**Activity** — every job with its status and full log.

A bar under the header shows whatever is running, which step of a chain it
is, and a Stop button.

**Chart colours.** The equity charts use amber for the model, blue (dashed)
for holding and magenta for the trend rule. The three were checked with the
dataviz palette validator against the dark card surface `#141a23`: lightness
band, chroma, colour-blind separation on every pair, normal-vision separation
and contrast all pass. Green and red are kept for up and down only. Each
equity chart has a legend and a hover readout with the three values, so
identity never rests on colour alone.

Removed from the page, still on the API: label preview, ambiguity and
feature-lag settings, training hyperparameters, backtest costs and minimum
confidence, exchange filters, monthly file
list, per-class tables and confusion matrix. The defaults the page uses are
the documented contracts (`stop_priority`, `feature_lag = 1`, 10 bps fee +
2 bps slippage).

## API

No `/api` route needs a token. If `server.allowed_ips` is set, requests
from other addresses get 403.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | version, symbol, bar count, TLS state, training availability |
| GET | `/api/data/info` | coverage, partitions, last dataset/leakage/verify reports |
| GET | `/api/data/bars?limit=N` | candles for the chart |
| POST | `/api/data/backfill` | `{years, start?, end?}` → job |
| POST | `/api/data/verify` | re-verify stored bars → job |
| POST | `/api/data/exchange-info` | refresh cached filters → job |
| POST | `/api/dataset/preview` | label counts only → job |
| POST | `/api/dataset/build` | build and write the dataset → job |
| POST | `/api/leakage/run` | run the leakage suite → job |
| POST | `/api/train/start` | `{minutes?, epochs, window, ...}` → training job |
| GET | `/api/train/status` | torch availability and dataset readiness |
| POST | `/api/backtest/run` | `{model_id?, fee_bps, slippage_bps, ...}` → backtest job |
| GET | `/api/backtest/last` | the persisted backtest report |
| POST | `/api/risk/run` | `{drawdown?, horizon_days?, refresh?}` → risk walk-forward job; report in `/api/data/info` as `risk`; with the configured settings, the next-bar reading is added to the predictions log |
| GET | `/api/paper` | the predictions log scored against the bars that followed (paper record) |
| GET | `/api/models` | registry listing |
| GET | `/api/models/{id}` | one model plus its runs |
| GET | `/api/jobs` | job summaries (no logs) |
| GET | `/api/jobs/{id}` | one job including its log and `series` (training: one point per epoch, newest 5,000) |
| POST | `/api/jobs/{id}/stop` | request a cooperative stop |
| WS | `/api/ws` | job progress snapshots every 500 ms when changed |

`scripts/smoke.py` exercises this surface against a running server:

```bash
.venv/bin/python scripts/smoke.py \
    --url https://127.0.0.1:8443 --insecure --leakage
```

## Security posture

- No token and no accounts: the server is meant for the owner's machine.
  The IP allowlist (`server.allowed_ips`, IPs or CIDR) is the only gate,
  and it is open by default.
- TLS by default; `--no-tls` exists only for loopback debugging.
- The server binds `0.0.0.0`, so anyone on the LAN can read the data and
  start jobs. Nothing here can place an order, but if the LAN is not
  trusted, set `allowed_ips: ["127.0.0.1"]` or run with `--host 127.0.0.1`.

## Why no React/Vite build

The plan called for React + TypeScript + Vite. This is plain ES2020 served
straight from `web/`, with Lightweight Charts vendored at
`web/vendor/lightweight-charts.standalone.production.js`.

The reason is operational: `python run.py` has to be the whole deployment.
A Vite build means `npm install`, a `web/dist` artefact that must be rebuilt
whenever the UI changes, and a second toolchain that can drift from the
Python side. The UI here is a handful of screens over a small JSON API; the
build step buys nothing and costs a failure mode.

If the UI grows into the Phase 2 dashboard (live equity, drift panel, order
log, WebSocket streams per model), the same API can back a Vite app without
changing the server. The swap is local to `web/`.

## Layout

```
run.py                       entrypoint: TLS, uvicorn
crypto_neural/server/
  app.py                     routes, static mount, WS
  jobs.py                    single-worker job queue, progress, logs
  tasks.py                   job bodies (backfill, dataset, leakage, verify)
  security.py                IP allowlist dependency
  tls.py                     self-signed certificate bootstrap
  reports.py                 data/reports/*.json persistence
crypto_neural/model.py         window transformer + GPU probe
crypto_neural/training.py      training loop, time budget, cooperative stop
crypto_neural/backtest.py      trade simulation, fees, equity curve
crypto_neural/risk.py          drawdown-risk walk-forward
crypto_neural/registry.py      SQLite models + training_runs
web/
  index.html                 single page, four views
  app.js                     API client, job stream, chart, renderers
  styles.css                 dark theme
  vendor/lightweight-charts… vendored chart library
scripts/smoke.py             post-deploy check of the running server
```

## Verified

- `pytest`: 110 tests, no network required.
- `scripts/smoke.py` against a live server: 11/11 checks, including a
  leakage run through the job API.
- Headless Chrome (2026-11-04 redesign): all four views render against the
  real data root with zero console errors; *Check the data* ran its two-step
  chain from the browser with the job bar showing step and progress. The full
  *Train and test* chain was not run from the browser in that check.
