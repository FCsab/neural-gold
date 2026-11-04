# Web UI

`python run.py` starts one HTTPS server that serves the UI and the JSON API.
Everything the workbench does is reachable from the browser: there is no
required CLI step after the server starts.

## Boot

```bash
.venv/bin/python run.py
```

First boot does three things on its own:

1. Generates a self-signed certificate in `certs/` (825 days, SANs for
   `localhost`, `127.0.0.1` and the bound address).
2. Generates a bearer token and writes it to `.token` (mode 0600), printing
   it once to the console.
3. Opens the SQLite index, creates `data/` and `data/reports/`.

Then open `https://<lan-ip>:8443/`, paste the token, and the browser keeps it
in `localStorage`.

The certificate is self-signed, so the browser shows a warning once per
device. For anything beyond a trusted LAN, point `server.cert_path` and
`server.key_path` at a real certificate.

## Three tabs: Data, Model, Backtest

The work is grouped by what you are doing, not by pipeline order:

- **Data** — market data, labels & dataset, look-ahead check, exchange rules.
- **Model** — training, walk-forward, and the detached research campaign.
- **Backtest** — the unseen test slice, traded with costs.

Sections stack inside a tab with a divider between them. A status dot per tab
and a next-step banner always name the one thing left to do.

Each tab shows a status dot (grey = not done, green = done, red = failed), and
a **Next step** banner above the tabs always names the one thing to do next and
takes you there. A first-run card explains what the whole thing is and can be
dismissed. Every panel opens with one sentence of context, and every control
has a line of plain-English help under it.

Jargon is kept but always explained:

| Term | What it says in the UI |
|---|---|
| `k_up` / `k_down` | profit target / stop loss, in multiples of typical movement |
| `max_holding_bars` | max holding time |
| `feature_lag` | information delay |
| ATR | “typical bar movement” |
| leakage | look-ahead check — the model must not see the future |
| partitions | monthly files |
| gaps | missing stretches |

Training results open with a verdict in words before the numbers: *“No edge
yet. Balanced accuracy 39.3% is below the 48.5% of always guessing the most
common outcome.”* The tables underneath carry a one-line explanation of what
precision, recall and the confusion matrix mean.

## What each section does

**Market data** — candlestick chart of stored bars (24h / 7d / 30d), what is
stored, a data check that looks for broken candles, duplicates and missing
stretches, and a download form that resumes where the stored data ends.

**Labels & dataset** — the triple-barrier settings with a help line each.
*Preview the counts* shows how many bars would fall into each outcome without
writing anything; *Build the dataset* writes `data/dataset.parquet` and records
the exact parameters used.

**Look-ahead check** — runs the mandatory suite with the current label
settings and shows every check with its metrics. A failure turns the tab dot
red and the next-step banner says not to train.

**Training** — start with a minutes budget, or leave it empty and press Stop
when you have seen enough. Shows the live epoch counter and log, then the
verdict, per-class table, confusion matrix and validation loss. After a
restart the panel falls back to the newest model in the registry, so results
do not disappear with the job list.

**Backtest** — replays the unseen test slice as trades, charging fees and
slippage on both sides, and draws the equity curve against simply holding
PAXG. Leads with a verdict; flags the case where the profit target is smaller
than the round-trip cost, because then nothing can be profitable.

**Walk-forward** — the honest evaluation. Pick a timeframe and a region of
label settings, plus how many seeds; each setting is trained and traded across
every 90-day block of the last six years. The leaderboard shows median quarter,
worst quarter, share clearing +5%, and share beating buy-and-hold — the last
one being the only number that says whether the model adds anything over doing
nothing. Runs resume: a restart skips the folds already on disk. See
`docs/overnight.md`.

**Research** — starts the detached, resumable campaign from the browser. It is
the same engine as `scripts/overnight_ctl.sh`, but the form also exposes the
hold-policy threshold grid, the seed list, the block sizes and the costs. The
campaign runs in its own process: closing the tab or restarting the server does
not stop it. The panel shows live progress, the recent log, a leaderboard ranked
by median excess over holding, and two charts for the selected setting — paired
bars for each quarter (strategy vs buy-and-hold) and a cumulative line. Click a
leaderboard row to chart that setting; partial settings are labelled `7/19` and
dimmed. Stop with the button or `touch data/overnight/STOP`.

The older single-split auto-search still exists on the API
(`/api/search/*`) but is no longer in the UI: it ranked settings on one
validation window, which produced validation +41% and test −4% from the same
model.

**Exchange rules** — cached `exchangeInfo` filters (tick size, step size,
notional limits). Never hardcoded anywhere in the codebase.

**Activity** — every job with its log. Jobs are queued and run one at a time,
deliberately: they share the Parquet store and the single SQLite writer.

## API

All `/api` routes require `Authorization: Bearer <token>`. The static page
does not, and contains no data.

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
| POST | `/api/walkforward/start` | `{interval, k_up[], k_down[], holding[], window[], seeds[], ...}` → sweep job |
| GET | `/api/walkforward/last` | leaderboard + progress, rewritten after each fold |
| POST | `/api/research/start` | start the detached campaign (`{interval, policy, grids, seeds, ...}`) |
| GET | `/api/research/status?spec=&folds=` | campaign status, leaderboard, and per-fold rows for charting |
| POST | `/api/research/stop` | write `<data>/overnight/STOP`; the fold in flight finishes |
| POST | `/api/search/start` | legacy single-split sweep (kept for the API) |
| GET | `/api/search/last` | legacy sweep report |
| GET | `/api/models` | registry listing |
| GET | `/api/models/{id}` | one model plus its runs |
| GET | `/api/jobs` | job summaries (no logs) |
| GET | `/api/jobs/{id}` | one job including its log |
| POST | `/api/jobs/{id}/stop` | request a cooperative stop |
| WS | `/api/ws?token=...` | job progress snapshots every 500 ms when changed |

`scripts/smoke.py` exercises this surface against a running server:

```bash
GOLDNEURAL_API_TOKEN=... .venv/bin/python scripts/smoke.py \
    --url https://127.0.0.1:8443 --insecure --leakage
```

## Security posture

- Bearer token on every API route; IP allowlist optional
  (`server.allowed_ips`, IPs or CIDR).
- TLS by default; `--no-tls` exists only for loopback debugging.
- The token is never logged: a failed auth writes the source IP and nothing
  else. It is never stored in SQLite or in `config.yaml`.
- The server binds `0.0.0.0` because the plan asks for LAN access. Put it
  behind a firewall if the LAN is not trusted; there is no account system.
- The Research tab can start a local process. It is behind the same bearer
  token; the server builds an argument list (never a shell) and does not accept
  a path from the client. The campaign writes only under its own data
  directory.

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
run.py                       entrypoint: TLS, token, uvicorn
gold_neural/server/
  app.py                     routes, static mount, WS
  jobs.py                    single-worker job queue, progress, logs
  tasks.py                   job bodies (backfill, dataset, leakage, verify)
  security.py                token resolution, IP allowlist, auth dependency
  tls.py                     self-signed certificate bootstrap
  reports.py                 data/reports/*.json persistence
  research.py                detached campaign: start, stop, status, fold rows
gold_neural/model.py         window transformer + GPU probe
gold_neural/training.py      training loop, time budget, cooperative stop
gold_neural/backtest.py      trade simulation, fees, equity curve
gold_neural/search.py        candidate grid, cost pruning, validation scoring
gold_neural/registry.py      SQLite models + training_runs
web/
  index.html                 single page, six panels
  app.js                     API client, job stream, chart, renderers
  styles.css                 dark theme
  vendor/lightweight-charts… vendored chart library
scripts/smoke.py             post-deploy check of the running server
```

## Verified

- `pytest`: 110 tests, no network required.
- `scripts/smoke.py` against a live server: 12/12 checks, including a
  leakage run through the job API.
- Headless Chromium: page loads, chart renders 288 candles, all six tabs
  switch, a label preview and a training run both complete from the browser,
  the Stop button ends an unbounded run in ~2 s, job log streams, zero
  console errors, no horizontal overflow at 1440 px or 760 px.
- Guidance layer: checked against a fresh empty data root and against the
  populated one — the next-step banner, tab dots, intro card and empty states
  all follow the real state.
- Backtest tab: verdict, stats, equity chart and 50 trade rows render from a
  real run; the cost diagnostic fires when the target is under the round-trip
  cost.
- Auto-search tab: a live 2-candidate sweep added its first leaderboard row
  while the job was still running, ended at `done` with 2/2 candidates, and
  never showed the previous sweep's rows during the new one.
