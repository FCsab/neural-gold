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

## What each tab does

**Data** — candlestick chart of stored bars (24h / 7d / 30d), coverage
summary, partition table, gap count, a backfill form, and a "verify stored
bars" job that re-checks OHLC ordering, duplicates and missing bars.

**Dataset** — triple-barrier sliders (`k_up`, `k_down`, holding horizon, ATR
period, feature lag, same-bar ambiguity). *Preview labels* shows class counts
and per-class realised return without writing anything; *Build dataset*
writes `data/dataset.parquet` and records the exact parameters used.

**Leakage** — runs the mandatory suite with the current label parameters and
shows every check with its metrics. A failure turns the header chip red.

**Exchange** — caches `exchangeInfo` filters (tick size, step size, notional
limits). These are never hardcoded anywhere in the codebase.

**Console** — every job with its log. Jobs are queued and run one at a time,
deliberately: they share the Parquet store and the single SQLite writer.

**Training** — start a run with a minutes budget or with no limit at all
(press Stop when you have seen enough), watch the live epoch counter and log,
then read the test metrics: per-class precision/recall/F1, the confusion
matrix, balanced accuracy against the majority baseline, and the validation
loss curve. Both stopping paths write a checkpoint. Models land in the
registry below the run panel.

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
gold_neural/model.py         window transformer + GPU probe
gold_neural/training.py      training loop, time budget, cooperative stop
gold_neural/registry.py      SQLite models + training_runs
web/
  index.html                 single page, six panels
  app.js                     API client, job stream, chart, renderers
  styles.css                 dark theme
  vendor/lightweight-charts… vendored chart library
scripts/smoke.py             post-deploy check of the running server
```

## Verified

- `pytest`: 82 tests, no network required.
- `scripts/smoke.py` against a live server: 12/12 checks, including a
  leakage run through the job API.
- Headless Chromium: page loads, chart renders 288 candles, all six tabs
  switch, a label preview and a training run both complete from the browser,
  the Stop button ends an unbounded run in ~2 s, job log streams, zero
  console errors, no horizontal overflow at 1440 px or 760 px.
