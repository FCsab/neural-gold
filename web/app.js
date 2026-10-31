"use strict";

/* gold-neural web UI.
 *
 * No build step: plain ES2020 loaded by the browser. The API is the same one
 * the CLI uses, so anything reachable here is reachable headlessly.
 */

const TOKEN_KEY = "goldneural.token";

const state = {
  token: localStorage.getItem(TOKEN_KEY) || "",
  status: null,
  info: null,
  models: [],
  jobs: [],
  knownStatus: new Map(),
  selectedJob: null,
  chart: null,
  series: null,
  rangeBars: 288,
  ws: null,
  wsRetry: null,
  refreshTimer: null,
  refreshing: false,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ util */
const fmtInt = (value) =>
  value === null || value === undefined ? "–" : Number(value).toLocaleString("en-US");

const fmtNum = (value, digits = 4) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "–"
    : Number(value).toFixed(digits);

const fmtDate = (value) => {
  if (value === null || value === undefined) return "–";
  if (typeof value === "string") return value.slice(0, 16).replace("T", " ") + " UTC";
  return new Date(value).toISOString().replace("T", " ").slice(0, 16) + " UTC";
};

const fmtPct = (value, digits = 3) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "–"
    : (Number(value) * 100).toFixed(digits) + "%";

const escapeHtml = (text) =>
  String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const LABEL_NAMES = { "-1": "stop", "0": "timeout", "1": "target" };
const labelName = (key) => LABEL_NAMES[String(key)] || String(key);

const fmtDuration = (seconds) => {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "–";
  const total = Math.max(0, Math.round(Number(seconds)));
  const minutes = Math.floor(total / 60);
  return minutes ? `${minutes}m ${total % 60}s` : `${total}s`;
};

const numberOr = (value) => {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

function toast(message, kind = "info", ttl = 4500) {
  const node = document.createElement("div");
  node.className = "toast " + (kind === "info" ? "" : kind);
  node.textContent = message;
  $("toasts").appendChild(node);
  setTimeout(() => node.remove(), ttl);
}

/* ------------------------------------------------------------------- api */
async function api(path, options = {}) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;

  let body;
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.body);
  }

  const response = await fetch(path, {
    method: options.method || "GET",
    headers,
    body,
  });

  if (response.status === 401) {
    showLogin("token rejected");
    throw new Error("unauthorised");
  }
  if (response.status === 403) {
    showLogin("source IP not allowed by the allowlist");
    throw new Error("forbidden");
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const parsed = await response.json();
      if (parsed && parsed.detail) {
        message =
          typeof parsed.detail === "string"
            ? parsed.detail
            : JSON.stringify(parsed.detail);
      }
    } catch {
      /* body was not JSON */
    }
    throw new Error(message);
  }
  return response.json();
}

/* ----------------------------------------------------------------- login */
function showLogin(message) {
  $("login").classList.remove("hidden");
  $("login-error").textContent = message || "";
  if (state.ws) {
    try {
      state.ws.close();
    } catch {
      /* already closed */
    }
    state.ws = null;
  }
}

function hideLogin() {
  $("login").classList.add("hidden");
  $("login-error").textContent = "";
}

/* ---------------------------------------------------------------- panels */
function switchPanel(name) {
  document
    .querySelectorAll("#tabs button[data-panel]")
    .forEach((button) => button.classList.toggle("active", button.dataset.panel === name));
  document
    .querySelectorAll(".panel")
    .forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${name}`));

  if (name === "data") {
    ensureChart();
    loadBars();
  }
  if (name === "console" && state.selectedJob) loadLog(state.selectedJob);
  if (name === "training") {
    renderTrainingPanel();
    const job = currentTrainingJob();
    if (job) updateTrainLog(job.id);
  }
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  bindEvents();

  // Log polling for the console pane. Registered before the token check so it
  // also runs for a session that starts at the login screen.
  setInterval(() => {
    if (!state.selectedJob) return;
    if (!$("panel-console").classList.contains("active")) return;
    const job = state.jobs.find((entry) => entry.id === state.selectedJob);
    if (job && (job.status === "running" || job.status === "queued")) loadLog(job.id);
  }, 1200);

  setInterval(() => {
    if (!$("panel-training").classList.contains("active")) return;
    const job = currentTrainingJob();
    if (job && (job.status === "running" || job.status === "queued")) updateTrainLog(job.id);
  }, 1500);

  if (!state.token) {
    showLogin("");
    return;
  }
  try {
    await refreshAll();
    hideLogin();
    connectWs();
    ensureChart();
    loadBars();
  } catch (error) {
    if (error.message === "unauthorised" || error.message === "forbidden") return;
    showLogin("");
    toast(error.message, "error");
  }
}

async function refreshAll() {
  if (state.refreshing) return;
  state.refreshing = true;
  try {
    const [status, info, jobs, models] = await Promise.all([
      api("/api/status"),
      api("/api/data/info"),
      api("/api/jobs"),
      api("/api/models"),
    ]);
    state.status = status;
    state.info = info;
    state.models = models.models || [];
    applyJobs(jobs.jobs || []);
    renderStatus();
    renderInfo();
    renderDataset();
    renderLeakage();
    renderExchange();
    renderTraining();
    renderTrainingPanel();
  } finally {
    state.refreshing = false;
  }
}

function scheduleRefresh() {
  clearTimeout(state.refreshTimer);
  state.refreshTimer = setTimeout(() => {
    refreshAll().catch(() => {});
  }, 400);
}

/* ------------------------------------------------------------------ jobs */
function applyJobs(jobs) {
  const finished = [];
  for (const job of jobs) {
    const before = state.knownStatus.get(job.id);
    if (before && before !== job.status && (job.status === "ok" || job.status === "failed")) {
      finished.push(job);
    }
    state.knownStatus.set(job.id, job.status);
  }

  state.jobs = jobs;
  renderJobs();
  renderJobStrip();
  renderTrainingPanel();

  for (const job of finished) {
    if (job.status === "ok") toast(`${job.kind} finished`, "ok");
    else toast(`${job.kind} failed: ${job.error || "unknown error"}`, "error", 9000);
    if (job.kind === "label-preview" && job.status === "ok") {
      renderPreviewResult(job.result);
    }
  }
  if (finished.length) scheduleRefresh();

  if (state.selectedJob && $("panel-console").classList.contains("active")) {
    loadLog(state.selectedJob);
  }
}

function renderJobStrip() {
  const strip = $("jobstrip");
  const active = state.jobs.find(
    (job) => job.status === "running" || job.status === "queued"
  );
  const badge = $("job-badge");
  const runningCount = state.jobs.filter((job) => job.status === "running").length;

  badge.textContent = String(runningCount);
  badge.classList.toggle("hidden", runningCount === 0);

  if (!active) {
    strip.className = "jobstrip idle";
    strip.querySelector(".jobstrip-kind").textContent = "idle";
    const last = state.jobs[0];
    strip.querySelector(".jobstrip-msg").textContent = last
      ? `last: ${last.kind} ${last.status}`
      : "no job running";
    $("jobstrip-bar").style.width = "0%";
    return;
  }

  strip.className = "jobstrip active";
  if (active.progress === null || active.progress === undefined) {
    strip.classList.add("indeterminate");
    $("jobstrip-bar").style.width = "30%";
  } else {
    $("jobstrip-bar").style.width = Math.round(active.progress * 100) + "%";
  }
  strip.querySelector(".jobstrip-kind").textContent = active.kind;
  strip.querySelector(".jobstrip-msg").textContent =
    active.message || active.status;
}

function renderJobs() {
  const list = $("job-list");
  if (!state.jobs.length) {
    list.innerHTML = '<p class="muted small">none yet</p>';
    return;
  }
  list.innerHTML = state.jobs
    .map((job) => {
      const selected = job.id === state.selectedJob ? " selected" : "";
      const when = new Date(job.created_at * 1000).toLocaleTimeString("en-GB");
      return `<div class="job-row${selected}" data-job="${job.id}">
        <span class="dot ${job.status}"></span>
        <span class="kind">${escapeHtml(job.kind)}</span>
        <span class="when">${when}</span>
      </div>`;
    })
    .join("");

  list.querySelectorAll("[data-job]").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedJob = row.dataset.job;
      renderJobs();
      loadLog(state.selectedJob);
    });
  });
}

async function loadLog(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    $("log-title").textContent = job.kind;
    $("log-meta").textContent = `${job.status} · ${job.log_lines} lines`;
    const log = $("log");
    const text = (job.log || []).join("\n") || "(no output)";
    if (log.textContent !== text) {
      log.textContent = text;
      log.scrollTop = log.scrollHeight;
    }
  } catch (error) {
    if (error.message !== "unauthorised") $("log").textContent = error.message;
  }
}

async function startJob(path, body, label) {
  try {
    const { job_id: jobId } = await api(path, { method: "POST", body });
    state.selectedJob = jobId;
    toast(`${label} queued`, "info");
    await refreshAll();
    loadLog(jobId);
    return jobId;
  } catch (error) {
    if (error.message !== "unauthorised" && error.message !== "forbidden") {
      toast(error.message, "error", 8000);
    }
    return null;
  }
}

/* ------------------------------------------------------------- websocket */
function connectWs() {
  if (!state.token) return;
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${location.host}/api/ws?token=${encodeURIComponent(state.token)}`;

  let socket;
  try {
    socket = new WebSocket(url);
  } catch {
    return;
  }
  state.ws = socket;

  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "jobs") applyJobs(message.jobs || []);
    } catch {
      /* ignore malformed frame */
    }
  };

  socket.onclose = (event) => {
    state.ws = null;
    if (event.code === 4401) {
      showLogin("token rejected");
      return;
    }
    if (!state.token) return;
    clearTimeout(state.wsRetry);
    state.wsRetry = setTimeout(connectWs, 2500);
  };
}

/* ----------------------------------------------------------------- chart */
function ensureChart() {
  if (state.chart || typeof LightweightCharts === "undefined") return;
  const container = $("chart");
  state.chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight,
    layout: {
      background: { color: "#11161f" },
      textColor: "#7d8ca1",
      fontSize: 11,
    },
    grid: {
      vertLines: { color: "#161d29" },
      horzLines: { color: "#161d29" },
    },
    rightPriceScale: { borderColor: "#1d2634" },
    timeScale: { borderColor: "#1d2634", timeVisible: true },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  state.series = state.chart.addCandlestickSeries({
    upColor: "#46b97a",
    downColor: "#e05c5c",
    borderVisible: false,
    wickUpColor: "#46b97a",
    wickDownColor: "#e05c5c",
    priceFormat: { type: "price", precision: 2, minMove: 0.01 },
  });
  new ResizeObserver(() => {
    if (!state.chart) return;
    state.chart.applyOptions({
      width: container.clientWidth,
      height: container.clientHeight,
    });
  }).observe(container);
}

async function loadBars() {
  ensureChart();
  const note = $("chart-note");
  try {
    const data = await api(`/api/data/bars?limit=${state.rangeBars}`);
    if (!state.series) {
      note.textContent = "chart library did not load";
      return;
    }
    const bars = data.bars || [];
    state.series.setData(bars);
    state.chart.timeScale().fitContent();
    if (!bars.length) {
      note.textContent = "no bars stored yet — run a backfill";
      return;
    }
    note.textContent = `${fmtInt(bars.length)} bars · last ${fmtDate(
      bars[bars.length - 1].time * 1000
    )}`;
  } catch (error) {
    if (error.message !== "unauthorised") note.textContent = error.message;
  }
}

/* -------------------------------------------------------------- renderers */
function renderStatus() {
  const status = state.status;
  if (!status) return;
  $("version").textContent = `v${status.version}`;

  const chips = [
    `<span class="chip">${status.symbol} <strong>${status.interval}</strong></span>`,
    `<span class="chip">bars <strong>${fmtInt(status.bars)}</strong></span>`,
    `<span class="chip ${status.tls ? "ok" : "bad"}">${status.tls ? "https" : "http"}</span>`,
  ];
  const leakage = state.info && state.info.leakage;
  if (leakage) {
    chips.push(
      `<span class="chip ${leakage.passed ? "ok" : "bad"}">leakage ${
        leakage.passed ? "pass" : "fail"
      }</span>`
    );
  }
  $("status-chips").innerHTML = chips.join("");
  $("training-reason").textContent =
    (status.training && status.training.reason) || "";
}

function renderInfo() {
  const info = state.info;
  if (!info) return;
  const coverage = info.coverage || {};
  const verify = info.verify;

  const rows = [
    ["bars", fmtInt(coverage.rows)],
    ["from", fmtDate(coverage.start_ms)],
    ["to", fmtDate(coverage.end_ms)],
    ["partitions", fmtInt((info.partitions || []).length)],
    ["gaps", verify ? fmtInt(verify.gap_count) : "–"],
  ];
  $("coverage-stats").innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");

  $("verify-note").textContent = verify
    ? `${verify.ok ? "clean" : "problems found"} · ${fmtDate(verify.generated_at)}`
    : "not verified yet — press “verify stored bars”";

  const partitions = (info.partitions || []).slice().reverse();
  $("partition-count").textContent = `(${partitions.length})`;
  $("partitions").innerHTML =
    '<thead><tr><th>month</th><th class="num">bars</th><th>from</th><th>to</th></tr></thead><tbody>' +
    partitions
      .map(
        (part) =>
          `<tr><td>${part.year}-${String(part.month).padStart(2, "0")}</td>` +
          `<td class="num">${fmtInt(part.rows)}</td>` +
          `<td>${fmtDate(part.start_ms)}</td><td>${fmtDate(part.end_ms)}</td></tr>`
      )
      .join("") +
    "</tbody>";
}

function classTable(byClass) {
  const names = { "-1": "stop", "0": "timeout", "1": "target" };
  const order = ["1", "0", "-1"].filter((key) => byClass[key]);
  if (!order.length) return "";
  const body = order
    .map((key) => {
      const row = byClass[key];
      return `<tr><td>${names[key]}</td><td class="num">${fmtInt(row.count)}</td>` +
        `<td class="num">${fmtPct(row.mean_return)}</td>` +
        `<td class="num">${fmtPct(row.std_return)}</td></tr>`;
    })
    .join("");
  return (
    '<table class="table"><thead><tr><th>class</th><th class="num">rows</th>' +
    '<th class="num">mean return</th><th class="num">std</th></tr></thead>' +
    `<tbody>${body}</tbody></table>`
  );
}

function renderDataset() {
  const report = state.info && state.info.dataset;
  const stats = $("dataset-stats");
  const table = $("class-table");

  if (!report) {
    stats.innerHTML = '<dt>—</dt><dd class="muted">nothing built yet</dd>';
    table.innerHTML = "";
    return;
  }

  const params = report.params || {};
  const split = report.split || {};
  const rows = [
    ["rows", fmtInt(report.rows)],
    ["features", fmtInt(report.features)],
    ["dropped (warm-up)", fmtInt(report.dropped)],
    [
      "train / val / test",
      `${fmtInt(split.train)} / ${fmtInt(split.validation)} / ${fmtInt(split.test)}`,
    ],
    ["k_up / k_down", `${params.k_up} / ${params.k_down}`],
    ["holding", `${params.max_holding_bars} bars`],
    ["feature lag", String(params.feature_lag)],
    ["built", fmtDate(report.generated_at)],
  ];
  stats.innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");
  table.innerHTML = classTable(report.by_class || {});
}

function renderPreviewResult(result) {
  const box = $("preview-result");
  if (!box) return;
  if (!result) {
    box.innerHTML = "";
    return;
  }
  const distribution = result.distribution || {};
  box.innerHTML =
    `<p class="muted small" style="margin-top:14px">` +
    `preview only · ${fmtInt(result.labelled)} labelled bars · ` +
    `horizon ${result.horizon_bars} bars · nothing written to disk</p>` +
    `<dl class="stats">` +
    `<dt>target / timeout / stop</dt>` +
    `<dd>${fmtInt(distribution.long)} / ${fmtInt(distribution.flat)} / ${fmtInt(distribution.stop)}</dd>` +
    `</dl>` +
    classTable(result.by_class || {});
}

function renderLeakage() {
  const report = state.info && state.info.leakage;
  const badge = $("leakage-badge");
  const box = $("leakage-checks");

  if (!report) {
    badge.textContent = "—";
    badge.className = "badge";
    box.innerHTML = '<p class="muted small">not run yet</p>';
    return;
  }

  badge.textContent = report.passed ? "PASS" : "FAIL";
  badge.className = "badge " + (report.passed ? "ok" : "bad");

  const checks = report.checks || [];
  box.innerHTML = checks
    .map((check) => {
      const metrics = Object.entries(check.metrics || {})
        .map(
          ([key, value]) =>
            `<span>${escapeHtml(key)}=${
              typeof value === "number" ? fmtNum(value, 6) : escapeHtml(value)
            }</span>`
        )
        .join("");
      return `<div class="check ${check.passed ? "pass" : "fail"}">
        <div class="check-head">
          <span class="check-name">${escapeHtml(check.name)}</span>
          <span class="badge ${check.passed ? "ok" : "bad"}">${
            check.passed ? "pass" : "fail"
          }</span>
        </div>
        <div class="check-detail">${escapeHtml(check.detail)}</div>
        ${metrics ? `<div class="check-metrics">${metrics}</div>` : ""}
      </div>`;
    })
    .join("");
}

function renderExchange() {
  const report = state.info && state.info.exchange;
  const table = $("exchange-table");

  if (!report) {
    table.innerHTML =
      '<tbody><tr><td class="muted">not fetched yet</td></tr></tbody>';
    return;
  }

  const filters = report.filters || {};
  const rows = Object.entries(filters).map(([name, payload]) => {
    const detail = Object.entries(payload)
      .filter(([key]) => key !== "filterType")
      .map(([key, value]) => `${key}=${value}`)
      .join(" · ");
    return `<tr><td>${escapeHtml(name)}</td><td>${escapeHtml(detail)}</td></tr>`;
  });
  table.innerHTML =
    '<thead><tr><th>filter</th><th>values</th></tr></thead>' +
    `<tbody>${rows.join("")}</tbody>`;
}

/* -------------------------------------------------------------- training */
function trainingAvailability() {
  return (state.status && state.status.training) || {
    available: false,
    reason: "the server did not report training availability",
  };
}

function renderTraining() {
  const availability = trainingAvailability();
  const badge = $("training-availability");
  badge.textContent = availability.available ? "ready" : "unavailable";
  badge.className = "badge " + (availability.available ? "ok" : "bad");

  const parts = [];
  if (availability.available) {
    parts.push(`torch ${availability.torch}`);
    parts.push(
      availability.gpu_reported == null
        ? "device probed per run"
        : availability.gpu_reported
        ? "GPU reported"
        : "CPU only"
    );
  } else {
    parts.push(availability.reason || "PyTorch is not installed");
  }
  $("training-reason").textContent = parts.join(" · ");
  renderModels();
}

function renderModels() {
  const models = state.models || [];
  $("model-count").textContent = `(${models.length})`;
  const table = $("model-table");
  if (!models.length) {
    table.innerHTML =
      '<tbody><tr><td class="muted">no models yet — run a training job</td></tr></tbody>';
    return;
  }
  const rows = models
    .map((model) => {
      const metrics = model.metrics || {};
      return (
        `<tr><td>${escapeHtml(model.name)}</td><td>${escapeHtml(model.status)}</td>` +
        `<td class="num">${fmtInt(model.window)}</td>` +
        `<td class="num">${fmtNum(metrics.balanced_accuracy, 3)}</td>` +
        `<td class="num">${fmtNum(metrics.test_rows, 0)}</td>` +
        `<td>${fmtDate(model.created_at)}</td></tr>`
      );
    })
    .join("");
  table.innerHTML =
    '<thead><tr><th>model</th><th>status</th><th class="num">window</th>' +
    '<th class="num">bal. acc</th><th class="num">test rows</th><th>created</th></tr></thead>' +
    `<tbody>${rows}</tbody>`;
}

function currentTrainingJob() {
  const jobs = state.jobs.filter((job) => job.kind === "training");
  if (!jobs.length) return null;
  return (
    jobs.find((job) => job.status === "running" || job.status === "queued") ||
    jobs[0]
  );
}

function renderTrainingPanel() {
  const job = currentTrainingJob();
  const badge = $("train-state");
  const live = $("train-live");
  const startButton = $("start-training");
  const stopButton = $("stop-training");

  if (!job) {
    badge.textContent = "idle";
    badge.className = "badge";
    live.innerHTML = "<dt>state</dt><dd>no run yet</dd>";
    startButton.disabled = false;
    stopButton.disabled = true;
    return;
  }

  const running = job.status === "running" || job.status === "queued";
  badge.textContent = job.status;
  badge.className =
    "badge " +
    (job.status === "ok"
      ? "ok"
      : job.status === "failed"
      ? "bad"
      : job.status === "stopped"
      ? "amber"
      : "");
  startButton.disabled = running;
  stopButton.disabled = !running;

  const elapsed = job.started_at
    ? (job.finished_at || Date.now() / 1000) - job.started_at
    : 0;
  const rows = [
    ["state", job.status],
    ["progress", job.message || "–"],
    ["elapsed", fmtDuration(elapsed)],
  ];
  live.innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");

  if (job.result && !running) renderTrainResult(job.result);
}

function sparkline(history) {
  const values = (history || [])
    .map((entry) => Number(entry.val_loss))
    .filter((value) => Number.isFinite(value));
  if (values.length < 2) return "";

  const width = 240;
  const height = 56;
  const pad = 3;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = (width - pad * 2) / (values.length - 1);

  const points = values.map((value, index) => {
    const x = pad + index * step;
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const last = points[points.length - 1].split(",");

  return `<svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <polyline points="${points.join(" ")}" fill="none" stroke="#d9b25f" stroke-width="1.6" vector-effect="non-scaling-stroke"/>
    <circle cx="${last[0]}" cy="${last[1]}" r="2.4" fill="#d9b25f"/>
  </svg>`;
}

function renderTrainResult(result) {
  const metrics = result.metrics || {};
  $("train-meta").textContent =
    `${result.stop_reason} · ${result.epochs_done} epochs · ` +
    `${fmtDuration(result.duration_s)} · ${result.device}`;

  const summary = [
    ["balanced accuracy", fmtNum(metrics.balanced_accuracy, 3)],
    ["baseline (majority)", fmtNum(metrics.baseline_accuracy, 3)],
    ["accuracy", fmtNum(metrics.accuracy, 3)],
    ["signal edge", metrics.signal_edge == null ? "–" : fmtPct(metrics.signal_edge)],
    [
      "long coverage",
      metrics.long_coverage == null ? "–" : fmtPct(metrics.long_coverage, 1),
    ],
    ["test rows", fmtInt(metrics.test_rows)],
    ["parameters", fmtInt(result.parameters)],
    ["device", `${result.device} · ${result.device_note}`],
  ];

  const perClass = metrics.per_class || {};
  const classRows = ["1", "0", "-1"]
    .filter((key) => perClass[key])
    .map((key) => {
      const row = perClass[key];
      return (
        `<tr><td>${labelName(key)}</td>` +
        `<td class="num">${fmtNum(row.precision, 3)}</td>` +
        `<td class="num">${fmtNum(row.recall, 3)}</td>` +
        `<td class="num">${fmtNum(row.f1, 3)}</td>` +
        `<td class="num">${fmtInt(row.support)}</td></tr>`
      );
    })
    .join("");

  const matrix = metrics.confusion_matrix || [];
  const matrixHeader =
    '<tr><th>actual \\ predicted</th>' +
    ["-1", "0", "1"].map((key) => `<th class="num">${labelName(key)}</th>`).join("") +
    "</tr>";
  const matrixRows = matrix
    .map((row, index) => {
      const key = ["-1", "0", "1"][index];
      return (
        `<tr><td>${labelName(key)}</td>` +
        row.map((count) => `<td class="num">${fmtInt(count)}</td>`).join("") +
        "</tr>"
      );
    })
    .join("");

  $("train-metrics").innerHTML = `
    <div class="result-grid">
      <div>
        <h3 class="subhead">Summary</h3>
        <dl class="stats">${summary
          .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(String(value))}</dd>`)
          .join("")}</dl>
      </div>
      <div>
        <h3 class="subhead">Per class</h3>
        ${
          classRows
            ? `<table class="table"><thead><tr><th>class</th><th class="num">precision</th>` +
              `<th class="num">recall</th><th class="num">f1</th><th class="num">support</th>` +
              `</tr></thead><tbody>${classRows}</tbody></table>`
            : '<p class="muted small">no metrics</p>'
        }
      </div>
    </div>
    <div class="result-grid">
      <div>
        <h3 class="subhead">Confusion matrix</h3>
        ${
          matrixRows
            ? `<table class="table"><thead>${matrixHeader}</thead><tbody>${matrixRows}</tbody></table>`
            : '<p class="muted small">no metrics</p>'
        }
      </div>
      <div>
        <h3 class="subhead">Validation loss</h3>
        ${sparkline(result.history) || '<p class="muted small">no epochs recorded</p>'}
        <p class="muted small">best epoch ${fmtInt(metrics.best_epoch)} · best val loss ${fmtNum(metrics.best_val_loss, 4)}</p>
      </div>
    </div>`;
}

async function updateTrainLog(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    const node = $("train-log");
    const text = (job.log || []).join("\n") || "(no output)";
    if (node.textContent !== text) {
      node.textContent = text;
      node.scrollTop = node.scrollHeight;
    }
  } catch {
    /* the panel simply keeps the previous log */
  }
}

function trainParams() {
  const minutes = numberOr($("train-minutes").value);
  return {
    minutes: minutes === null ? null : minutes,
    epochs: Number($("train-epochs").value) || 50,
    window: Number($("train-window").value) || 64,
    batch_size: Number($("train-batch").value) || 256,
    lr: Number($("train-lr").value) || 0.0003,
    patience: Number($("train-patience").value) || 5,
    device: $("train-device").value,
  };
}

/* ----------------------------------------------------------------- events */
function datasetParams() {
  const params = {
    k_up: numberOr($("k-up").value),
    k_down: numberOr($("k-down").value),
    max_holding_bars: numberOr($("horizon").value),
    atr_period: numberOr($("atr-period").value),
    feature_lag: numberOr($("feature-lag").value),
    ambiguity: $("ambiguity").value,
  };
  for (const key of Object.keys(params)) {
    if (params[key] === null) delete params[key];
  }
  return params;
}

function bindEvents() {
  $("tabs").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-panel]");
    if (button) switchPanel(button.dataset.panel);
  });

  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const token = $("login-token").value.trim();
    if (!token) {
      $("login-error").textContent = "token is empty";
      return;
    }
    state.token = token;
    localStorage.setItem(TOKEN_KEY, token);
    try {
      await refreshAll();
      hideLogin();
      connectWs();
      ensureChart();
      loadBars();
      $("login-token").value = "";
    } catch (error) {
      localStorage.removeItem(TOKEN_KEY);
      state.token = "";
      if (error.message !== "unauthorised" && error.message !== "forbidden") {
        $("login-error").textContent = error.message;
      }
    }
  });

  $("logout").addEventListener("click", () => {
    localStorage.removeItem(TOKEN_KEY);
    state.token = "";
    state.status = null;
    state.info = null;
    showLogin("");
  });

  $("range-seg").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-bars]");
    if (!button) return;
    $("range-seg")
      .querySelectorAll("button")
      .forEach((node) => node.classList.toggle("active", node === button));
    state.rangeBars = Number(button.dataset.bars);
    loadBars();
  });

  $("refresh-bars").addEventListener("click", loadBars);
  $("verify").addEventListener("click", () =>
    startJob("/api/data/verify", {}, "verify")
  );
  $("refresh-exchange").addEventListener("click", () =>
    startJob("/api/data/exchange-info", {}, "exchange-info")
  );

  $("backfill-form").addEventListener("submit", (event) => {
    event.preventDefault();
    startJob(
      "/api/data/backfill",
      { years: numberOr($("backfill-years").value) || 2 },
      "backfill"
    );
  });

  $("preview-labels").addEventListener("click", () =>
    startJob("/api/dataset/preview", datasetParams(), "label-preview")
  );
  $("build-dataset").addEventListener("click", () =>
    startJob("/api/dataset/build", datasetParams(), "dataset-build")
  );

  $("leakage-form").addEventListener("submit", (event) => {
    event.preventDefault();
    startJob(
      "/api/leakage/run",
      { samples: numberOr($("leakage-samples").value) || 10, ...datasetParams() },
      "leakage"
    );
  });

  $("start-training").addEventListener("click", async () => {
    const params = trainParams();
    const jobId = await startJob("/api/train/start", params, "training");
    if (jobId) {
      renderTrainingPanel();
      updateTrainLog(jobId);
    }
  });

  $("stop-training").addEventListener("click", async () => {
    const job = currentTrainingJob();
    if (!job) return;
    try {
      await api(`/api/jobs/${job.id}/stop`, { method: "POST", body: {} });
      toast("stop requested — checkpoint will be written", "info");
    } catch (error) {
      if (error.message !== "unauthorised") toast(error.message, "error");
    }
  });
}

document.addEventListener("DOMContentLoaded", boot);
