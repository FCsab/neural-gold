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
  btChart: null,
  btSeries: null,
  btHoldSeries: null,
  search: null,
  searchChart: null,
  searchSeries: null,
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
  if (name === "backtest") {
    ensureBtChart();
    renderBacktestChart();
  }
  if (name === "search") {
    ensureSearchChart();
    renderSearch();
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

  // The sweep writes its report after every candidate; poll it so the
  // leaderboard fills in while the job is still running.
  setInterval(() => {
    const running = state.jobs.some(
      (job) => job.kind === "search" && (job.status === "running" || job.status === "queued")
    );
    if (!running) return;
    api("/api/search/last")
      .then((report) => {
        state.search = report;
        renderSearch();
      })
      .catch(() => {});
  }, 3000);

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
    if (info.search) state.search = info.search;
    applyJobs(jobs.jobs || []);
    renderStatus();
    renderInfo();
    renderDataset();
    renderLeakage();
    renderExchange();
    renderTraining();
    renderTrainingPanel();
    renderBacktest();
    renderSearch();
    renderPipeline();
    renderIntro();
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

const JOB_LABELS = {
  backfill: "Downloading bars",
  "dataset-build": "Building the dataset",
  "label-preview": "Counting labels",
  leakage: "Running the look-ahead check",
  verify: "Checking stored data",
  "exchange-info": "Fetching exchange rules",
  training: "Training",
  backtest: "Backtesting",
  search: "Searching settings",
};

const jobLabel = (kind) => JOB_LABELS[kind] || kind;

const STOP_REASONS = {
  completed: "finished all epochs",
  time_budget: "stopped at the time limit",
  stopped: "stopped by you",
  early_stopping: "stopped early — validation stopped improving",
};
const stopReasonLabel = (reason) => STOP_REASONS[reason] || reason;

/** One sentence explaining why a run ended, in terms of the settings. */
function stopNote(result) {
  const metrics = result.metrics || {};
  const patience = (result.config && result.config.patience) || null;
  const best = fmtInt(metrics.best_epoch);

  if (result.stop_reason === "early_stopping") {
    return (
      `It stopped at epoch ${fmtInt(result.epochs_done)} because validation loss had not improved for ` +
      `${patience ? fmtInt(patience) + " epochs" : "several epochs"}. ` +
      `Best weights: epoch ${best}. ` +
      `That is the patience setting doing its job, not an error.`
    );
  }
  if (result.stop_reason === "time_budget") {
    return (
      `It stopped when the time limit ran out. Best weights: epoch ${best}. ` +
      `Raise the limit, or clear it to run until you press Stop.`
    );
  }
  if (result.stop_reason === "stopped") {
    return `You pressed Stop. The best weights so far (epoch ${best}) were kept.`;
  }
  return "";
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
      ? `Nothing is running. Last action: ${jobLabel(last.kind).toLowerCase()} — ${last.status}.`
      : "Nothing is running. Anything you start shows up here.";
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
  strip.querySelector(".jobstrip-kind").textContent = jobLabel(active.kind);
  strip.querySelector(".jobstrip-msg").textContent =
    active.status === "queued" ? `queued — ${jobLabel(active.kind).toLowerCase()}` : active.message || "working…";
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

/* -------------------------------------------------------------- guidance */
function computeStage() {
  const info = state.info || {};
  const coverage = info.coverage || {};
  const rows = coverage.rows || 0;
  const leakage = info.leakage;
  const models = state.models || [];
  return {
    rows,
    hasBars: rows > 0,
    hasDataset: Boolean(info.dataset),
    datasetRows: info.dataset ? info.dataset.rows : 0,
    leakageState: leakage ? (leakage.passed ? "pass" : "fail") : "none",
    models,
    hasModel: models.length > 0,
    hasBacktest: Boolean(info.backtest),
    search: info.search || null,
    hasSearch: Boolean(info.search),
  };
}

function nextStepFor(stage) {
  if (!stage.hasBars) {
    return {
      panel: "data",
      text: "Download price history from Binance. Everything else is built from these bars.",
      action: "Take me there",
    };
  }
  if (!stage.hasDataset) {
    return {
      panel: "dataset",
      text: `You have ${fmtInt(stage.rows)} bars. Turn them into labelled examples the model can learn from.`,
      action: "Take me there",
    };
  }
  if (stage.leakageState === "none") {
    return {
      panel: "leakage",
      text: "Dataset is built. Now prove the features cannot see the future — this gate exists for a reason.",
      action: "Take me there",
    };
  }
  if (stage.leakageState === "fail") {
    return {
      panel: "leakage",
      text: "The look-ahead check failed. Do not train until it passes; a leaked model will look brilliant and lose money.",
      action: "Review the checks",
      tone: "error",
    };
  }
  if (!stage.hasModel) {
    return {
      panel: "training",
      text: "Data is ready and clean. Train a first model — 10 minutes is a reasonable first budget.",
      action: "Take me there",
    };
  }
  if (!stage.hasBacktest) {
    return {
      panel: "backtest",
      text: "You have a model. Now see what its trades would actually have made after fees, next to simply holding PAXG.",
      action: "Take me there",
    };
  }
  if (!stage.hasSearch) {
    return {
      panel: "search",
      text: "Let the machine try dozens of label settings while you are away — it skips the ones that cannot pay for their fees.",
      action: "Take me there",
    };
  }
  if (stage.search && stage.search.status === "running") {
    return {
      panel: "search",
      text: "A sweep is running. Results appear on the leaderboard as each candidate finishes.",
      action: "Watch it",
    };
  }
  return {
    panel: "search",
    text: "Everything is set up. Review the best candidate, or start another sweep with a different budget.",
    action: "Open auto-search",
  };
}

function renderPipeline() {
  const stage = computeStage();

  const dots = {
    data: stage.hasBars ? "done" : "",
    dataset: stage.hasDataset ? "done" : "",
    leakage:
      stage.leakageState === "pass" ? "done" : stage.leakageState === "fail" ? "error" : "",
    training: stage.hasModel ? "done" : "",
    backtest: stage.hasBacktest ? "done" : "",
    search: stage.search && stage.search.status === "done" ? "done" : "",
  };
  Object.entries(dots).forEach(([key, cls]) => {
    const dot = document.querySelector(`[data-dot="${key}"]`);
    if (dot) dot.className = "tab-dot" + (cls ? " " + cls : "");
  });

  const next = nextStepFor(stage);
  state.nextPanel = next.panel;
  const banner = $("next-step");
  banner.hidden = false;
  banner.className = "nextstep" + (next.tone ? " " + next.tone : "");
  $("next-step-text").textContent = next.text;
}

function renderIntro() {
  const hidden = localStorage.getItem("goldneural.introHidden") === "1";
  $("intro").hidden = hidden;
}

/** Balanced accuracy of a constant predictor: 1 / number of classes present. */
function balancedBaseline(metrics) {
  if (metrics.baseline_balanced_accuracy != null) {
    return Number(metrics.baseline_balanced_accuracy);
  }
  const perClass = metrics.per_class;
  if (perClass) {
    const present = Object.values(perClass).filter((row) => (row.support || 0) > 0).length;
    if (present) return 1 / present;
  }
  // Older payloads only carry the plain accuracy; better than nothing.
  return Number(metrics.baseline_accuracy);
}

function verdictFor(metrics) {
  if (!metrics || metrics.balanced_accuracy == null) return null;
  const balanced = Number(metrics.balanced_accuracy);
  // Compare like with like: a constant predictor scores 1/3 on *balanced*
  // accuracy, not the 48% plain accuracy of always answering the big class.
  const baseline = balancedBaseline(metrics);
  if (!Number.isFinite(balanced) || !Number.isFinite(baseline)) return null;
  const delta = (balanced - baseline) * 100;
  const points = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}`;

  if (delta >= 5) {
    return {
      tone: "ok",
      html: `<strong>Better than a constant guess.</strong> Balanced accuracy ${fmtPct(
        balanced,
        1
      )} against ${fmtPct(
        baseline,
        1
      )} for always answering one outcome (${points} points). Still a hint, not proof — costs decide in the backtest.`,
    };
  }
  if (delta >= 2) {
    return {
      tone: "",
      html: `<strong>Marginally better than a constant guess.</strong> Balanced accuracy ${fmtPct(
        balanced,
        1
      )} against ${fmtPct(
        baseline,
        1
      )} (${points} points). A gap this small rarely survives fees — check the backtest before believing it.`,
    };
  }
  if (delta <= -2) {
    return {
      tone: "bad",
      html: `<strong>Worse than a constant guess.</strong> Balanced accuracy ${fmtPct(
        balanced,
        1
      )} against ${fmtPct(
        baseline,
        1
      )} (${points} points). That points at the setup or the labels, not just at training time.`,
    };
  }
  return {
    tone: "",
    html: `<strong>At chance.</strong> Balanced accuracy ${fmtPct(
      balanced,
      1
    )} against ${fmtPct(baseline, 1)} for a constant guess (${points} points). No usable pattern found yet.`,
  };
}

function renderStatus() {
  const status = state.status;
  if (!status) return;
  $("version").textContent = `v${status.version}`;

  const info = state.info || {};
  const coverage = info.coverage || {};
  const rows = coverage.rows || 0;
  const years =
    coverage.start_ms && coverage.end_ms
      ? (coverage.end_ms - coverage.start_ms) / (365.25 * 86400000)
      : null;

  const chips = [`<span class="chip">${status.symbol} · ${status.interval}</span>`];
  chips.push(
    `<span class="chip">${
      rows ? `${fmtInt(rows)} bars${years ? ` · ${years.toFixed(1)} yrs` : ""}` : "no bars yet"
    }</span>`
  );

  const leakage = info.leakage;
  chips.push(
    leakage
      ? `<span class="chip ${leakage.passed ? "ok" : "bad"}">look-ahead ${
          leakage.passed ? "passed" : "FAILED"
        }</span>`
      : '<span class="chip">look-ahead not run</span>'
  );

  const models = state.models || [];
  chips.push(
    `<span class="chip">${
      models.length ? `${models.length} model${models.length === 1 ? "" : "s"}` : "no models yet"
    }</span>`
  );

  $("status-chips").innerHTML = chips.join("");
}

function renderInfo() {
  const info = state.info;
  if (!info) return;
  const coverage = info.coverage || {};
  const verify = info.verify;

  const bars = coverage.rows || 0;
  const rows = bars
    ? [
        ["bars stored", fmtInt(bars)],
        ["from", fmtDate(coverage.start_ms)],
        ["to", fmtDate(coverage.end_ms)],
        ["monthly files", fmtInt((info.partitions || []).length)],
        ["missing stretches", verify ? fmtInt(verify.gap_count) : "not checked"],
      ]
    : [["status", "No price history yet — press “Download bars”."]];
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

  const note = $("class-note");
  if (note) note.hidden = !report;

  if (!report) {
    stats.innerHTML =
      '<dt>—</dt><dd class="muted">Nothing built yet. Press “Build the dataset”.</dd>';
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
    startButton.disabled = false;
    stopButton.disabled = true;

    // Job history is in memory only, but models are persisted. After a restart
    // the last recorded model with actual metrics is the useful thing to show.
    // (A run stopped before its first epoch has no metrics at all.)
    const latest = (state.models || []).find(
      (model) => model.metrics && Object.keys(model.metrics).length
    );
    if (latest) {
      live.innerHTML =
        `<dt>state</dt><dd>idle</dd>` +
        `<dt>last model</dt><dd>${escapeHtml(latest.name)}</dd>` +
        `<dt>recorded</dt><dd>${fmtDate(latest.created_at)}</dd>`;
      $("train-log").textContent = "No run in this session. Metrics below are from the registry.";
      $("train-meta").textContent = `last recorded model · ${fmtDate(latest.created_at)}`;
      renderTrainResult(
        {
          metrics: latest.metrics,
          history: [],
          parameters: null,
          device: null,
          device_note: "",
          stop_reason: "recorded",
          duration_s: null,
          epochs_done: null,
        },
        { stored: true }
      );
    } else {
      live.innerHTML =
        "<dt>state</dt><dd>No run yet. Pick a time limit above and press “Start training”.</dd>";
      $("train-log").textContent = "Start a run to see its progress here.";
      $("train-meta").textContent = "";
      $("train-metrics").innerHTML =
        '<p class="muted small">No finished run yet.</p>';
    }
    return;
  }

  const running = job.status === "running" || job.status === "queued";
  badge.textContent = job.status;

  if (running) {
    // A new run invalidates whatever the panel was showing.
    $("train-meta").textContent = "";
    $("train-metrics").innerHTML =
      '<p class="muted small">Waiting for the first epoch…</p>';
  }
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

  if (job.result && !running) {
    const hasMetrics = job.result.metrics && job.result.metrics.test_rows;
    if (hasMetrics) {
      renderTrainResult(job.result);
    } else {
      $("train-meta").textContent = stopReasonLabel(job.result.stop_reason);
      const note =
        (job.result.metrics && job.result.metrics.note) ||
        "Stopped before the first epoch finished, so there are no test metrics. Give it a longer time limit.";
      $("train-metrics").innerHTML = `<p class="muted small">${escapeHtml(note)}</p>`;
    }
  }
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

function renderTrainResult(result, options = {}) {
  const stored = Boolean(options.stored);
  const metrics = result.metrics || {};
  if (!stored) {
    $("train-meta").textContent =
      `${stopReasonLabel(result.stop_reason)} · ${result.epochs_done} epochs · ` +
      `${fmtDuration(result.duration_s)} · ${result.device}`;
  }

  const summary = [
    ["Balanced accuracy", fmtNum(metrics.balanced_accuracy, 3)],
    ["Constant-guess baseline", fmtNum(balancedBaseline(metrics), 3)],
    ["Plain accuracy", fmtNum(metrics.accuracy, 3)],
    ["Always-guess accuracy", fmtNum(metrics.baseline_accuracy, 3)],
    ["Edge (buy minus stop)", metrics.signal_edge == null ? "–" : fmtPct(metrics.signal_edge)],
    [
      "Bars it wanted to buy",
      metrics.long_coverage == null ? "–" : fmtPct(metrics.long_coverage, 1),
    ],
    ["Test rows", fmtInt(metrics.test_rows)],
  ];
  if (!stored) {
    summary.push(["Parameters", fmtInt(result.parameters)]);
    summary.push(["Device", `${result.device} · ${result.device_note}`]);
  }

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

  const verdict = verdictFor(metrics);
  const note = stopNote(result);
  $("train-metrics").innerHTML = `
    ${verdict ? `<div class="verdict ${verdict.tone}">${verdict.html}</div>` : ""}
    ${note ? `<p class="hint" style="margin:-4px 0 16px">${escapeHtml(note)}</p>` : ""}
    <div class="result-grid">
      <div>
        <h3 class="subhead">Summary</h3>
        <dl class="stats">${summary
          .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(String(value))}</dd>`)
          .join("")}</dl>
        <p class="hint" style="margin-top:12px">
          Balanced accuracy is the average recall across the three outcomes, so a
          model cannot win by always answering the same one. Compare it with the
          constant-guess line, not with plain accuracy. Edge is the average return
          when it said “buy” minus the average when it said “stop”, before fees.
        </p>
      </div>
      <div>
        <h3 class="subhead">Per class</h3>
        ${
          classRows
            ? `<table class="table"><thead><tr><th>outcome</th><th class="num">precision</th>` +
              `<th class="num">recall</th><th class="num">f1</th><th class="num">real cases</th>` +
              `</tr></thead><tbody>${classRows}</tbody></table>` +
              `<p class="hint" style="margin-top:8px">precision = when it said this, how often it was right. recall = of all the real cases, how many it caught.</p>`
            : '<p class="muted small">no metrics</p>'
        }
      </div>
    </div>
    <div class="result-grid">
      <div>
        <h3 class="subhead">Confusion matrix</h3>
        ${
          matrixRows
            ? `<table class="table"><thead>${matrixHeader}</thead><tbody>${matrixRows}</tbody></table>` +
              `<p class="hint" style="margin-top:8px">Rows are what really happened, columns what the model predicted. The diagonal is where it was right.</p>`
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
    epochs: Number($("train-epochs").value) || 100,
    window: Number($("train-window").value) || 128,
    batch_size: Number($("train-batch").value) || 256,
    lr: Number($("train-lr").value) || 0.0001,
    patience: Number($("train-patience").value) || 10,
    device: $("train-device").value,
  };
}

/* -------------------------------------------------------------- backtest */
function fmtMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "–";
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function backtestParams() {
  return {
    initial_equity: Number($("bt-equity").value) || 10000,
    fee_bps: numberOr($("bt-fees").value) ?? 10,
    slippage_bps: numberOr($("bt-slip").value) ?? 2,
    min_confidence: numberOr($("bt-conf").value) ?? 0,
  };
}

function ensureBtChart() {
  if (state.btChart || typeof LightweightCharts === "undefined") return;
  const container = $("bt-chart");
  state.btChart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight || 320,
    layout: { background: { color: "#11161f" }, textColor: "#7d8ca1", fontSize: 11 },
    grid: { vertLines: { color: "#161d29" }, horzLines: { color: "#161d29" } },
    rightPriceScale: { borderColor: "#1d2634" },
    timeScale: { borderColor: "#1d2634", timeVisible: true },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  state.btSeries = state.btChart.addLineSeries({
    color: "#d9b25f",
    lineWidth: 2,
    priceFormat: { type: "price", precision: 0, minMove: 1 },
  });
  state.btHoldSeries = state.btChart.addLineSeries({
    color: "#5a6c85",
    lineWidth: 1,
    lineStyle: 2,
    priceFormat: { type: "price", precision: 0, minMove: 1 },
  });
  new ResizeObserver(() => {
    if (!state.btChart) return;
    state.btChart.applyOptions({
      width: container.clientWidth,
      height: container.clientHeight,
    });
  }).observe(container);
}

function renderBacktestChart() {
  const report = state.info && state.info.backtest;
  if (!report || !state.btSeries) return;
  const metrics = report.metrics || {};
  const curve = (report.equity_curve || []).map((point) => ({
    time: Math.floor(point.time),
    value: point.equity,
  }));

  const startSeconds =
    report.slice && report.slice.first_ms
      ? Math.floor(report.slice.first_ms / 1000)
      : curve.length
      ? curve[0].time
      : null;
  if (startSeconds === null) return;

  const seen = new Set();
  const points = [];
  for (const point of [{ time: startSeconds, value: metrics.initial_equity }, ...curve]) {
    if (seen.has(point.time)) continue;
    seen.add(point.time);
    points.push(point);
  }
  points.sort((a, b) => a.time - b.time);

  state.btSeries.setData(points);
  const last = points[points.length - 1];
  state.btHoldSeries.setData([
    { time: startSeconds, value: metrics.initial_equity },
    { time: last.time, value: metrics.initial_equity * (1 + (metrics.buy_hold_return || 0)) },
  ]);
  state.btChart.timeScale().fitContent();
}

function renderBacktest() {
  const report = state.info && state.info.backtest;
  const badge = $("bt-badge");

  if (!report) {
    badge.textContent = "not run";
    badge.className = "badge";
    $("bt-verdict").innerHTML = "";
    $("bt-stats").innerHTML =
      '<dt>—</dt><dd>No backtest yet. Press “Run the backtest”.</dd>';
    $("bt-trades").innerHTML = "";
    $("bt-trade-count").textContent = "";
    $("bt-meta").textContent = "";
    return;
  }

  const metrics = report.metrics || {};
  const total = Number(metrics.total_return || 0);
  badge.textContent = `${total >= 0 ? "+" : ""}${(total * 100).toFixed(2)}%`;
  badge.className = "badge " + (total > 0 ? "ok" : "bad");

  const verdict = report.verdict || {};
  $("bt-verdict").innerHTML = verdict.text
    ? `<div class="verdict ${verdict.tone || ""}">${escapeHtml(verdict.text)}</div>`
    : "";

  const rows = [
    ["Final money", `${fmtMoney(metrics.final_equity)} USDT`],
    [
      "Profit / loss",
      `${metrics.pnl >= 0 ? "+" : ""}${fmtMoney(metrics.pnl)} USDT (${(total * 100).toFixed(2)}%)`,
    ],
    ["Just holding PAXG", fmtPct(metrics.buy_hold_return, 2)],
    ["Versus holding", fmtPct(metrics.excess_return, 2)],
    ["Trades", fmtInt(metrics.trades)],
    ["Win rate", metrics.win_rate == null ? "–" : fmtPct(metrics.win_rate, 1)],
    ["Max drawdown", fmtPct(metrics.max_drawdown, 2)],
    [
      "Winning trades captured",
      metrics.avg_gross_win == null
        ? "–"
        : `${(metrics.avg_gross_win * 10000).toFixed(1)} bps`,
    ],
    [
      "Round-trip cost",
      metrics.round_trip_cost == null
        ? "–"
        : `${(metrics.round_trip_cost * 10000).toFixed(0)} bps`,
    ],
    ["Sharpe (daily)", metrics.sharpe == null ? "–" : fmtNum(metrics.sharpe, 2)],
    ["Test window", `${fmtDate(metrics.start_ms)} → ${fmtDate(metrics.end_ms)}`],
    [
      "Costs charged",
      `${metrics.fees_bps} bps fee + ${metrics.slippage_bps} bps slippage, each side`,
    ],
  ];
  $("bt-stats").innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");

  $("bt-meta").textContent = report.model
    ? `${report.model.name} · ${fmtInt(report.slice && report.slice.rows)} scored rows`
    : "";

  const trades = report.trades || [];
  $("bt-trade-count").textContent = `(${fmtInt(report.trade_count || trades.length)})`;
  const table = $("bt-trades");
  if (!trades.length) {
    table.innerHTML = '<tbody><tr><td class="muted">no trades were taken</td></tr></tbody>';
  } else {
    const body = trades
      .slice()
      .reverse()
      .map(
        (trade) =>
          `<tr><td>${fmtDate(trade.entry_ms)}</td><td>${fmtDate(trade.exit_ms)}</td>` +
          `<td class="num">${fmtNum(trade.entry, 2)}</td><td class="num">${fmtNum(trade.exit, 2)}</td>` +
          `<td class="num">${fmtNum(trade.net_pct, 3)}%</td><td class="num">${fmtMoney(trade.pnl)}</td>` +
          `<td class="num">${fmtInt(trade.bars)}</td></tr>`
      )
      .join("");
    table.innerHTML =
      '<thead><tr><th>entry</th><th>exit</th><th class="num">buy at</th><th class="num">sell at</th>' +
      '<th class="num">net</th><th class="num">P&amp;L</th><th class="num">bars</th></tr></thead>' +
      `<tbody>${body}</tbody>`;
  }

  renderBacktestChart();
}

/* ---------------------------------------------------------------- search */
function searchParams() {
  return {
    total_minutes: (Number($("search-hours").value) || 3) * 60,
    per_candidate_minutes: Number($("search-per").value) || 4,
    max_candidates: Number($("search-max").value) || 40,
    fee_bps: numberOr($("search-fees").value) ?? 10,
    slippage_bps: numberOr($("search-slip").value) ?? 2,
    device: $("search-device").value,
  };
}

function ensureSearchChart() {
  if (state.searchChart || typeof LightweightCharts === "undefined") return;
  const container = $("search-chart");
  state.searchChart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: container.clientHeight || 320,
    layout: { background: { color: "#11161f" }, textColor: "#7d8ca1", fontSize: 11 },
    grid: { vertLines: { color: "#161d29" }, horzLines: { color: "#161d29" } },
    rightPriceScale: { borderColor: "#1d2634" },
    timeScale: { borderColor: "#1d2634", timeVisible: true },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  state.searchSeries = state.searchChart.addLineSeries({
    color: "#d9b25f",
    lineWidth: 2,
    priceFormat: { type: "price", precision: 0, minMove: 1 },
  });
  new ResizeObserver(() => {
    if (!state.searchChart) return;
    state.searchChart.applyOptions({
      width: container.clientWidth,
      height: container.clientHeight,
    });
  }).observe(container);
}

function renderSearchChart(report) {
  if (!state.searchSeries || !report) return;
  const seen = new Set();
  const points = [];
  for (const point of report.best_curve || []) {
    const time = Math.floor(point.time);
    if (seen.has(time)) continue;
    seen.add(time);
    points.push({ time, value: point.equity });
  }
  points.sort((a, b) => a.time - b.time);
  state.searchSeries.setData(points);
  state.searchChart.timeScale().fitContent();
}

function renderSearch() {
  const jobRunning = state.jobs.some(
    (job) => job.kind === "search" && (job.status === "running" || job.status === "queued")
  );
  // While a new sweep is starting, do not show the previous run's leaderboard:
  // it looks like results for the run that is just beginning.
  const report = state.search || (!jobRunning ? state.info && state.info.search : null);
  const badge = $("search-state");
  const table = $("search-table");

  if (!report) {
    badge.textContent = jobRunning ? "starting" : "never run";
    badge.className = "badge" + (jobRunning ? " amber" : "");
    $("search-stats").innerHTML = jobRunning
      ? '<dt>—</dt><dd>Starting the sweep…</dd>'
      : '<dt>—</dt><dd>No search yet. Set a budget and press “Start the search”.</dd>';
    $("search-best").textContent = "";
    $("search-note").textContent = "";
    $("search-curve-meta").textContent = "";
    table.innerHTML = "";
    $("stop-search").disabled = true;
    return;
  }

  // A sweep writes "running" to disk as it goes. If the server restarts, the
  // report keeps saying "running" forever because nothing is left to rewrite
  // it — say "interrupted" rather than pretending a dead sweep is alive.
  const stale = report.status === "running" && !jobRunning;
  const running = (report.status === "running" && !stale) || jobRunning;
  badge.textContent = stale ? "interrupted" : running ? "running" : report.status;
  badge.className =
    "badge " +
    (report.status === "done" && !running ? "ok" : stale ? "bad" : running ? "amber" : "");

  const rows = [
    [
      "candidates finished",
      `${fmtInt(report.candidates_done)} / ${fmtInt(report.candidates_total)}`,
    ],
    ["elapsed", `${fmtNum(report.elapsed_minutes, 1)} min`],
    ["mean ATR", `${fmtNum(report.atr_bps, 1)} bps`],
    ["round-trip cost", `${fmtInt(report.round_trip_cost_bps)} bps`],
    ["best so far", report.best ? report.best.candidate.key : "–"],
    [
      "best validation",
      report.best ? fmtPct(report.best.validation.metrics.total_return, 2) : "–",
    ],
    ["best test", report.best ? fmtPct(report.best.test.metrics.total_return, 2) : "–"],
  ];
  $("search-stats").innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");

  $("search-note").textContent = stale
    ? "The sweep is no longer running — the server restarted while it was going. " +
      "The candidates below are complete; start a new sweep to continue."
    : `Candidates must aim at least ${fmtNum(report.round_trip_cost_bps * 2.5, 0)} bps ` +
      `(2.5x the ${fmtNum(report.round_trip_cost_bps, 0)} bps round trip) or they are skipped.`;

  $("search-best").textContent = report.best
    ? report.best.test.verdict
      ? report.best.test.verdict.text
      : ""
    : "";

  const ranked = report.leaderboard || [];
  if (!ranked.length) {
    table.innerHTML =
      '<tbody><tr><td class="muted">' +
      (running ? "first candidate is training…" : "no candidate produced a usable model") +
      "</td></tr></tbody>";
  } else {
    const body = ranked
      .map((row) => {
        const candidate = row.candidate;
        const validation = row.validation.metrics;
        const test = row.test.metrics;
        const cls = (value) => (value >= 0 ? "pos" : "neg");
        const scoreCell =
          row.score === null || row.score === undefined
            ? '<td class="num muted" title="fewer than 20 trades, so it is not ranked">—</td>'
            : `<td class="num ${cls(row.score)}" title="validation return minus a quarter of its max drawdown">${(
                row.score * 100
              ).toFixed(1)}%</td>`;
        return (
          `<tr><td class="num">${fmtInt(row.rank)}</td>` + scoreCell +
          `<td class="num">${candidate.k_up}</td><td class="num">${candidate.k_down}</td>` +
          `<td class="num">${fmtInt(candidate.holding)}</td><td class="num">${fmtInt(candidate.window)}</td>` +
          `<td class="num ${cls(validation.total_return)}">${fmtPct(validation.total_return, 2)}</td>` +
          `<td class="num">${fmtInt(validation.trades)}</td>` +
          `<td class="num ${cls(test.total_return)}">${fmtPct(test.total_return, 2)}</td></tr>`
        );
      })
      .join("");
    table.innerHTML =
      '<thead><tr><th class="num">#</th>' +
      '<th class="num" title="validation return minus a quarter of its max drawdown — the ranking key">score</th>' +
      '<th class="num">target x</th><th class="num">stop x</th>' +
      '<th class="num">hold</th><th class="num">window</th><th class="num">validation</th>' +
      '<th class="num">trades</th><th class="num">test</th></tr></thead>' +
      `<tbody>${body}</tbody>`;
  }

  $("search-curve-meta").textContent = report.best
    ? `${report.best.model_name || report.best.candidate.key}`
    : "";
  ensureSearchChart();
  renderSearchChart(report);
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

  $("next-step-action").addEventListener("click", () => {
    if (state.nextPanel) switchPanel(state.nextPanel);
  });

  $("dismiss-intro").addEventListener("click", () => {
    localStorage.setItem("goldneural.introHidden", "1");
    $("intro").hidden = true;
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

  $("run-backtest").addEventListener("click", () => {
    startJob("/api/backtest/run", backtestParams(), "backtest");
  });

  $("start-search").addEventListener("click", async () => {
    const jobId = await startJob("/api/search/start", searchParams(), "search");
    if (jobId) {
      state.search = null;
      renderSearch();
    }
  });

  $("stop-search").addEventListener("click", async () => {
    const job = state.jobs.find(
      (entry) =>
        entry.kind === "search" &&
        (entry.status === "running" || entry.status === "queued")
    );
    if (!job) return;
    try {
      await api(`/api/jobs/${job.id}/stop`, { method: "POST", body: {} });
      toast("stop requested — the current candidate will finish and save", "info");
    } catch (error) {
      if (error.message !== "unauthorised") toast(error.message, "error");
    }
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
