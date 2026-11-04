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
  walkforward: null,
  research: null,
  researchSummary: null,
  researchSpec: null,
  rangeBars: 288,
  ws: null,
  wsRetry: null,
  refreshTimer: null,
  refreshing: false,
  apiOnline: false,
  wsConnected: false,
  lastRefresh: null,
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
    .querySelectorAll("#tabs button[data-group]")
    .forEach((button) => button.classList.remove("active"));
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
    renderWalkforward();
  }
  if (name === "research") {
    refreshResearch().catch((error) => {
      if (error.message !== "unauthorised") toast(error.message, "error");
    });
  }
}

const PANEL_GROUPS = {
  data: ["data", "dataset", "leakage", "exchange"],
  model: ["training", "search", "research"],
  backtest: ["backtest"],
};

function switchGroup(name) {
  const panels = PANEL_GROUPS[name] || [name];
  document
    .querySelectorAll("#tabs button[data-group]")
    .forEach((button) => button.classList.toggle("active", button.dataset.group === name));
  document.querySelectorAll("#tabs button[data-panel]").forEach((button) => {
    button.classList.toggle("active", panels.includes(button.dataset.panel));
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle(
      "active",
      panels.includes(panel.id.replace("panel-", ""))
    );
  });

  if (panels.includes("data")) {
    ensureChart();
    loadBars();
    renderExchange();
  }
  if (panels.includes("training")) {
    renderTrainingPanel();
    const job = currentTrainingJob();
    if (job) updateTrainLog(job.id);
  }
  if (panels.includes("backtest")) {
    ensureBtChart();
    renderBacktestChart();
  }
  if (panels.includes("search")) renderWalkforward();
  if (panels.includes("research")) {
    refreshResearch().catch((error) => {
      if (error.message !== "unauthorised") toast(error.message, "error");
    });
  }
  window.scrollTo({ top: 0, behavior: "auto" });
}

/** Open the tab group that owns a panel; used by the next-step banner. */
function openPanel(name) {
  for (const [group, panels] of Object.entries(PANEL_GROUPS)) {
    if (!panels.includes(name)) continue;
    switchGroup(group);
    const target = document.getElementById(`panel-${name}`);
    if (target) {
      requestAnimationFrame(() =>
        target.scrollIntoView({ behavior: "smooth", block: "start" })
      );
    }
    return;
  }
  switchPanel(name);
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

  // The sweep rewrites its status file after every fold; poll it so the
  // leaderboard fills in while the job is still running.
  setInterval(() => {
    const running = state.jobs.some(
      (job) =>
        job.kind === "walkforward" && (job.status === "running" || job.status === "queued")
    );
    if (!running) return;
    api("/api/walkforward/last")
      .then((report) => {
        state.walkforward = report;
        renderWalkforward();
      })
      .catch(() => {});
  }, 3000);

  setInterval(() => {
    if ($("panel-research").classList.contains("active")) {
      refreshResearch().catch(() => {});
    }
  }, 4000);

  setInterval(() => {
    if (state.token) refreshAll().catch(() => {});
  }, 15000);

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
    const [status, info, jobs, models, researchSummary] = await Promise.all([
      api("/api/status"),
      api("/api/data/info"),
      api("/api/jobs"),
      api("/api/models"),
      api("/api/research/status?folds=0").catch(() => null),
    ]);
    state.status = status;
    state.info = info;
    state.apiOnline = true;
    state.lastRefresh = Date.now();
    state.models = models.models || [];
    state.researchSummary = researchSummary;
    if (info.walkforward) state.walkforward = info.walkforward;
    applyJobs(jobs.jobs || []);
    renderStatus();
    renderInfo();
    renderDataset();
    renderLeakage();
    renderExchange();
    renderTraining();
    renderTrainingPanel();
    renderBacktest();
    renderWalkforward();
    renderPipeline();
    renderIntro();
  } catch (error) {
    state.apiOnline = false;
    renderStatus();
    throw error;
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

  socket.onopen = () => {
    state.wsConnected = true;
    renderStatus();
  };

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
    state.wsConnected = false;
    renderStatus();
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
    search: info.walkforward || null,
    hasSearch: Boolean(info.walkforward),
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
      text: "One model is one window. Test settings across every 90-day block of the last six years instead.",
      action: "Take me there",
    };
  }
  if (stage.search && stage.search.status === "running") {
    return {
      panel: "search",
      text: "The sweep is running. A setting appears on the leaderboard once all of its folds finish.",
      action: "Watch it",
    };
  }
  return {
    panel: "search",
    text: "Everything is set up. Compare the leaderboard, or test another region.",
    action: "Open walk-forward",
  };
}

function renderPipeline() {
  const stage = computeStage();
  const rs = state.researchSummary || state.research;

  const dots = {
    data: stage.hasBars ? "done" : "",
    dataset: stage.hasDataset ? "done" : "",
    leakage:
      stage.leakageState === "pass" ? "done" : stage.leakageState === "fail" ? "error" : "",
    training: stage.hasModel ? "done" : "",
    backtest: stage.hasBacktest ? "done" : "",
    search: stage.search && stage.search.status === "done" ? "done" : "",
    research: rs
      ? rs.running
        ? "warn"
        : rs.status && rs.status.status === "done"
          ? "done"
          : ""
      : "",
  };
  Object.entries(dots).forEach(([key, cls]) => {
    const dot = document.querySelector(`[data-dot="${key}"]`);
    if (dot) dot.className = "tab-dot" + (cls ? " " + cls : "");
  });

  const groupDots = {
    data: stage.hasBars ? "done" : "",
    model:
      rs && rs.running
        ? "warn"
        : stage.hasModel || stage.search || (rs && rs.best_spec)
          ? "done"
          : "",
    backtest: stage.hasBacktest ? "done" : "",
  };
  Object.entries(groupDots).forEach(([key, cls]) => {
    const dot = document.querySelector(`[data-group-dot="${key}"]`);
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

  const updated = state.lastRefresh
    ? new Date(state.lastRefresh).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "waiting";
  const chips = [
    `<span class="chip signal ${state.apiOnline ? "ok" : "bad"}">${state.apiOnline ? "API online" : "API offline"}</span>`,
    `<span class="chip signal ${state.wsConnected ? "ok" : ""}">${state.wsConnected ? "live updates" : "reconnecting"}</span>`,
    `<span class="chip">${status.symbol} · ${status.interval}</span>`,
  ];
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
  chips.push(`<span class="chip freshness">updated ${updated}</span>`);

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
  const firstY = height - pad - ((values[0] - min) / span) * (height - pad * 2);

  return `<svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
    <line x1="${pad}" x2="${width - pad}" y1="${firstY.toFixed(1)}" y2="${firstY.toFixed(1)}"
      stroke="#5a6c85" stroke-width="1" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>
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
        <p class="muted small">lower is better · dashed = first epoch · best epoch ${fmtInt(metrics.best_epoch)} · best val loss ${fmtNum(metrics.best_val_loss, 4)}</p>
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
    epochs: numberOr($("train-epochs").value) ?? 0,
    window: Number($("train-window").value) || 64,
    batch_size: Number($("train-batch").value) || 256,
    lr: Number($("train-lr").value) || 0.0003,
    patience: numberOr($("train-patience").value) ?? 0,
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
  if (!report || !state.btSeries) {
    const verdict = $("bt-chart-verdict");
    if (verdict) verdict.textContent = "";
    return;
  }
  const metrics = report.metrics || {};
  const total = Number(metrics.total_return) || 0;
  const hold = Number(metrics.buy_hold_return) || 0;
  const lead = total - hold;
  const verdict = $("bt-chart-verdict");
  if (verdict) {
    verdict.textContent =
      `Model ${fmtPct(total, 1)} vs holding ${fmtPct(hold, 1)} — ` +
      `${lead >= 0 ? "ahead" : "behind"} by ${fmtPct(Math.abs(lead), 1)}`;
    verdict.className = "chart-verdict " + (lead >= 0 ? "pos" : "neg");
  }
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

/* ---------------------------------------------------------- walk-forward */
function numberList(id, cast = Number) {
  return $(id)
    .value.split(",")
    .map((part) => cast(part.trim()))
    .filter((value) => Number.isFinite(value));
}

function wfParams() {
  return {
    interval: $("wf-interval").value,
    policy: $("wf-policy").value,
    k_up: numberList("wf-k-up"),
    k_down: numberList("wf-k-down"),
    holding: numberList("wf-holding", (v) => parseInt(v, 10)),
    window: numberList("wf-window", (v) => parseInt(v, 10)),
    seeds: numberList("wf-seeds", (v) => parseInt(v, 10)),
    test_days: Number($("wf-test-days").value) || 90,
    val_days: Number($("wf-val-days").value) || 90,
    epochs: Number($("wf-epochs").value) || 15,
  };
}

function currentWalkforwardJob() {
  const jobs = state.jobs.filter((job) => job.kind === "walkforward");
  if (!jobs.length) return null;
  return (
    jobs.find((job) => job.status === "running" || job.status === "queued") || jobs[0]
  );
}

const shareText = (value, digits = 0) =>
  value === null || value === undefined ? "–" : (Number(value) * 100).toFixed(digits) + "%";

function renderWalkforward() {
  const report = state.walkforward;
  const job = currentWalkforwardJob();
  const running = Boolean(
    job && (job.status === "running" || job.status === "queued")
  );

  $("start-walkforward").disabled = running;
  $("stop-walkforward").disabled = !running;

  const badge = $("wf-state");
  if (job) {
    badge.textContent = job.status;
    badge.className =
      "badge " +
      (running ? "amber" : job.status === "ok" ? "ok" : job.status === "failed" ? "bad" : "");
  } else {
    badge.textContent = report ? report.status : "idle";
    badge.className = "badge" + (report && report.status === "done" ? " ok" : "");
  }

  const stats = $("wf-stats");
  if (!report) {
    stats.innerHTML =
      '<dt>—</dt><dd>No sweep yet. Pick a region above and press Start.</dd>';
    $("wf-table").innerHTML = "";
    return;
  }

  stats.innerHTML = [
    ["status", running ? "running" : report.status],
    [
      "settings",
      `${fmtInt(report.specs_done)}${report.specs_total ? " / " + fmtInt(report.specs_total) : ""}`,
    ],
    ["folds scored", fmtInt(report.folds_total)],
    ["elapsed", `${fmtNum(report.elapsed_minutes, 1)} min`],
    ["now", running && job ? job.message : report.current || "—"],
  ]
    .map(([key, value]) => `<dt>${key}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");

  const rows = (report.leaderboard || []).slice(0, 20);
  const table = $("wf-table");
  if (!rows.length) {
    table.innerHTML =
      '<tbody><tr><td class="muted">No setting has finished all of its folds yet — a setting needs every fold before it is ranked.</td></tr></tbody>';
    return;
  }

  table.innerHTML =
    '<thead><tr><th>setting</th><th class="num">folds</th>' +
    '<th class="num">median quarter</th><th class="num">worst</th>' +
    '<th class="num">≥ +5%</th><th class="num">beats hold</th>' +
    '<th class="num">compounded</th></tr></thead><tbody>' +
    rows
      .map((row) => {
        const tone = (value) => (value >= 0 ? "pos" : "neg");
        return (
          `<tr><td>${escapeHtml(row.spec)}</td>` +
          `<td class="num">${fmtInt(row.folds)}</td>` +
          `<td class="num ${tone(row.median_return)}">${shareText(row.median_return, 2)}</td>` +
          `<td class="num neg">${shareText(row.worst_return, 2)}</td>` +
          `<td class="num">${shareText(row.share_over_5pct, 0)}</td>` +
          `<td class="num">${shareText(row.share_beating_hold, 0)}</td>` +
          `<td class="num ${tone(row.compounded)}">${shareText(row.compounded, 1)}</td>` +
          "</tr>"
        );
      })
      .join("") +
    "</tbody>";
}

/* --------------------------------------------------------------- research */
function rsParams() {
  return {
    interval: $("rs-interval").value,
    policy: $("rs-policy").value,
    k_up: numberList("rs-k-up"),
    k_down: numberList("rs-k-down"),
    holding: numberList("rs-holding", (value) => parseInt(value, 10)),
    window: numberList("rs-window", (value) => parseInt(value, 10)),
    seeds: numberList("rs-seeds", (value) => parseInt(value, 10)),
    epochs: numberOr($("rs-epochs").value) ?? 15,
    minutes_per_fold: numberOr($("rs-minutes").value) ?? 3,
    test_days: numberOr($("rs-test-days").value) ?? 90,
    val_days: numberOr($("rs-val-days").value) ?? 90,
    min_train_days: numberOr($("rs-min-train-days").value) ?? 365,
    fee_bps: numberOr($("rs-fee-bps").value) ?? 10,
    slippage_bps: numberOr($("rs-slip-bps").value) ?? 2,
    stop_thresholds: numberList("rs-thresholds"),
    reentry_margin: numberOr($("rs-reentry").value) ?? 0.05,
    min_flat_bars: numberOr($("rs-min-flat").value) ?? 1,
    device: $("rs-device").value,
  };
}

function currentResearchSpec() {
  const report = state.research;
  return state.researchSpec || (report && report.best_spec) || null;
}

async function refreshResearch(spec = state.researchSpec) {
  if (!state.token) return;
  const suffix = spec ? `?spec=${encodeURIComponent(spec)}` : "";
  state.research = await api(`/api/research/status${suffix}`);
  state.researchSummary = state.research;
  renderPipeline();
  renderResearch();
}

function renderResearch() {
  const report = state.research;
  const status = (report && report.status) || null;
  const running = Boolean(report && report.running);
  const stopping = Boolean(report && report.stop_requested);

  $("rs-start").disabled = running;
  $("rs-stop").disabled = !running || stopping;

  let label = "idle";
  if (running) label = stopping ? "stopping" : "running";
  else if (status && status.status) {
    label = status.status === "running" ? "exited" : status.status;
  }

  const badge = $("rs-state");
  badge.textContent = label;
  badge.className =
    "badge" +
    (running ? " warn" : status && status.status === "done" ? " ok" : "");

  const stats = $("rs-stats");
  if (!status) {
    stats.innerHTML =
      "<dt>—</dt><dd>No campaign yet. Set a grid and press Start.</dd>";
  } else {
    const rows = [
      ["state", label],
      ["policy", status.policy || "—"],
      [
        "settings",
        `${fmtInt(status.specs_done ?? status.specs_tracked)}${
          status.specs_total ? " / " + fmtInt(status.specs_total) : ""
        }`,
      ],
      [
        "remaining",
        status.remaining == null ? "—" : `${fmtInt(status.remaining)} settings`,
      ],
      ["seed", status.seed == null ? "—" : String(status.seed)],
      ["folds scored", fmtInt(status.folds_total)],
      ["rate", `${fmtNum(status.folds_per_minute, 1)} folds/min`],
      ["elapsed", `${fmtNum(status.elapsed_minutes, 1)} min`],
      ["now", status.current || "—"],
    ];
    stats.innerHTML = rows
      .map(
        ([key, value]) =>
          `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd>`
      )
      .join("");
  }

  const log = $("rs-log");
  const lines = (report && report.log_tail) || [];
  log.textContent = lines.length ? lines.join("\n") : "No campaign log yet.";

  const folds = (report && report.folds) || [];
  const selected = currentResearchSpec();
  const note = folds.length ? `${folds.length} quarters · ${selected || ""}` : "";
  $("rs-quarter-note").textContent = note;
  $("rs-cum-note").textContent = note;
  $("rs-best").textContent = selected || "";

  renderResearchLeaderboard(status, selected);
  renderResearchCharts(folds);

  const quarterVerdict = $("rs-quarter-verdict");
  const cumVerdict = $("rs-cum-verdict");
  if (!folds.length) {
    quarterVerdict.textContent = "";
    quarterVerdict.className = "chart-verdict";
    cumVerdict.textContent = "";
    cumVerdict.className = "chart-verdict";
  } else {
    const excesses = folds.map((fold) => Number(fold.excess) || 0);
    const sorted = [...excesses].sort((a, b) => a - b);
    const median =
      sorted.length % 2
        ? sorted[(sorted.length - 1) / 2]
        : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
    const wins = excesses.filter((value) => value > 0).length;
    const worst = Math.min(...folds.map((fold) => Number(fold.return) || 0));
    quarterVerdict.textContent =
      `Beat holding in ${wins} of ${folds.length} quarters · ` +
      `median excess ${fmtPct(median, 2)} · worst quarter ${fmtPct(worst, 2)}`;
    quarterVerdict.className =
      "chart-verdict " + (median >= 0 ? "pos" : "neg");

    let strategy = 1;
    let hold = 1;
    for (const fold of folds) {
      strategy *= 1 + (Number(fold.return) || 0);
      hold *= 1 + (Number(fold.buy_hold) || 0);
    }
    const lead = strategy - hold;
    cumVerdict.textContent =
      `Strategy ${fmtPct(strategy - 1, 1)} vs holding ${fmtPct(hold - 1, 1)} — ` +
      `${lead >= 0 ? "ahead" : "behind"} by ${fmtPct(Math.abs(lead), 1)}`;
    cumVerdict.className = "chart-verdict " + (lead >= 0 ? "pos" : "neg");
  }
}

function renderResearchLeaderboard(status, selected) {
  const table = $("rs-leaderboard");
  const rows = [...((status && status.leaderboard) || [])];
  rows.sort((a, b) => {
    const completeA = a.complete !== false ? 1 : 0;
    const completeB = b.complete !== false ? 1 : 0;
    if (completeA !== completeB) return completeB - completeA;
    const winsA = Number(a.share_beating_hold) || 0;
    const winsB = Number(b.share_beating_hold) || 0;
    if (winsA !== winsB) return winsB - winsA;
    return (Number(b.compounded) || 0) - (Number(a.compounded) || 0);
  });
  const note = $("rs-board-note");
  if (note) {
    note.textContent = rows.length
      ? `${rows.length} settings · sorted by wins, then compounded`
      : "";
  }
  if (!rows.length) {
    table.innerHTML =
      '<tbody><tr><td class="muted">No setting has finished all of its folds yet.</td></tr></tbody>';
    return;
  }

  const tone = (value) => (value >= 0 ? "pos" : "neg");
  table.innerHTML =
    "<thead><tr><th>setting</th><th class=\"num\">folds</th>" +
    "<th class=\"num\">median</th><th class=\"num\">excess</th>" +
    "<th class=\"num\">worst</th><th class=\"num\">beats hold</th>" +
    "<th class=\"num\">compounded</th></tr></thead><tbody>" +
    rows
      .map((row) => {
        const isSelected = selected && row.spec === selected;
        const complete = row.complete !== false;
        const wanted = Number(row.expected_folds || row.folds || 0);
        const folds =
          wanted > Number(row.folds || 0)
            ? `${fmtInt(row.folds)}/${fmtInt(wanted)}`
            : fmtInt(row.folds);
        const wins = Math.round(
          (Number(row.share_beating_hold) || 0) * (Number(row.folds) || 0)
        );
        return (
          `<tr data-spec="${escapeHtml(row.spec)}" class="${isSelected ? "selected" : ""}${complete ? "" : " partial"}">` +
          `<td>${escapeHtml(row.spec)}${complete ? "" : ' <span class="muted small">partial</span>'}</td>` +
          `<td class="num">${folds}</td>` +
          `<td class="num ${tone(row.median_return)}">${shareText(row.median_return, 2)}</td>` +
          `<td class="num ${tone(row.median_excess)}">${shareText(row.median_excess, 2)}</td>` +
          `<td class="num neg">${shareText(row.worst_return, 2)}</td>` +
          `<td class="num" title="${shareText(row.share_beating_hold, 1)}">${wins}/${fmtInt(row.folds)}</td>` +
          `<td class="num ${tone(row.compounded)}">${shareText(row.compounded, 1)}</td>` +
          "</tr>"
        );
      })
      .join("") +
    "</tbody>";
}

function svgTag(name, attrs = {}, content = "") {
  const parts = Object.entries(attrs)
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([key, value]) => `${key}="${escapeHtml(value)}"`)
    .join(" ");
  return `<${name}${parts ? " " + parts : ""}>${content}</${name}>`;
}

function quarterLabel(fold, index) {
  const ms = Number(fold.test_start_ms);
  return Number.isFinite(ms) && ms > 0 && ms < 8.64e15
    ? new Date(ms).toISOString().slice(0, 7)
    : `#${index + 1}`;
}

function renderResearchCharts(folds) {
  renderQuarterChart($("rs-quarter-chart"), folds);
  renderCumulativeChart($("rs-cum-chart"), folds);
}

function renderQuarterChart(host, folds) {
  if (!host) return;
  if (!folds || !folds.length) {
    host.innerHTML =
      '<p class="muted small">Waiting for the first setting to finish all of its folds.</p>';
    return;
  }

  const W = Math.max(360, host.clientWidth || 720);
  const H = 250;
  const m = { l: 52, r: 12, t: 14, b: 34 };
  const plotW = W - m.l - m.r;
  const plotH = H - m.t - m.b;

  const values = folds.flatMap((fold) => [
    Number(fold.return) || 0,
    Number(fold.buy_hold) || 0,
  ]);
  let lo = Math.min(0, ...values);
  let hi = Math.max(0, ...values);
  const pad = (hi - lo) * 0.14 || 0.01;
  lo -= pad;
  hi += pad;

  const y = (value) => m.t + ((hi - value) / (hi - lo)) * plotH;
  const zero = y(0);
  const group = plotW / folds.length;
  const bw = Math.max(3, Math.min(15, group * 0.3));
  const parts = [];

  [hi, 0, lo].forEach((value) => {
    parts.push(
      svgTag("line", {
        x1: m.l,
        x2: W - m.r,
        y1: y(value),
        y2: y(value),
        stroke: "#1d2634",
        "stroke-width": 1,
      })
    );
    parts.push(
      svgTag(
        "text",
        {
          x: m.l - 8,
          y: y(value) + 4,
          "text-anchor": "end",
          fill: "#7d8ca1",
          "font-size": 11,
        },
        escapeHtml(`${(value * 100).toFixed(0)}%`)
      )
    );
  });

  const every = Math.max(1, Math.ceil(folds.length / 8));
  folds.forEach((fold, index) => {
    const cx = m.l + group * (index + 0.5);
    const strategy = Number(fold.return) || 0;
    const hold = Number(fold.buy_hold) || 0;
    const label = quarterLabel(fold, index);
    const tip =
      `${label} · strategy ${fmtPct(strategy, 2)} · hold ${fmtPct(hold, 2)}` +
      ` · excess ${fmtPct(strategy - hold, 2)}`;
    const bar = (x, value, colour) => {
      const top = Math.min(y(value), zero);
      const height = Math.max(1, Math.abs(y(value) - zero));
      return svgTag(
        "rect",
        { x, y: top, width: bw, height, rx: 1, fill: colour },
        svgTag("title", {}, escapeHtml(tip))
      );
    };
    parts.push(bar(cx - bw - 1, strategy, "#d9b25f"));
    parts.push(bar(cx + 1, hold, "#5a6c85"));
    if (index % every === 0) {
      parts.push(
        svgTag(
          "text",
          {
            x: cx,
            y: H - 12,
            "text-anchor": "middle",
            fill: "#7d8ca1",
            "font-size": 10,
          },
          escapeHtml(label)
        )
      );
    }
  });

  host.innerHTML = svgTag(
    "svg",
    {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: H,
      role: "img",
      "aria-label": "Quarterly return, strategy versus buy and hold",
    },
    parts.join("")
  );
}

function renderCumulativeChart(host, folds) {
  if (!host) return;
  if (!folds || !folds.length) {
    host.innerHTML =
      '<p class="muted small">Waiting for the first setting to finish all of its folds.</p>';
    return;
  }

  const W = Math.max(360, host.clientWidth || 720);
  const H = 250;
  const m = { l: 52, r: 14, t: 14, b: 30 };
  const plotW = W - m.l - m.r;
  const plotH = H - m.t - m.b;

  let strategy = 1;
  let hold = 1;
  const points = folds.map((fold) => {
    strategy *= 1 + (Number(fold.return) || 0);
    hold *= 1 + (Number(fold.buy_hold) || 0);
    return { strategy, hold, fold };
  });

  const values = points.flatMap((point) => [point.strategy, point.hold]);
  let lo = Math.min(1, ...values);
  let hi = Math.max(1, ...values);
  const pad = (hi - lo) * 0.1 || 0.01;
  lo -= pad;
  hi += pad;

  const y = (value) => m.t + ((hi - value) / (hi - lo)) * plotH;
  const x = (index) =>
    m.l +
    (points.length === 1 ? plotW / 2 : (plotW * index) / (points.length - 1));
  const line = (key) =>
    points
      .map(
        (point, index) =>
          `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(
            point[key]
          ).toFixed(1)}`
      )
      .join(" ");

  const parts = [];
  [hi, 1, lo].forEach((value) => {
    parts.push(
      svgTag("line", {
        x1: m.l,
        x2: W - m.r,
        y1: y(value),
        y2: y(value),
        stroke: "#1d2634",
        "stroke-width": 1,
      })
    );
    parts.push(
      svgTag(
        "text",
        {
          x: m.l - 8,
          y: y(value) + 4,
          "text-anchor": "end",
          fill: "#7d8ca1",
          "font-size": 11,
        },
        escapeHtml(`${((value - 1) * 100).toFixed(0)}%`)
      )
    );
  });

  parts.push(
    svgTag("path", {
      d: line("strategy"),
      fill: "none",
      stroke: "#d9b25f",
      "stroke-width": 2,
    })
  );
  parts.push(
    svgTag("path", {
      d: line("hold"),
      fill: "none",
      stroke: "#5a6c85",
      "stroke-width": 1.5,
      "stroke-dasharray": "4 4",
    })
  );

  points.forEach((point, index) => {
    parts.push(
      svgTag(
        "circle",
        { cx: x(index), cy: y(point.strategy), r: 3, fill: "#d9b25f" },
        svgTag(
          "title",
          {},
          escapeHtml(
            `${quarterLabel(point.fold, index)} · strategy ${fmtPct(
              point.strategy - 1,
              1
            )} · hold ${fmtPct(point.hold - 1, 1)}`
          )
        )
      )
    );
  });

  const first = points[0];
  const last = points[points.length - 1];
  parts.push(
    svgTag(
      "text",
      { x: m.l, y: H - 8, "text-anchor": "start", fill: "#7d8ca1", "font-size": 10 },
      escapeHtml(quarterLabel(first.fold, 0))
    )
  );
  parts.push(
    svgTag(
      "text",
      { x: W - m.r, y: H - 8, "text-anchor": "end", fill: "#7d8ca1", "font-size": 10 },
      escapeHtml(quarterLabel(last.fold, points.length - 1))
    )
  );
  parts.push(
    svgTag(
      "text",
      {
        x: W - m.r,
        y: y(last.strategy) - 6,
        "text-anchor": "end",
        fill: "#d9b25f",
        "font-size": 11,
      },
      escapeHtml(fmtPct(last.strategy - 1, 1))
    )
  );
  parts.push(
    svgTag(
      "text",
      {
        x: W - m.r,
        y: y(last.hold) + 12,
        "text-anchor": "end",
        fill: "#5a6c85",
        "font-size": 11,
      },
      escapeHtml(fmtPct(last.hold - 1, 1))
    )
  );

  host.innerHTML = svgTag(
    "svg",
    {
      viewBox: `0 0 ${W} ${H}`,
      width: "100%",
      height: H,
      role: "img",
      "aria-label": "Cumulative return, strategy versus buy and hold",
    },
    parts.join("")
  );
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
    const group = event.target.closest("button[data-group]");
    if (group) {
      switchGroup(group.dataset.group);
      return;
    }
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
    if (state.nextPanel) openPanel(state.nextPanel);
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

  $("start-walkforward").addEventListener("click", async () => {
    const params = wfParams();
    if (!params.k_up.length || !params.k_down.length || !params.holding.length) {
      toast("each grid needs at least one value", "error");
      return;
    }
    const jobId = await startJob("/api/walkforward/start", params, "walk-forward");
    if (jobId) {
      state.walkforward = null;
      renderWalkforward();
    }
  });

  $("stop-walkforward").addEventListener("click", async () => {
    const job = currentWalkforwardJob();
    if (!job) return;
    try {
      await api(`/api/jobs/${job.id}/stop`, { method: "POST", body: {} });
      toast("stop requested — the fold in flight will finish and save", "info");
    } catch (error) {
      if (error.message !== "unauthorised") toast(error.message, "error");
    }
  });

  $("rs-start").addEventListener("click", async () => {
    const params = rsParams();
    if (
      !params.k_up.length ||
      !params.k_down.length ||
      !params.holding.length ||
      !params.window.length ||
      !params.seeds.length
    ) {
      toast("each grid needs at least one value", "error");
      return;
    }
    try {
      await api("/api/research/start", { method: "POST", body: params });
      state.researchSpec = null;
      state.research = null;
      toast(
        "campaign started — it keeps running if you close this tab",
        "ok",
        7000
      );
      await refreshResearch();
    } catch (error) {
      if (error.message !== "unauthorised") toast(error.message, "error", 9000);
    }
  });

  $("rs-stop").addEventListener("click", async () => {
    try {
      await api("/api/research/stop", { method: "POST", body: {} });
      toast(
        "stop requested — the fold in flight will finish and save",
        "info",
        7000
      );
      await refreshResearch();
    } catch (error) {
      if (error.message !== "unauthorised") toast(error.message, "error");
    }
  });

  $("rs-leaderboard").addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-spec]");
    if (!row) return;
    state.researchSpec = row.dataset.spec;
    refreshResearch(state.researchSpec).catch((error) => {
      if (error.message !== "unauthorised") toast(error.message, "error");
    });
  });

  window.addEventListener("resize", () => {
    if (!$("panel-research").classList.contains("active")) return;
    renderResearchCharts((state.research && state.research.folds) || []);
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
