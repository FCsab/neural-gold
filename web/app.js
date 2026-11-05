"use strict";

/* crypto-neural web UI.
 *
 * No build step: plain ES2020 loaded by the browser. Three things to do and
 * one place to watch them: Overview (data), Train (one model, end to end),
 * Risk model (the drawdown-risk walk-forward), Activity (jobs and logs). Every
 * expert knob still exists in the API and the CLI; the page only exposes the
 * ones that change the answer.
 */

const state = {
  status: null,
  info: null,
  models: [],
  jobs: [],
  known: new Map(),
  riskChart: null,
  riskLines: null,
  paper: null,
  selectedJob: null,
  chain: null, // { name, steps: [{label, state, note}], current }
  chart: null,
  candles: null,
  btChart: null,
  btModel: null,
  btHold: null,
  rangeBars: 288,
  ws: null,
  online: false,
};

const $ = (id) => document.getElementById(id);

/* ----------------------------------------------------------------- format */
const isNum = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const fmtInt = (v) => (isNum(v) ? Number(v).toLocaleString("en-US") : "–");
const fmtPct = (v, d = 1) => (isNum(v) ? (Number(v) * 100).toFixed(d) + "%" : "–");
const fmtSigned = (v, d = 1) =>
  isNum(v) ? (Number(v) >= 0 ? "+" : "") + (Number(v) * 100).toFixed(d) + "%" : "–";
const fmtMoney = (v) =>
  isNum(v) ? Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 }) : "–";
const fmtDate = (v) => {
  if (v === null || v === undefined) return "–";
  const d = typeof v === "string" ? new Date(v) : new Date(Number(v));
  return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
};
const fmtDay = (v) => (v ? new Date(Number(v)).toISOString().slice(0, 10) : "–");
const esc = (t) =>
  String(t ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function toast(message, kind = "", ttl = 5000) {
  const node = document.createElement("div");
  node.className = "toast " + kind;
  node.textContent = message;
  $("toasts").appendChild(node);
  setTimeout(() => node.remove(), ttl);
}

/* -------------------------------------------------------------------- api */
async function api(path, options = {}) {
  const headers = {};
  let body;
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.body);
  }
  const response = await fetch(path, { method: options.method || "GET", headers, body });
  if (response.status === 403) {
    showLogin("This computer's address is not on the server's allowlist.");
    throw new Error("unauthorised");
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const parsed = await response.json();
      if (parsed && parsed.detail) {
        message = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
      }
    } catch {
      /* body was not JSON */
    }
    throw new Error(message);
  }
  return response.json();
}

const post = (path, body = {}) => api(path, { method: "POST", body });

/* ------------------------------------------------------ allowlist block */
function showLogin(message) {
  $("login").classList.remove("hidden");
  $("login-error").textContent = message || "";
  if (state.ws) state.ws.close();
}

function hideLogin() {
  $("login").classList.add("hidden");
}

/* ------------------------------------------------------------------ views */
function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  if (name === "overview") resizeCharts();
  if (name === "train") renderBacktestChart();
  if (name === "risk") renderRisk();
  if (name === "activity" && state.selectedJob) loadLog(state.selectedJob);
}

const activeView = () => (document.querySelector(".view.active") || {}).id || "";

/* ---------------------------------------------------------------- refresh */
async function refreshAll() {
  const [status, info, jobs, models, paper] = await Promise.all([
    api("/api/status"),
    api("/api/data/info"),
    api("/api/jobs"),
    api("/api/models"),
    api("/api/paper").catch(() => null),
  ]);
  state.status = status;
  state.info = info;
  state.paper = paper;
  state.models = models.models || [];
  state.online = true;
  applyJobs(jobs.jobs || []);
  renderHeader();
  renderData();
  renderTrain();
  renderRisk();
  renderPaper();
  renderHeroRisk();
  loadHero();
}

function applyJobs(jobs) {
  for (const job of jobs) {
    const before = state.known.get(job.id);
    if (before && before !== job.status && !state.chain) {
      if (job.status === "ok") toast(`${jobName(job.kind)}: done`, "ok");
      if (job.status === "failed") toast(`${jobName(job.kind)} failed: ${job.error || "unknown error"}`, "bad", 9000);
    }
    state.known.set(job.id, job.status);
  }
  state.jobs = jobs;
  renderJobBar();
  renderJobs();
  trackTraining();
}

const JOB_NAMES = {
  backfill: "Downloading bars",
  verify: "Checking stored bars",
  "dataset-build": "Building the dataset",
  "label-preview": "Counting labels",
  leakage: "Look-ahead check",
  training: "Training",
  backtest: "Backtesting",
  risk: "Risk test",
  "exchange-info": "Fetching exchange rules",
};
const jobName = (kind) => JOB_NAMES[kind] || kind;
const isLive = (job) => job && (job.status === "running" || job.status === "queued");

/* ----------------------------------------------------------------- header */
function renderHeader() {
  const status = state.status;
  if (!status) return;
  const info = state.info || {};
  const coverage = info.coverage || {};
  const parts = [state.online ? "online" : "offline", `${status.symbol} · ${status.interval}`];
  if (coverage.end_ms) {
    const hours = (Date.now() - coverage.end_ms) / 3_600_000;
    parts.push(hours < 2 ? "data up to date" : `data ${Math.round(hours / 24) || 1} day(s) old`);
  }
  $("status-text").textContent = parts.join(" · ");
  $("status").classList.toggle("bad", !state.online);
}

function renderJobBar() {
  const bar = $("jobbar");
  const live = state.jobs.find(isLive);
  const running = state.jobs.filter(isLive).length;
  $("job-count").textContent = String(running);
  $("job-count").classList.toggle("hidden", running === 0);

  if (!live) {
    bar.classList.add("hidden");
    return;
  }
  bar.classList.remove("hidden");
  const chain = state.chain;
  const prefix = chain ? `Step ${chain.current + 1} of ${chain.steps.length} · ` : "";
  const message = live.status === "queued" ? "waiting for the previous job" : live.message || "working…";
  $("jobbar-text").textContent = `${prefix}${jobName(live.kind)} — ${message}`;
  const fill = $("jobbar-fill");
  if (isNum(live.progress)) {
    fill.style.width = Math.round(live.progress * 100) + "%";
    fill.classList.remove("indeterminate");
  } else {
    fill.style.width = "30%";
    fill.classList.add("indeterminate");
  }
  $("jobbar-stop").dataset.job = live.id;
  $("jobbar-stop").textContent = live.kind === "training" ? "Stop training (keeps the best weights)" : "Stop";
}

/* ------------------------------------------------------------- job chains */
async function waitForJob(jobId) {
  for (;;) {
    const job = await api(`/api/jobs/${jobId}`);
    if (!isLive(job)) return job;
    await sleep(1500);
  }
}

/** Run jobs one after another. A step returns a job id; a failed step ends the chain. */
async function runChain(name, steps) {
  if (state.chain) {
    toast("Something is already running. Wait for it, or stop it first.", "bad");
    return false;
  }
  state.chain = {
    name,
    current: 0,
    steps: steps.map((step) => ({ label: step.label, state: "waiting", note: "" })),
  };
  renderSteps();
  let ok = true;
  try {
    for (let index = 0; index < steps.length; index += 1) {
      state.chain.current = index;
      const view = state.chain.steps[index];
      view.state = "running";
      renderSteps();
      const { job_id: jobId } = await steps[index].start();
      state.selectedJob = jobId;
      await refreshAll().catch(() => {});
      const job = await waitForJob(jobId);
      const accepted = job.status === "ok" || (job.status === "stopped" && steps[index].stoppable);
      const problem = accepted && steps[index].check ? steps[index].check(job) : null;
      if (!accepted || problem) {
        view.state = "failed";
        view.note = problem || (job.status === "stopped" ? "stopped" : job.error || "failed");
        ok = false;
        break;
      }
      view.state = "done";
      view.note = steps[index].summary ? steps[index].summary(job) : "";
      renderSteps();
    }
  } catch (error) {
    const view = state.chain.steps[state.chain.current];
    view.state = "failed";
    view.note = error.message;
    ok = false;
  }
  renderSteps();
  const finished = state.chain;
  state.chain = null;
  await refreshAll().catch(() => {});
  toast(ok ? `${finished.name}: done` : `${finished.name}: stopped at “${finished.steps[finished.current].label}”`, ok ? "ok" : "bad", 8000);
  return ok;
}

function renderSteps() {
  // Only the Train view has a step list; other chains show in the job bar.
  const chain = state.chain;
  if (!chain || chain.name !== "Train and test") return;
  $("steps").innerHTML = chain.steps
    .map(
      (step) =>
        `<li class="${step.state}"><span>${esc(step.label)}</span>` +
        (step.note ? `<span class="note">${esc(step.note)}</span>` : "") +
        "</li>"
    )
    .join("");
  $("run-train").disabled = true;
}

/* ------------------------------------------------------------------- data */
function renderData() {
  const info = state.info;
  if (!info) return;
  const coverage = info.coverage || {};
  const years = coverage.start_ms ? (coverage.end_ms - coverage.start_ms) / (365.25 * 86_400_000) : 0;
  const interval = (state.status && state.status.interval) || "";

  $("data-summary").textContent = coverage.rows
    ? `${years.toFixed(1)} years of ${interval} bars, from ${fmtDay(coverage.start_ms)} to ${fmtDate(coverage.end_ms)}.`
    : "No price history yet. Press “Download latest bars” to fetch two years.";

  const facts = [];
  if (coverage.rows) facts.push(["ok", `${fmtInt(coverage.rows)} bars stored`]);
  const verify = info.verify;
  if (verify) {
    facts.push(
      verify.ok
        ? ["ok", "No broken candles or gaps"]
        : ["bad", `Problems found: ${fmtInt(verify.gap_count)} gap(s). Download again to fill them.`]
    );
  } else if (coverage.rows) {
    facts.push(["", "Not checked yet"]);
  }
  const leakage = info.leakage;
  if (leakage) {
    facts.push(
      leakage.passed
        ? ["ok", "Look-ahead check passed: features cannot see the future"]
        : ["bad", "Look-ahead check FAILED: do not trust any model until this passes"]
    );
  }
  $("data-facts").innerHTML = facts.map(([tone, text]) => `<li class="${tone}">${esc(text)}</li>`).join("");
}

/* Series colours for the equity charts, validated for colour-blind separation
 * on the dark card surface (#141a23): model = amber, holding = blue (also
 * dashed), trend rule = magenta. Green and red are kept for up/down only. */
const SERIES = { model: "#c98500", hold: "#3987e5", trend: "#d55181" };

async function loadHero() {
  try {
    const data = await api("/api/data/bars?limit=289");
    const bars = data.bars || [];
    if (!bars.length) return;
    const last = bars[bars.length - 1];
    const first = bars[0];
    const change = last.close / first.close - 1;
    $("hero-price").textContent = last.close.toLocaleString("en-US", { maximumFractionDigits: 2 });
    const up = change >= 0;
    $("hero-change").className = "change " + (up ? "pos" : "neg");
    $("hero-change").textContent = `${up ? "▲" : "▼"} ${fmtSigned(change, 2)} in 24 hours`;
    $("hero-high").textContent = Math.max(...bars.map((b) => b.high)).toLocaleString("en-US", { maximumFractionDigits: 0 });
    $("hero-low").textContent = Math.min(...bars.map((b) => b.low)).toLocaleString("en-US", { maximumFractionDigits: 0 });
    $("hero-time").textContent = fmtDate(last.time * 1000).slice(5);
  } catch {
    /* the status line says offline */
  }
}

function renderHeroRisk() {
  const report = state.info && state.info.risk;
  const box = $("hero-risk");
  if (!report || !report.today) {
    box.className = "hero-risk";
    $("hero-risk-action").textContent = "not run";
    $("hero-risk-note").textContent = "Run the test on the Risk model page.";
    return;
  }
  const today = report.today;
  const words = { long: "Stay long", flat: "Step aside", keep: "Keep position" };
  box.className = `hero-risk ${today.action || ""}`;
  $("hero-risk-action").textContent = words[today.action] || today.action;
  $("hero-risk-note").textContent =
    `${fmtPct(today.risk, 0)} risk of a ${fmtPct(report.params.drawdown, 0)} fall` +
    (report.status === "draft" ? " · draft model, not advice" : "");
}

function labelParams() {
  return {
    k_up: Number($("k-up").value) || 2.0,
    k_down: Number($("k-down").value) || 1.5,
    max_holding_bars: parseInt($("holding").value, 10) || 12,
  };
}

function updateData() {
  const coverage = (state.info && state.info.coverage) || {};
  const years = coverage.start_ms ? (coverage.end_ms - coverage.start_ms) / (365.25 * 86_400_000) : 2;
  // Ask for what is already held so nothing is trimmed; only the gap is fetched.
  const wanted = Math.min(10, Math.max(2, Math.ceil(years * 10) / 10));
  runChain("Download", [
    { label: "Download bars", start: () => post("/api/data/backfill", { years: wanted }) },
  ]).then(() => loadBars());
}

function checkData() {
  runChain("Check the data", [
    { label: "Check stored bars", start: () => post("/api/data/verify") },
    {
      label: "Look-ahead check",
      start: () => post("/api/leakage/run", { ...labelParams(), samples: 10 }),
      check: (job) => (job.result && job.result.passed === false ? "a feature can see the future" : null),
    },
  ]);
}

/* ------------------------------------------------------------------ chart */
const CHART_OPTIONS = {
  layout: { background: { color: "transparent" }, textColor: "#8b97a8", fontSize: 11 },
  grid: { vertLines: { visible: false }, horzLines: { color: "#1a212b" } },
  rightPriceScale: { borderVisible: false },
  // The default 0.5 px minimum spacing cannot fit eight years of daily points
  // into a card, so fitContent would silently drop the oldest years.
  timeScale: { borderVisible: false, timeVisible: true, minBarSpacing: 0.05 },
  crosshair: {
    vertLine: { color: "#4a5567", labelBackgroundColor: "#2a3342" },
    horzLine: { color: "#4a5567", labelBackgroundColor: "#2a3342" },
  },
  handleScroll: { mouseWheel: false },
};

function makeChart(container) {
  const chart = LightweightCharts.createChart(container, {
    ...CHART_OPTIONS,
    width: container.clientWidth,
    height: container.clientHeight,
  });
  // A chart built while its view is hidden starts at zero width; fit the data
  // again the first time it gets a real size, or the left of the range is lost.
  let sized = container.clientWidth > 0;
  new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    if (!sized && container.clientWidth > 0) {
      sized = true;
      chart.timeScale().fitContent();
    }
  }).observe(container);
  return chart;
}

function resizeCharts() {
  if (state.chart) state.chart.timeScale().fitContent();
}

async function loadBars() {
  if (typeof LightweightCharts === "undefined") return;
  if (!state.chart) {
    state.chart = makeChart($("chart"));
    state.candles = state.chart.addCandlestickSeries({
      upColor: "#4fbf86",
      downColor: "#e06666",
      borderVisible: false,
      wickUpColor: "#4fbf86",
      wickDownColor: "#e06666",
    });
    state.chart.subscribeCrosshairMove((param) => {
      const bar = param.time && param.seriesData.get(state.candles);
      const node = $("chart-readout");
      if (!bar) {
        node.textContent = "Hover the chart to read a candle.";
        return;
      }
      const move = bar.close / bar.open - 1;
      const f = (v) => v.toLocaleString("en-US", { maximumFractionDigits: 2 });
      node.innerHTML =
        `<span>${esc(fmtDate(param.time * 1000))}</span>` +
        `<span>open <b>${f(bar.open)}</b></span><span>high <b>${f(bar.high)}</b></span>` +
        `<span>low <b>${f(bar.low)}</b></span><span>close <b>${f(bar.close)}</b></span>` +
        `<span class="${move >= 0 ? "pos" : "neg"}">${move >= 0 ? "▲" : "▼"} ${fmtSigned(move, 2)}</span>`;
    });
  }
  try {
    const data = await api(`/api/data/bars?limit=${state.rangeBars}`);
    state.candles.setData(data.bars || []);
    state.chart.timeScale().fitContent();
  } catch {
    /* shown by the status line */
  }
}

/* ------------------------------------------------------------------ train */
function trainAndTest() {
  const labels = labelParams();
  const minutes = $("train-minutes").value;
  runChain("Train and test", [
    {
      label: "Build the dataset",
      start: () => post("/api/dataset/build", labels),
      summary: (job) => (job.result ? `${fmtInt(job.result.rows)} labelled bars` : ""),
    },
    {
      label: "Look-ahead check",
      start: () => post("/api/leakage/run", { ...labels, samples: 10 }),
      check: (job) => (job.result && job.result.passed === false ? "failed: a feature can see the future, so training was skipped" : null),
      summary: () => "passed",
    },
    {
      label: "Train",
      stoppable: true,
      start: () => post("/api/train/start", { minutes: minutes === "" ? null : Number(minutes), epochs: 0, patience: 0 }),
      summary: (job) => (job.result ? `${fmtInt(job.result.epochs_done)} epochs` : ""),
    },
    {
      label: "Backtest on unseen data",
      start: () => post("/api/backtest/run", {}),
    },
  ]).then(() => {
    $("run-train").disabled = false;
  });
}

function accuracyVerdict(metrics) {
  if (!metrics || !isNum(metrics.balanced_accuracy)) return null;
  const score = Number(metrics.balanced_accuracy);
  const baseline = isNum(metrics.baseline_balanced_accuracy) ? Number(metrics.baseline_balanced_accuracy) : 1 / 3;
  const gap = (score - baseline) * 100;
  if (gap >= 5) return { tone: "ok", text: `It predicts better than guessing (${fmtPct(score)} vs ${fmtPct(baseline)}).` };
  if (gap >= 2) return { tone: "", text: `It predicts slightly better than guessing (${fmtPct(score)} vs ${fmtPct(baseline)}), which rarely survives fees.` };
  return { tone: "bad", text: `It predicts no better than guessing (${fmtPct(score)} vs ${fmtPct(baseline)}).` };
}

function renderTrain() {
  if (!state.chain) $("run-train").disabled = Boolean(state.jobs.find(isLive));
  const training = state.status && state.status.training;
  if (training && !training.available) {
    $("run-train").disabled = true;
    $("run-train").title = training.reason || "PyTorch is not installed";
  }

  const report = state.info && state.info.backtest;
  const model = (state.models || []).find((m) => m.metrics && Object.keys(m.metrics).length);
  const result = $("result");

  if (!report && !model) {
    result.innerHTML = '<p class="muted">No model yet. Press “Train and test”.</p>';
  } else {
    const parts = [];
    const metrics = (report && report.metrics) || {};
    if (report && report.verdict && report.verdict.text) {
      parts.push(`<div class="verdict ${esc(report.verdict.tone || "")}">${esc(report.verdict.text)}</div>`);
    }
    const accuracy = accuracyVerdict(model && model.metrics);
    if (accuracy) parts.push(`<p class="sentence ${accuracy.tone}">${esc(accuracy.text)}</p>`);
    if (report) {
      const tone = (v) => (Number(v) >= 0 ? "pos" : "neg");
      parts.push(
        '<div class="tiles">' +
          tile("Model", fmtSigned(metrics.total_return), tone(metrics.total_return)) +
          tile("Just holding", fmtSigned(metrics.buy_hold_return), tone(metrics.buy_hold_return)) +
          tile("Trades", fmtInt(metrics.trades)) +
          tile("Worst drop", fmtPct(metrics.max_drawdown), "neg") +
          "</div>"
      );
      parts.push(
        `<p class="hint">Tested ${fmtDay(metrics.start_ms)} to ${fmtDay(metrics.end_ms)} on data the model never saw. ` +
          `Fees ${esc(metrics.fees_bps)} bps and slippage ${esc(metrics.slippage_bps)} bps are charged on every buy and sell.` +
          (model ? ` Model: ${esc(model.name)}.` : "") +
          "</p>"
      );
    }
    result.innerHTML = parts.join("");
  }
  $("bt-chart").classList.toggle("hidden", !report);
  $("bt-legend").classList.toggle("hidden", !report);
  if (activeView() === "view-train") renderBacktestChart();
  renderModels();
}

/* ---------------------------------------------------------- live training */
const GUESS_COLOR = "#8592a3";

/** The newest training job, live or finished; its curves stay on screen. */
function latestTraining() {
  return state.jobs
    .filter((job) => job.kind === "training")
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))[0];
}

function trackTraining() {
  const job = latestTraining();
  if (!job) return;
  const changed = !state.live || state.live.id !== job.id || state.live.status !== job.status;
  if (!changed) return;
  state.live = { id: job.id, status: job.status, job: (state.live && state.live.id === job.id && state.live.job) || null };
  if (state.liveTimer) clearInterval(state.liveTimer);
  state.liveTimer = null;
  loadLive();
  if (isLive(job)) state.liveTimer = setInterval(loadLive, 2000);
}

async function loadLive() {
  const live = state.live;
  if (!live) return;
  try {
    const job = await api(`/api/jobs/${live.id}`);
    if (!state.live || state.live.id !== job.id) return;
    state.live.job = job;
    if (!isLive(job) && state.liveTimer) {
      clearInterval(state.liveTimer);
      state.liveTimer = null;
    }
    renderLive();
  } catch {
    /* the status line shows a dropped connection */
  }
}

function epochChart(container) {
  const chart = makeChart(container);
  chart.applyOptions({
    localization: { timeFormatter: (t) => `epoch ${t}` },
    timeScale: { timeVisible: false, tickMarkFormatter: (t) => String(t) },
  });
  return chart;
}

function liveVerdict(job, points) {
  const last = points[points.length - 1];
  const best = Number(last.best_epoch) || 0;
  const since = Number(last.epoch) - best;
  const bestPoint = points.find((p) => Number(p.epoch) === best) || last;
  const minutes = isNum(last.elapsed_s) ? Math.round(Number(last.elapsed_s) / 60) : null;
  const where = isLive(job) ? "Training" : "Finished";
  const head = `${where}: epoch ${fmtInt(last.epoch)}${minutes !== null ? `, ${minutes} min in` : ""}.`;
  const score = Number(bestPoint.val_balanced_accuracy);
  const guess = Number(bestPoint.baseline_balanced_accuracy);
  const gap = (score - guess) * 100;
  let tone = "bad";
  let skill = `At its best epoch (${fmtInt(best)}) it scores ${fmtPct(score)} on held-out data, no better than always guessing the same answer (${fmtPct(guess)}).`;
  if (gap >= 5) {
    tone = "ok";
    skill = `At its best epoch (${fmtInt(best)}) it scores ${fmtPct(score)} on held-out data, against ${fmtPct(guess)} for always guessing the same answer.`;
  } else if (gap >= 2) {
    tone = "";
    skill = `At its best epoch (${fmtInt(best)}) it scores ${fmtPct(score)} on held-out data, slightly above ${fmtPct(guess)} for always guessing; that rarely survives fees.`;
  }
  let trend = "";
  if (since >= 10) {
    trend = ` The held-out error has not improved for ${fmtInt(since)} epochs: from there on it was mostly memorising.`;
    if (isLive(job)) trend += " Stopping now keeps the best weights.";
  } else if (best && since === 0) {
    trend = " The held-out error just reached a new low.";
  }
  return { tone, text: `${head} ${skill}${trend}` };
}

function renderLive() {
  const job = state.live && state.live.job;
  const points = ((job && job.series) || []).filter((p) => isNum(p.epoch));
  const card = $("live-card");
  card.classList.toggle("hidden", !job || (!points.length && !isLive(job)));
  if (!job) return;
  const states = { ok: "finished", stopped: "stopped (best weights kept)", failed: "failed" };
  $("live-state").textContent = `· ${isLive(job) ? "running" : states[job.status] || job.status}`;
  if (!points.length) {
    $("live-verdict").className = "sentence";
    $("live-verdict").textContent = "Waiting for the first epoch to finish…";
    return;
  }
  const verdict = liveVerdict(job, points);
  $("live-verdict").className = `sentence ${verdict.tone}`;
  $("live-verdict").textContent = verdict.text;
  if (typeof LightweightCharts === "undefined") return;

  if (!state.lossChart) {
    state.lossChart = epochChart($("live-loss"));
    state.lossTrain = state.lossChart.addLineSeries({ color: SERIES.hold, lineWidth: 2, lineStyle: 2, title: "training" });
    state.lossVal = state.lossChart.addLineSeries({ color: SERIES.model, lineWidth: 2, title: "held-out" });
    state.scoreChart = epochChart($("live-score"));
    state.scoreGuess = state.scoreChart.addLineSeries({
      color: GUESS_COLOR, lineWidth: 1, lineStyle: 2, title: "guessing",
      crosshairMarkerVisible: false, lastValueVisible: false,
    });
    state.scoreModel = state.scoreChart.addLineSeries({
      color: SERIES.model, lineWidth: 2, title: "model",
      priceFormat: { type: "custom", formatter: (v) => fmtPct(v) },
    });
    state.scoreGuess.applyOptions({ priceFormat: { type: "custom", formatter: (v) => fmtPct(v) } });
  }
  const at = (key) => points.filter((p) => isNum(p[key])).map((p) => ({ time: Number(p.epoch), value: Number(p[key]) }));
  state.lossTrain.setData(at("train_loss"));
  state.lossVal.setData(at("val_loss"));
  state.scoreModel.setData(at("val_balanced_accuracy"));
  state.scoreGuess.setData(at("baseline_balanced_accuracy"));
  state.lossChart.timeScale().fitContent();
  state.scoreChart.timeScale().fitContent();
}

function tile(label, value, tone = "") {
  return `<div class="tile"><span>${esc(label)}</span><strong class="${tone}">${esc(value)}</strong></div>`;
}

function renderBacktestChart() {
  const report = state.info && state.info.backtest;
  if (!report || typeof LightweightCharts === "undefined") return;
  if (!state.btChart) {
    state.btChart = makeChart($("bt-chart"));
    state.btModel = state.btChart.addLineSeries({ color: SERIES.model, lineWidth: 2 });
    state.btHold = state.btChart.addLineSeries({ color: SERIES.hold, lineWidth: 2, lineStyle: 2 });
  }
  const metrics = report.metrics || {};
  const start = metrics.initial_equity || 10000;
  const seen = new Set();
  const points = [];
  const first = metrics.start_ms ? Math.floor(metrics.start_ms / 1000) : null;
  const curve = (report.equity_curve || []).map((p) => ({ time: Math.floor(p.time), value: p.equity }));
  if (first !== null) curve.unshift({ time: first, value: start });
  for (const point of curve.sort((a, b) => a.time - b.time)) {
    if (seen.has(point.time)) continue;
    seen.add(point.time);
    points.push(point);
  }
  if (!points.length) return;
  state.btModel.setData(points);
  state.btHold.setData([
    { time: points[0].time, value: start },
    { time: points[points.length - 1].time, value: start * (1 + (metrics.buy_hold_return || 0)) },
  ]);
  state.btChart.timeScale().fitContent();
}

function renderModels() {
  const models = state.models || [];
  $("model-count").textContent = models.length ? `(${models.length})` : "";
  $("models").innerHTML = models.length
    ? "<thead><tr><th>model</th><th>verdict</th><th class='num'>score</th><th>when</th></tr></thead><tbody>" +
      models
        .map((m) => {
          const metrics = m.metrics || {};
          const verdict = m.status === "candidate" ? "beat guessing" : "no better than guessing";
          return `<tr><td>${esc(m.name)}</td><td>${verdict}</td><td class="num">${fmtPct(metrics.balanced_accuracy)}</td><td>${fmtDate(m.created_at)}</td></tr>`;
        })
        .join("") +
      "</tbody>"
    : "<tbody><tr><td class='muted'>none yet</td></tr></tbody>";
}

/* ------------------------------------------------------------- risk model */
function runRisk() {
  runChain("Risk test", [
    {
      label: "Risk test",
      start: () =>
        post("/api/risk/run", {
          drawdown: Number($("risk-drawdown").value),
          horizon_days: Number($("risk-horizon").value),
        }),
    },
  ]);
}

function renderRisk() {
  if (!state.chain) $("risk-run").disabled = Boolean(state.jobs.find(isLive));
  const report = state.info && state.info.risk;
  const box = $("risk-result");
  for (const id of ["risk-chart", "risk-legend", "risk-readout"]) $(id).classList.toggle("hidden", !report);
  if (!report) {
    box.innerHTML = '<p class="muted">Not run yet. Press “Run the test”.</p>';
    $("risk-folds").innerHTML = "";
    $("risk-fold-count").textContent = "";
    return;
  }
  const s = report.strategy || {};
  const h = report.hold || {};
  const t = report.trend || {};
  const today = report.today || {};
  const period = report.period || {};
  const params = report.params || {};
  const verdict = report.verdict || {};
  const tone = (a, b) => (Number(a) >= Number(b) ? "pos" : "neg");

  box.innerHTML =
    `<div class="verdict ${esc(verdict.tone || "")}">${esc(verdict.text || "")}</div>` +
    `<div class="today ${esc(today.action || "")}"><strong>Next bar (${esc(today.for_bar || today.as_of || "")}):</strong>${esc(today.text || "")}` +
    (today.trend ? `<br><span class="muted">Trend rule: ${esc(today.trend.text)}</span>` : "") +
    (report.status === "draft"
      ? '<br><span class="muted">This rule did not beat holding in the test above, so treat the reading as information, not advice.</span>'
      : "") +
    "</div>" +
    '<div class="tiles">' +
    tile("Risk model, per year", fmtSigned(s.cagr), tone(s.cagr, h.cagr)) +
    tile("Just holding, per year", fmtSigned(h.cagr)) +
    tile("Worst drop (model / holding)", `${fmtPct(s.max_drawdown, 0)} / ${fmtPct(h.max_drawdown, 0)}`) +
    tile("Beaten by random exits", isNum(s.placebo_p) ? fmtPct(s.placebo_p, 0) : "–") +
    "</div>" +
    skillLine(report) +
    trendLine(s, h, t) +
    `<p class="hint">Tested ${esc(period.start)} to ${esc(period.end)} (${esc(period.years)} years, ${esc(period.folds)} blocks). ` +
    `Out of the market ${fmtPct(1 - (s.exposure || 0), 0)} of the time, ${esc(s.switches_per_year)} switches a year. ` +
    `Sharpe ${Number(s.sharpe).toFixed(2)} against ${Number(h.sharpe).toFixed(2)} for holding; the trend rule made ${fmtSigned(t.cagr)} a year. ` +
    `“Beaten by random exits” is how often the same exits moved to random dates did as well; under 5% means the timing mattered. ` +
    `Costs: ${esc(params.fee_bps)} bps fee + ${esc(params.slippage_bps)} bps slippage per side.</p>`;

  renderRiskChart(report);

  const folds = report.folds || [];
  $("risk-fold-count").textContent = folds.length ? `(${folds.length})` : "";
  $("risk-folds").innerHTML =
    "<thead><tr><th>block</th><th>rule used</th><th class='num'>risk model</th><th class='num'>holding</th><th class='num'>in the market</th></tr></thead><tbody>" +
    folds
      .map(
        (f) =>
          `<tr><td>${fmtDay(f.start_ms)} – ${fmtDay(f.end_ms)}</td>` +
          `<td>${esc(ruleName(f.rule))}</td>` +
          `<td class="num ${tone(f.return, f.hold_return)}">${fmtSigned(f.return)}</td>` +
          `<td class="num">${fmtSigned(f.hold_return)}</td><td class="num">${fmtPct(f.exposure, 0)}</td></tr>`
      )
      .join("") +
    "</tbody>";
}

/** The predictions log scored against what followed. Verdict first. */
function renderPaper() {
  const paper = state.paper;
  const box = $("paper-result");
  const sched = (state.status && state.status.readings) || {};
  $("paper-schedule").innerHTML = sched.scheduled
    ? `Readings are taken automatically just after every bar opens. Next: ${esc(fmtDate(sched.next_at))}.`
    : "Readings are only taken when the test is run by hand, so most will be late. To take one " +
      "on time after every bar, set <code>record_readings: true</code> under <code>server:</code> " +
      "in config.yaml and restart.";
  if (!paper || !paper.readings) {
    box.innerHTML = `<p class="muted">${esc((paper && paper.verdict) || "No readings yet.")}</p>`;
    $("paper-rows").innerHTML = "";
    $("paper-count").textContent = "";
    return;
  }
  const r = paper.risk_rule || {};
  const t = paper.trend_rule || {};
  const h = paper.hold || {};
  const forecast = isNum(paper.brier)
    ? `<p class="sentence">Of ${fmtInt(paper.decided)} readings whose outcome is known, ` +
      `${fmtInt(paper.falls)} were followed by the fall. Forecast error ${Number(paper.brier).toFixed(4)} ` +
      `against ${Number(paper.brier_base_rate).toFixed(4)} for always guessing the rate that actually ` +
      `happened (lower is better)${isNum(paper.auc) ? `, score ${Number(paper.auc).toFixed(2)} where 0.50 is a coin toss` : ""}.</p>`
    : "";
  box.innerHTML =
    `<div class="verdict ${paper.days >= 90 ? "warn" : ""}">${esc(paper.verdict)}</div>` +
    '<div class="tiles">' +
    tile("Risk rule", fmtSigned(r.total_return, 2), Number(r.total_return) >= Number(h.total_return) ? "pos" : "neg") +
    tile("Trend rule", fmtSigned(t.total_return, 2), Number(t.total_return) >= Number(h.total_return) ? "pos" : "neg") +
    tile("Just holding", fmtSigned(h.total_return, 2)) +
    tile("Readings on time", `${fmtInt(paper.on_time)} of ${fmtInt(paper.readings)}`) +
    "</div>" +
    forecast +
    `<p class="hint">${esc(paper.first_bar || "")} to ${esc(paper.last_bar || "")}. Between readings ` +
    `the position is held, as someone following the log would hold it on days nothing ran.</p>`;
  const rows = (paper.rows || []).slice().reverse();
  $("paper-count").textContent = `(${rows.length})`;
  const fell = (v) => (v === null || v === undefined ? "pending" : v ? "yes" : "no");
  $("paper-rows").innerHTML =
    "<thead><tr><th>for bar</th><th>recorded</th><th class='num'>risk</th><th>risk rule</th><th>trend rule</th><th>fell?</th></tr></thead><tbody>" +
    rows
      .map(
        (row) =>
          `<tr><td>${esc(row.for_bar)}</td><td class="${row.on_time ? "" : "neg"}">${esc(row.recorded_at)}${row.on_time ? "" : " (late)"}</td>` +
          `<td class="num">${fmtPct(row.risk, 0)}</td><td>${esc(row.action)}</td><td>${esc(row.trend_action || "–")}</td>` +
          `<td>${fell(row.fell)}</td></tr>`
      )
      .join("") +
    "</tbody>";
}

/** Prediction skill in words. AUC 0.5 is a coin toss; 1.0 is perfect. */
function skillLine(report) {
  const prediction = report.prediction || {};
  const auc = Number(prediction.auc);
  if (!isNum(auc)) return "";
  const params = report.params || {};
  const what = `a ${fmtPct(params.drawdown, 0)} fall`;
  const words =
    auc >= 0.6
      ? `It does predict ${what} better than chance`
      : auc >= 0.55
      ? `It predicts ${what} only slightly better than chance`
      : `It cannot predict ${what} better than chance`;
  return (
    `<p class="sentence">${esc(words)} (score ${auc.toFixed(2)}, where 0.50 is a coin toss), ` +
    `using the ${esc(params.model === "ensemble" ? "neural network and trees together" : params.model)} ` +
    `on ${esc(params.data_symbol || "")} history.</p>`
  );
}

/** Say so when the no-model trend rule beat both on risk, and whether its
 * timing passed the same random-dates test the model has to pass. */
function trendLine(strategy, hold, trend) {
  if (!isNum(trend.sharpe) || trend.sharpe <= Math.max(strategy.sharpe, hold.sharpe)) return "";
  const p = Number(trend.placebo_p);
  const luck = !isNum(p)
    ? ""
    : p < 0.05
    ? ` Random exits at the same frequency did as well only ${fmtPct(p, 0)} of the time, so its timing mattered.`
    : ` But random exits at the same frequency did as well ${fmtPct(p, 0)} of the time, so its timing is not distinguishable from luck.`;
  return (
    `<div class="verdict warn">The trend rule, which uses no model at all (long only while the price is above ` +
    `its 200-day average), did better on risk than both: ${fmtSigned(trend.cagr)} a year with a worst drop of ` +
    `${fmtPct(trend.max_drawdown, 0)}, Sharpe ${Number(trend.sharpe).toFixed(2)}.${esc(luck)}</div>`
  );
}

/** "q0.925" -> "step aside in the riskiest 7.5% of readings" */
function ruleName(rule) {
  if (rule === "hold") return "just hold";
  const q = Number(String(rule).slice(1));
  return isNum(q) ? `step aside in the riskiest ${+((1 - q) * 100).toFixed(1)}% of readings` : String(rule);
}

function renderRiskChart(report) {
  if (typeof LightweightCharts === "undefined" || activeView() !== "view-risk") return;
  if (!state.riskChart) {
    state.riskChart = makeChart($("risk-chart"));
    // Eight years of compounding: equal distances mean equal percentage moves.
    state.riskChart.applyOptions({ rightPriceScale: { mode: LightweightCharts.PriceScaleMode.Logarithmic } });
    const line = (color, style = 0) =>
      state.riskChart.addLineSeries({
        color,
        lineWidth: 2,
        lineStyle: style,
        priceLineVisible: false,
        priceFormat: { type: "price", precision: 0, minMove: 1 },
      });
    state.riskLines = {
      trend: line(SERIES.trend),
      hold: line(SERIES.hold, 2),
      strategy: line(SERIES.model),
    };
    state.riskChart.subscribeCrosshairMove((param) => {
      const node = $("risk-readout");
      if (!param.time) {
        node.textContent = "Hover the chart to compare the three on any day.";
        return;
      }
      const value = (key) => {
        const point = param.seriesData.get(state.riskLines[key]);
        return point ? fmtMoney(point.value) : "–";
      };
      node.innerHTML =
        `<span>${esc(fmtDay(param.time * 1000))} · 10,000 became</span>` +
        `<span><i class="s-model"></i>model <b>${value("strategy")}</b></span>` +
        `<span><i class="s-hold"></i>holding <b>${value("hold")}</b></span>` +
        `<span><i class="s-trend"></i>trend <b>${value("trend")}</b></span>`;
    });
  }
  const equity = report.equity || {};
  const times = equity.time || [];
  for (const key of ["strategy", "hold", "trend"]) {
    const values = equity[key] || [];
    state.riskLines[key].setData(times.map((time, i) => ({ time, value: values[i] * 10000 })));
  }
  state.riskChart.timeScale().fitContent();
}

/* --------------------------------------------------------------- activity */
function renderJobs() {
  const host = $("jobs");
  if (!state.jobs.length) {
    host.innerHTML = '<p class="muted">Nothing has run yet.</p>';
    return;
  }
  host.innerHTML = state.jobs
    .map((job) => {
      const when = new Date(job.created_at * 1000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
      const selected = job.id === state.selectedJob ? " selected" : "";
      return `<button class="job${selected}" data-job="${job.id}"><span class="dot ${job.status}"></span>${esc(jobName(job.kind))}<span class="when">${when}</span></button>`;
    })
    .join("");
}

async function loadLog(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    const text = (job.log || []).join("\n") || "(no output)";
    const node = $("log");
    if (node.textContent !== text) {
      node.textContent = text;
      node.scrollTop = node.scrollHeight;
    }
  } catch (error) {
    if (error.message !== "unauthorised") $("log").textContent = error.message;
  }
}

/* -------------------------------------------------------------- websocket */
function connectWs() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${location.host}/api/ws`);
  state.ws = socket;
  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "jobs") {
        const before = state.jobs.filter(isLive).length;
        applyJobs(message.jobs || []);
        // A job just finished: its report is new on disk.
        if (state.jobs.filter(isLive).length < before && !state.chain) refreshAll().catch(() => {});
      }
    } catch {
      /* ignore a malformed frame */
    }
  };
  socket.onclose = (event) => {
    state.ws = null;
    if (event.code === 4403) {
      return showLogin("This computer's address is not on the server's allowlist.");
    }
    setTimeout(connectWs, 2500);
  };
}

/* ------------------------------------------------------------------- boot */
function bindEvents() {
  document.querySelectorAll("#nav button").forEach((button) =>
    button.addEventListener("click", () => showView(button.dataset.view))
  );
  document.querySelectorAll("#range button").forEach((button) =>
    button.addEventListener("click", () => {
      document.querySelectorAll("#range button").forEach((b) => b.classList.toggle("active", b === button));
      state.rangeBars = Number(button.dataset.bars);
      loadBars();
    })
  );
  $("update-data").addEventListener("click", updateData);
  $("check-data").addEventListener("click", checkData);
  $("run-train").addEventListener("click", trainAndTest);
  $("risk-run").addEventListener("click", runRisk);
  $("hero-risk").addEventListener("click", () => showView("risk"));
  $("jobbar-stop").addEventListener("click", async (event) => {
    const jobId = event.currentTarget.dataset.job;
    if (!jobId) return;
    try {
      await post(`/api/jobs/${jobId}/stop`);
    } catch (error) {
      if (error.message !== "unauthorised") toast(error.message, "bad");
    }
  });
  $("jobs").addEventListener("click", (event) => {
    const row = event.target.closest("[data-job]");
    if (!row) return;
    state.selectedJob = row.dataset.job;
    renderJobs();
    loadLog(state.selectedJob);
  });
}

async function start() {
  await refreshAll();
  hideLogin();
  if (!state.ws) connectWs();
  loadBars();
}

function boot() {
  bindEvents();
  setInterval(() => {
    refreshAll().catch(() => {
      state.online = false;
      renderHeader();
    });
  }, 15000);
  setInterval(() => {
    if (activeView() === "view-activity" && state.selectedJob) {
      const job = state.jobs.find((j) => j.id === state.selectedJob);
      if (isLive(job)) loadLog(job.id);
    }
  }, 4000);

  // Old builds kept an access token here; it is no longer used.
  localStorage.removeItem("cryptoneural.token");
  start().catch((error) => {
    if (error.message !== "unauthorised") toast(error.message, "bad");
  });
}

document.addEventListener("DOMContentLoaded", boot);
