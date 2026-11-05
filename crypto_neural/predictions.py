"""Predictions log: readings written down *before* their outcome exists.

The walk-forward says how a rule would have done. The plan's gate asks for
something it cannot provide: the same rule run forward, in time, where every
reading is stamped with the wall-clock moment it was made. Only that record
shows what a rerun cannot reconstruct — readings that came late, days the job
did not run, a model that drifted after its last test.

Two rules make the log worth keeping:

* **First reading wins.** A reading is keyed by the bar it is for and the
  fingerprint of the configuration that made it. Running the job again for
  the same bar does not replace it; otherwise re-running until the answer
  looks right would be possible, and invisible.
* **Late readings are executed late.** A reading recorded more than
  ``GRACE_MS`` after its bar opened is filled at the first bar that opens
  after it was recorded, not at the bar it was for. The walk-forward fills
  at the open of the bar; this is where paper and simulation can disagree.

Scoring is pure pandas: :func:`score_paper` takes the log and the bars and
returns the paper record. Nothing here places, sizes, or routes an order.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import label_drawdown
from .risk import _auc, net_returns, performance

# A job that starts a few minutes after the bar opened still trades close to
# that open. Ten minutes on a 4h bar is generous without being a loophole.
GRACE_MS = 10 * 60 * 1000

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at_ms INTEGER NOT NULL,
    for_bar_ms     INTEGER NOT NULL,
    symbol         TEXT NOT NULL,
    interval       TEXT NOT NULL,
    source         TEXT NOT NULL,
    config_hash    TEXT NOT NULL,
    model          TEXT NOT NULL,
    drawdown       REAL NOT NULL,
    horizon_bars   INTEGER NOT NULL,
    risk           REAL,
    rule           TEXT NOT NULL,
    exit_line      REAL,
    entry_line     REAL,
    action         TEXT NOT NULL,
    trend_action   TEXT,
    trend_distance REAL,
    last_close     REAL,
    status         TEXT,
    UNIQUE (for_bar_ms, config_hash)
);
CREATE INDEX IF NOT EXISTS idx_predictions_bar ON predictions_log(for_bar_ms);
"""

COLUMNS = [
    "recorded_at_ms", "for_bar_ms", "symbol", "interval", "source", "config_hash",
    "model", "drawdown", "horizon_bars", "risk", "rule", "exit_line", "entry_line",
    "action", "trend_action", "trend_distance", "last_close", "status",
]

ACTIONS = {"long", "flat", "keep"}


def config_fingerprint(params: dict[str, Any]) -> str:
    """Stable short hash of everything that defines what the model predicts.

    Two readings are only comparable if they came from the same settings;
    changing the drawdown or the model starts a new series in the log rather
    than silently splicing two different predictors together.
    """
    text = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def reading_from_report(
    report: dict[str, Any],
    *,
    symbol: str,
    interval: str,
    source: str,
    recorded_at_ms: int | None = None,
) -> dict[str, Any]:
    """The log row for a finished risk walk-forward's next-bar reading."""
    today = report.get("today") or {}
    if "for_bar_ms" not in today:
        raise ValueError("report has no next-bar reading (for_bar_ms); rerun the risk job")
    if today.get("action") not in ACTIONS:
        raise ValueError(f"unknown action {today.get('action')!r}")
    params = report["params"]
    trend = today.get("trend") or {}
    return {
        "recorded_at_ms": int(time.time() * 1000) if recorded_at_ms is None else int(recorded_at_ms),
        "for_bar_ms": int(today["for_bar_ms"]),
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "config_hash": config_fingerprint(params),
        "model": str(params.get("model")),
        "drawdown": float(params["drawdown"]),
        "horizon_bars": int(params["horizon_bars"]),
        "risk": today.get("risk"),
        "rule": str(today.get("rule", "hold")),
        "exit_line": today.get("exit_line"),
        "entry_line": today.get("entry_line"),
        "action": today["action"],
        "trend_action": trend.get("action"),
        "trend_distance": trend.get("distance"),
        "last_close": trend.get("close"),
        "status": report.get("status"),
    }


class PredictionLog:
    """Append-only SQLite log. WAL with a busy timeout, because a cron job
    and the server may both append; neither ever updates or deletes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def record(self, reading: dict[str, Any]) -> bool:
        """Insert unless this bar already has a reading from this config.

        Returns True when the row is new. The first reading is kept.
        """
        missing = [column for column in COLUMNS if column not in reading]
        if missing:
            raise ValueError(f"reading missing fields: {missing}")
        placeholders = ", ".join("?" for _ in COLUMNS)
        with self._connect() as conn:
            cursor = conn.execute(
                f"INSERT OR IGNORE INTO predictions_log ({', '.join(COLUMNS)}) "
                f"VALUES ({placeholders})",
                [reading[column] for column in COLUMNS],
            )
            return cursor.rowcount == 1

    def readings(self, config_hash: str | None = None) -> pd.DataFrame:
        query = f"SELECT {', '.join(COLUMNS)} FROM predictions_log"
        args: tuple[Any, ...] = ()
        if config_hash:
            query += " WHERE config_hash = ?"
            args = (config_hash,)
        query += " ORDER BY for_bar_ms, recorded_at_ms"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=args)

    def latest_hash(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT config_hash FROM predictions_log ORDER BY recorded_at_ms DESC, id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None


def prediction_log_path(data_root: str | Path) -> Path:
    return Path(data_root) / "predictions.sqlite"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@dataclass
class PaperReport:
    config_hash: str | None
    readings: int
    on_time: int
    late: int
    first_bar: str | None
    last_bar: str | None
    days: float
    decided: int
    falls: int
    mean_risk: float | None
    brier: float | None
    brier_base_rate: float | None
    auc: float | None
    risk_rule: dict[str, Any]
    trend_rule: dict[str, Any]
    hold: dict[str, Any]
    verdict: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def format(self) -> str:
        def line(name: str, stats: dict[str, Any]) -> str:
            if not stats:
                return f"  {name:<10} –"
            return (
                f"  {name:<10} {stats['total_return'] * 100:+6.2f}% total  "
                f"worst drop {stats['max_drawdown'] * 100:5.1f}%  "
                f"in the market {stats['exposure'] * 100:3.0f}%"
            )

        lines = [
            f"paper record: {self.verdict}",
            f"  {self.readings} readings ({self.on_time} on time, {self.late} late), "
            f"{self.first_bar} -> {self.last_bar} ({self.days:.1f} days)",
            f"  outcomes decided: {self.decided}, falls: {self.falls}",
        ]
        if self.brier is not None:
            lines.append(
                f"  Brier {self.brier:.4f} vs {self.brier_base_rate:.4f} for a constant "
                f"(lower is better)" + (f", AUC {self.auc}" if self.auc is not None else "")
            )
        lines += [line("risk rule", self.risk_rule), line("trend rule", self.trend_rule),
                  line("hold", self.hold)]
        return "\n".join(lines)


def _stamp(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d %H:%M UTC")


def _fill_times(readings: pd.DataFrame, bar_times: np.ndarray, grace_ms: int) -> np.ndarray:
    """When each reading could actually trade: its bar's open if recorded in
    time, otherwise the first open after it was recorded (-1 if none yet)."""
    fills = np.full(len(readings), -1, dtype=np.int64)
    for i, (bar_ms, recorded) in enumerate(
        zip(readings["for_bar_ms"].to_numpy(np.int64), readings["recorded_at_ms"].to_numpy(np.int64))
    ):
        if recorded <= bar_ms + grace_ms:
            fills[i] = bar_ms
            continue
        later = np.searchsorted(bar_times, recorded, side="right")
        if later < len(bar_times):
            fills[i] = bar_times[later]
    return fills


def _paper_positions(
    fills: np.ndarray, actions: list[str], bar_times: np.ndarray
) -> np.ndarray:
    """Position on each bar from the newest reading filled at or before it.

    Between readings the position is held, as a person following the log
    would hold it on days the job did not run. "keep" keeps it.
    """
    positions = np.full(len(bar_times), np.nan)
    order = np.argsort(fills, kind="stable")
    current = np.nan
    pointer = 0
    for i, t in enumerate(bar_times):
        while pointer < len(order) and fills[order[pointer]] <= t:
            action = actions[order[pointer]]
            if action == "long":
                current = 1.0
            elif action == "flat":
                current = 0.0
            elif np.isnan(current):  # "keep" before any position: long by default
                current = 1.0
            pointer += 1
        positions[i] = current
    return positions


def score_paper(
    readings: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    fee_bps: float,
    slippage_bps: float,
    bars_per_year: float,
    grace_ms: int = GRACE_MS,
) -> PaperReport:
    """The paper record of one configuration's readings against stored bars.

    ``readings`` must come from one config (one ``config_hash``): mixing
    settings would score a predictor that never existed.
    """
    hashes = readings["config_hash"].unique() if len(readings) else []
    if len(hashes) > 1:
        raise ValueError("readings from more than one configuration; filter by config_hash")
    empty = PaperReport(
        config_hash=hashes[0] if len(hashes) else None, readings=0, on_time=0, late=0,
        first_bar=None, last_bar=None, days=0.0, decided=0, falls=0, mean_risk=None,
        brier=None, brier_base_rate=None, auc=None, risk_rule={}, trend_rule={}, hold={},
        verdict="No readings yet. Each risk run records one; a record needs months.",
    )
    if not len(readings) or bars.empty:
        return empty

    # One reading per bar: the first one recorded.
    readings = (
        readings.sort_values(["for_bar_ms", "recorded_at_ms"])
        .drop_duplicates("for_bar_ms", keep="first")
        .reset_index(drop=True)
    )
    bar_times = bars["open_time"].to_numpy(np.int64)
    fills = _fill_times(readings, bar_times, grace_ms)
    on_time = int((readings["recorded_at_ms"] <= readings["for_bar_ms"] + grace_ms).sum())

    # Realised outcome of each reading, with the walk-forward's own label.
    drawdown = float(readings["drawdown"].iloc[0])
    horizon = int(readings["horizon_bars"].iloc[0])
    labels = label_drawdown(bars, drawdown, horizon)[["open_time", "label"]]
    scored = readings.merge(
        labels, left_on="for_bar_ms", right_on="open_time", how="left", validate="many_to_one"
    )
    decided = scored[scored["label"].notna() & scored["risk"].notna()]
    y = decided["label"].to_numpy(dtype=float)
    p = decided["risk"].to_numpy(dtype=float)
    brier = brier_base = auc = None
    if len(decided):
        brier = round(float(np.mean((p - y) ** 2)), 5)
        # The constant here knows the realised rate in hindsight, so it is a
        # harder benchmark than any forecaster could face in real time.
        rate = float(y.mean())
        brier_base = round(float(np.mean((rate - y) ** 2)), 5)
        if 0 < y.sum() < len(y):
            auc = _auc(y.astype(int), p)

    # Paper paths from the first fill to the newest bar.
    valid = fills >= 0
    if not valid.any():
        report = empty
        report.readings = len(readings)
        report.on_time = on_time
        report.late = len(readings) - on_time
        report.verdict = "Readings recorded, but no bar has opened after them yet."
        return report
    start = int(np.searchsorted(bar_times, fills[valid].min(), side="left"))
    window = bar_times[start:]
    opens = bars["open"].to_numpy(np.float64)[start:]
    bar_returns = np.zeros(len(window))
    bar_returns[:-1] = opens[1:] / opens[:-1] - 1.0
    cost = (fee_bps + slippage_bps) / 10_000.0

    def path(actions: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
        positions = _paper_positions(fills[valid], actions, window)
        positions = np.nan_to_num(positions, nan=1.0)
        returns = net_returns(positions, bar_returns, cost)
        return positions, performance(returns, positions, bars_per_year)

    risk_actions = [a for a, ok in zip(readings["action"], valid) if ok]
    trend_actions = [
        (a if a in ACTIONS else "long") for a, ok in zip(readings["trend_action"].fillna("long"), valid) if ok
    ]
    _, risk_stats = path(risk_actions)
    _, trend_stats = path(trend_actions)
    _, hold_stats = path(["long"] * int(valid.sum()))

    days = (window[-1] - window[0]) / 86_400_000 if len(window) > 1 else 0.0
    if days < 90:
        verdict = (
            f"{days:.0f} days of paper record. Too short to compare with the "
            "walk-forward; the gate needs months of on-time readings."
        )
    else:
        verdict = (
            f"{days:.0f} days of paper record: risk rule "
            f"{risk_stats['total_return'] * 100:+.1f}%, trend rule "
            f"{trend_stats['total_return'] * 100:+.1f}%, holding "
            f"{hold_stats['total_return'] * 100:+.1f}%, after "
            f"{fee_bps:g} + {slippage_bps:g} bps per side."
        )

    rows = []
    for (_, r), fill in zip(scored.iterrows(), fills):
        rows.append({
            "for_bar": _stamp(r["for_bar_ms"]),
            "recorded_at": _stamp(r["recorded_at_ms"]),
            "on_time": bool(r["recorded_at_ms"] <= r["for_bar_ms"] + grace_ms),
            "filled_at": _stamp(fill) if fill >= 0 else None,
            "risk": None if pd.isna(r["risk"]) else float(r["risk"]),
            "action": r["action"],
            "trend_action": r["trend_action"],
            "fell": None if pd.isna(r["label"]) else bool(r["label"]),
        })

    return PaperReport(
        config_hash=str(readings["config_hash"].iloc[0]),
        readings=len(readings),
        on_time=on_time,
        late=len(readings) - on_time,
        first_bar=_stamp(window[0]),
        last_bar=_stamp(window[-1]),
        days=round(float(days), 2),
        decided=int(len(decided)),
        falls=int(y.sum()) if len(decided) else 0,
        mean_risk=round(float(p.mean()), 4) if len(decided) else None,
        brier=brier,
        brier_base_rate=brier_base,
        auc=auc,
        risk_rule=risk_stats,
        trend_rule=trend_stats,
        hold=hold_stats,
        verdict=verdict,
        rows=rows[-200:],
    )


def paper_record(
    log: PredictionLog,
    bars: pd.DataFrame,
    *,
    fee_bps: float,
    slippage_bps: float,
    bars_per_year: float,
    config_hash: str | None = None,
) -> PaperReport:
    """Score the newest configuration's readings (or ``config_hash``'s)."""
    chosen = config_hash or log.latest_hash()
    readings = log.readings(chosen) if chosen else log.readings().iloc[0:0]
    return score_paper(
        readings, bars, fee_bps=fee_bps, slippage_bps=slippage_bps,
        bars_per_year=bars_per_year,
    )
