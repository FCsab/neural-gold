"""Volatility sizing: hold less BTC when the next day looks turbulent.

Why this exists: the models here forecast *how much* price will move far
better than *which way* (5-minute barrier hit vs timeout AUC 0.62, direction
0.52; 4h crash risk 0.65). A long-only holder can use a volatility forecast
without any direction call: weight ``min(1, target / forecast)``.

The test was pre-registered in docs/risk.md (2026-11-07) before this ran on
real data. Every constant below is part of that registration; changing one
after seeing the result is choosing on the test period.

* Target: log realised volatility of the six open-to-open log returns a
  position decided at bar ``t`` earns, ``open[t] -> open[t+6]``. Its label is
  decided at bar ``t+6``; training rows reaching into a test block are purged
  (``risk.risk_folds``).
* Model: LightGBM regression, the risk model's regularised tree settings,
  averaged over ``RiskConfig.seeds``. Features: the risk model's default set.
* No-model forecast: EWMA (span 30) of squared close-to-close log returns
  through bar ``t-1``, scaled to six bars. If the model cannot beat this
  formula, the model adds nothing, whatever the sizing earns.
* ``target`` per block is the median forecast over that block's training
  rows: from the past only. A position changes only when the weight moves by
  at least ``BAND``, and every change pays fee + slippage on the amount traded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import REGIME_FIELDS, FeatureConfig, RiskConfig
from .features import build_features, feature_names
from .leakage import LeakageError, check_prefix_invariance
from .risk import (
    _day,
    _lightgbm,
    contiguous_tail,
    net_returns,
    performance,
    placebo_p_value,
    risk_folds,
)

# Pre-registered (docs/risk.md, 2026-11-07). Do not tune.
HORIZON_BARS = 6
EWMA_SPAN = 30
BAND = 0.10
MIN_BLOCK_SHARE = 0.60
MAX_PLACEBO_P = 0.05
MAX_GAP_MS = 7 * 86_400_000


def vol_target(bars: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """``log(sqrt(sum r²))`` of the ``horizon`` open-to-open log returns from
    ``open[t]``, and the bar at which that label is decided (``t + horizon``).

    NaN where the window runs past the last open, or where price did not move
    at all (log of zero is not a volatility).
    """
    opens = bars["open"].to_numpy(dtype=np.float64)
    n = len(opens)
    r = np.full(n, np.nan)
    r[:-1] = np.log(opens[1:] / opens[:-1])
    window = pd.Series(r * r).rolling(horizon, min_periods=horizon).sum()
    summed = window.shift(-(horizon - 1)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(summed > 0, 0.5 * np.log(summed), np.nan)
    return y, np.arange(n, dtype=np.int64) + horizon


def ewma_forecast(bars: pd.DataFrame, horizon: int, span: int) -> np.ndarray:
    """The formula a model has to beat: EWMA of squared close-to-close log
    returns through bar ``t-1``, scaled to ``horizon`` bars, in log-vol units.

    ``adjust=False`` keeps it a pure recursion over the past, so the value at
    ``t`` is the same whether or not later bars exist.
    """
    close = bars["close"].astype("float64")
    r = np.log(close / close.shift(1))
    var = (r * r).ewm(span=span, adjust=False, min_periods=span).mean().shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (0.5 * np.log(horizon * var)).to_numpy(dtype=np.float64)


def size_positions(raw: np.ndarray, band: float) -> np.ndarray:
    """Follow ``raw`` weights, trading only when they move by ``band`` or more.

    Without a band a weight that wobbles by a percent every four hours pays a
    round trip on that percent each time, which is cost with no purpose.
    """
    positions = np.empty(len(raw))
    current: float | None = None
    for i, weight in enumerate(raw):
        if current is None or abs(weight - current) >= band:
            current = float(weight)
        positions[i] = current
    return positions


def _fit_regressor(
    x_train: np.ndarray, y_train: np.ndarray, evaluate: list[np.ndarray], seeds: int
) -> list[np.ndarray]:
    lgb = _lightgbm()
    outputs = [np.zeros(len(x)) for x in evaluate]
    for seed in range(seeds):
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=7,
            max_depth=3,
            min_child_samples=200,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            random_state=seed,
            verbose=-1,
        )
        model.fit(x_train, y_train)
        for out, x in zip(outputs, evaluate):
            out += model.predict(x)
    return [out / seeds for out in outputs]


@dataclass
class VolSizingReport:
    params: dict[str, Any]
    period: dict[str, Any]
    forecast: dict[str, Any]
    hold: dict[str, Any]
    ewma_sized: dict[str, Any]
    model_sized: dict[str, Any]
    criteria: dict[str, bool]
    outcome: str
    verdict: str
    folds: list[dict[str, Any]]
    equity: dict[str, list[float]]
    feature_check: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def format(self) -> str:
        def line(name: str, s: dict[str, Any]) -> str:
            extra = f"  placebo p {s['placebo_p']}" if "placebo_p" in s else ""
            return (
                f"  {name:<12} {s['cagr'] * 100:+6.1f}%/yr  worst drop "
                f"{s['max_drawdown'] * 100:5.1f}%  sharpe {s['sharpe']:.2f}  "
                f"exposure {s['exposure'] * 100:3.0f}%{extra}"
            )

        f = self.forecast
        period = self.period
        lines = [
            f"volatility sizing ({self.outcome}): {self.verdict}",
            f"out of sample {period['start']} -> {period['end']} ({period['years']:.1f} "
            f"years, {period['folds']} blocks), costs {self.params['fee_bps']} + "
            f"{self.params['slippage_bps']} bps per side",
            f"  forecast MSE (log vol): model {f['mse_model']:.4f}, EWMA "
            f"{f['mse_ewma']:.4f}; model better in {f['blocks_model_better']} of "
            f"{f['blocks']} blocks",
            line("hold", self.hold),
            line("EWMA-sized", self.ewma_sized),
            line("model-sized", self.model_sized),
            "  criteria: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in self.criteria.items()),
        ]
        return "\n".join(lines)


def decide(
    forecast: dict[str, Any],
    hold: dict[str, Any],
    ewma: dict[str, Any],
    model: dict[str, Any],
) -> tuple[dict[str, bool], str, str]:
    """The pre-registered rule, applied as written."""
    criteria = {
        "model_forecast_beats_ewma": bool(
            forecast["mse_model"] < forecast["mse_ewma"]
            and forecast["blocks_model_better"] >= MIN_BLOCK_SHARE * forecast["blocks"]
        ),
        "sharpe_above_hold_and_ewma": bool(
            model["sharpe"] > hold["sharpe"] and model["sharpe"] > ewma["sharpe"]
        ),
        "drop_below_hold": bool(model["max_drawdown"] < hold["max_drawdown"]),
        "timing_beats_chance": bool(model["placebo_p"] <= MAX_PLACEBO_P),
    }
    ewma_works = (
        ewma["sharpe"] > hold["sharpe"]
        and ewma["max_drawdown"] < hold["max_drawdown"]
        and ewma["placebo_p"] <= MAX_PLACEBO_P
    )
    if all(criteria.values()):
        return criteria, "works", (
            "Sizing by the model's volatility forecast beat holding on Sharpe and "
            "worst drop, beat the EWMA formula, and its timing beat chance."
        )
    if ewma_works and not criteria["model_forecast_beats_ewma"]:
        return criteria, "formula only", (
            "Sizing by volatility works, but the model forecasts no better than "
            "the EWMA formula: use the formula, not the model."
        )
    failed = [name for name, ok in criteria.items() if not ok]
    return criteria, "does not work", (
        "Not better than holding by the pre-registered rule; failed: " + ", ".join(failed) + "."
    )


def run_vol_sizing(
    bars: pd.DataFrame,
    cfg: RiskConfig,
    base_features: FeatureConfig,
    *,
    bars_per_year: float,
    progress: Callable[[float, str], None] | None = None,
    check_samples: int = 10,
) -> VolSizingReport:
    bars, skipped = contiguous_tail(bars, max_gap_ms=MAX_GAP_MS)
    features_cfg = replace(
        base_features, **{name: tuple(getattr(cfg, name)) for name in REGIME_FIELDS}
    )

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    report(0.0, "checking the features cannot see the future")
    check = check_prefix_invariance(bars, features_cfg, samples=check_samples)
    if not check.passed:
        raise LeakageError(f"feature look-ahead check failed: {check.detail}")

    names = feature_names(features_cfg)
    x = build_features(bars, features_cfg)[names].to_numpy(dtype=np.float64)
    y, exit_index = vol_target(bars, HORIZON_BARS)
    ewma = ewma_forecast(bars, HORIZON_BARS, EWMA_SPAN)

    opens = bars["open"].to_numpy(dtype=np.float64)
    n = len(bars)
    bar_returns = np.zeros(n)
    bar_returns[:-1] = opens[1:] / opens[:-1] - 1.0
    tradable = np.isfinite(x).all(axis=1) & np.isfinite(ewma) & (np.arange(n) < n - 1)
    rows = np.flatnonzero(tradable)
    labelled = np.isfinite(y)
    folds = risk_folds(rows, labelled, exit_index, cfg.min_train_bars, cfg.test_bars)
    if not folds:
        raise ValueError(f"no test block fits in {n} bars; download more history")

    times = bars["open_time"].to_numpy(dtype=np.int64)
    oos = np.concatenate([test for _, test in folds])
    raw_model = np.empty(len(oos))
    raw_ewma = np.empty(len(oos))
    sq_model: list[np.ndarray] = []
    sq_ewma: list[np.ndarray] = []
    fold_rows = []
    start = 0
    for index, (train, test) in enumerate(folds):
        report(0.05 + 0.85 * index / len(folds), f"block {index + 1} of {len(folds)}")
        pred_test, pred_train = _fit_regressor(x[train], y[train], [x[test], x[train]], cfg.seeds)
        model_target = float(np.median(pred_train))
        ewma_target = float(np.median(ewma[train]))
        block = slice(start, start + len(test))
        raw_model[block] = np.minimum(1.0, np.exp(model_target - pred_test))
        raw_ewma[block] = np.minimum(1.0, np.exp(ewma_target - ewma[test]))
        start += len(test)

        scored = labelled[test]
        err_model = (pred_test[scored] - y[test][scored]) ** 2
        err_ewma = (ewma[test][scored] - y[test][scored]) ** 2
        sq_model.append(err_model)
        sq_ewma.append(err_ewma)
        fold_rows.append({
            "fold": index,
            "start": _day(times[test[0]]),
            "end": _day(times[test[-1]]),
            "train_rows": int(len(train)),
            "mse_model": round(float(err_model.mean()), 4) if len(err_model) else None,
            "mse_ewma": round(float(err_ewma.mean()), 4) if len(err_ewma) else None,
            "mean_weight_model": round(float(raw_model[block].mean()), 3),
        })

    cost = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    returns_oos = bar_returns[oos]
    hold_pos = np.ones(len(oos))
    model_pos = size_positions(raw_model, BAND)
    ewma_pos = size_positions(raw_ewma, BAND)
    arms = {}
    for name, pos in (("hold", hold_pos), ("ewma", ewma_pos), ("model", model_pos)):
        r = net_returns(pos, returns_oos, cost)
        stats = performance(r, pos, bars_per_year)
        if name != "hold":
            stats["placebo_p"] = placebo_p_value(pos, returns_oos, cost, stats["sharpe"], bars_per_year)
        arms[name] = (r, stats)

    scored_blocks = [
        (m.mean(), e.mean()) for m, e in zip(sq_model, sq_ewma) if len(m)
    ]
    forecast = {
        "mse_model": round(float(np.concatenate(sq_model).mean()), 5),
        "mse_ewma": round(float(np.concatenate(sq_ewma).mean()), 5),
        "blocks": len(scored_blocks),
        "blocks_model_better": int(sum(m < e for m, e in scored_blocks)),
    }
    criteria, outcome, verdict = decide(
        forecast, arms["hold"][1], arms["ewma"][1], arms["model"][1]
    )

    report(0.95, "summarising")
    step = max(1, int(round(bars_per_year / 365)))
    sample = np.unique(np.r_[np.arange(0, len(oos), step), len(oos) - 1])
    equity = {"time": [int(times[oos[i]] // 1000) for i in sample]}
    for name in arms:
        equity[name] = [round(float(v), 4) for v in np.cumprod(1 + arms[name][0])[sample]]

    report(1.0, "done")
    return VolSizingReport(
        params={
            "horizon_bars": HORIZON_BARS, "ewma_span": EWMA_SPAN, "band": BAND,
            "min_block_share": MIN_BLOCK_SHARE, "max_placebo_p": MAX_PLACEBO_P,
            "test_bars": cfg.test_bars, "min_train_bars": cfg.min_train_bars,
            "seeds": cfg.seeds, "fee_bps": cfg.fee_bps, "slippage_bps": cfg.slippage_bps,
            "features": names,
        },
        period={
            "start": _day(times[oos[0]]), "end": _day(times[oos[-1]]),
            "years": round(len(oos) / bars_per_year, 2), "folds": len(folds),
            "history_start": _day(times[0]), "skipped_gap": skipped,
        },
        forecast=forecast,
        hold=arms["hold"][1],
        ewma_sized=arms["ewma"][1],
        model_sized=arms["model"][1],
        criteria=criteria,
        outcome=outcome,
        verdict=verdict,
        folds=fold_rows,
        equity=equity,
        feature_check={"passed": True, "detail": check.detail},
    )
