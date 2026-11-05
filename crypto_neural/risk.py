"""Drawdown-risk model: stay long, step aside when a large fall is likely.

Why this exists instead of a direction model: on six years of PAXG, every
direction target tried (triple-barrier races, up/down over 5, 20 and 60 days)
scored worse out of sample than guessing the class frequencies, with price,
dollar (EUR/USDT) and BTC features alike. The one target with a stable
out-of-sample signal was the size of the next fall, because volatility
clusters. A long-only holder only needs that: when to step aside.

The evaluation is an expanding walk-forward over 90-day blocks:

* Each block's model is a small seed-ensemble of gradient-boosted trees,
  trained only on rows whose label window closed before the block starts
  (purged by each row's own exit index, so no training label sees a test
  price).
* The exit line is a quantile of the model's own training predictions. Which
  quantile a block uses is chosen by the Sharpe ratio of each candidate over
  the *earlier* blocks only; the first ``warmup_folds`` blocks simply hold.
  Choosing on the block being reported would manufacture the result.
* Positions are decided from features that see bars up to ``t-1`` and are
  executed at the open of bar ``t``. Every switch pays fee + slippage. Open
  positions are valued at the last open, without an exit charge, for every
  strategy alike.

The result is compared against buy-and-hold and a no-model trend rule (long
while the close is above its 200-day average). A rule is only a candidate if
it beats holding on Sharpe *and* has a shallower worst drop.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import REGIME_FIELDS, FeatureConfig, RiskConfig
from .features import build_features, feature_names
from .labels import label_drawdown
from .leakage import LeakageError, check_prefix_invariance

HOLD = 1.0  # candidate key: never step aside


class Stopped(RuntimeError):
    """Raised when the caller asks the walk-forward to stop."""


@dataclass
class RiskReport:
    params: dict[str, Any]
    period: dict[str, Any]
    strategy: dict[str, Any]
    hold: dict[str, Any]
    trend: dict[str, Any]
    candidates: list[dict[str, Any]]
    folds: list[dict[str, Any]]
    prediction: dict[str, Any]
    today: dict[str, Any]
    equity: dict[str, list[float]]
    feature_check: dict[str, Any]
    status: str
    verdict: dict[str, str]
    features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def format(self) -> str:
        def line(name: str, stats: dict[str, Any]) -> str:
            return (
                f"  {name:<10} {stats['cagr'] * 100:+6.1f}%/yr  "
                f"worst drop {stats['max_drawdown'] * 100:5.1f}%  "
                f"sharpe {stats['sharpe']:.2f}  "
                f"exposure {stats['exposure'] * 100:3.0f}%  "
                f"switches/yr {stats['switches_per_year']:.1f}"
            )

        period = self.period
        lines = [
            f"risk model ({self.status}): {self.verdict['text']}",
            f"out of sample {period['start']} -> {period['end']} "
            f"({period['years']:.1f} years, {period['folds']} blocks), "
            f"costs {self.params['fee_bps']} + {self.params['slippage_bps']} bps per side",
            *(
                [f"  history starts {period['history_start']}, after a "
                 f"{period['skipped_gap']['days']}-day gap in the data"]
                if period.get("skipped_gap") else []
            ),
            line("strategy", self.strategy),
            line("hold", self.hold),
            line("trend", self.trend),
            f"  timing vs random exits: p = {self.strategy.get('placebo_p')} "
            f"(trend rule p = {self.trend.get('placebo_p')})",
            f"  prediction AUC {self.prediction['auc']} "
            f"(0.5 = no skill), base rate {self.prediction['base_rate']}",
            f"  next bar ({self.today.get('for_bar', self.today.get('as_of'))}): "
            f"{self.today['text']}",
            *(
                [f"  trend rule: {self.today['trend']['text']}"]
                if self.today.get("trend") else []
            ),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------
def _lightgbm():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "lightgbm is not installed: .venv/bin/pip install lightgbm"
        ) from exc
    return lightgbm


def _fit_lgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    evaluate: list[np.ndarray],
    seeds: int,
) -> list[np.ndarray]:
    """Average P(fall) over a few seeds. Heavily regularised on purpose.

    Rows whose labels overlap carry fewer independent episodes than their
    count suggests; shallow trees with large leaves are what survived out of
    sample.
    """
    lgb = _lightgbm()
    outputs = [np.zeros(len(x)) for x in evaluate]
    for seed in range(seeds):
        model = lgb.LGBMClassifier(
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
            out += model.predict_proba(x)[:, 1]
    return [out / seeds for out in outputs]


def _fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    evaluate: list[np.ndarray],
    cfg: RiskConfig,
) -> list[np.ndarray]:
    """A small neural network, early-stopped on a purged slice of training.

    The last 15% of the training rows are held out for early stopping, and
    the ``horizon_bars`` rows before them are dropped so no fitted label sees
    a held-out price. Inputs are standardised with training statistics only.
    Plain cross-entropy (no class weighting) keeps the output a probability
    on the same scale as the trees', so the two can be averaged.
    """
    import torch
    from torch import nn

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    held = max(1, int(len(x_train) * 0.15))
    fit_end = max(1, len(x_train) - held - cfg.horizon_bars)
    x_fit, y_fit = x_train[:fit_end], y_train[:fit_end]
    x_val, y_val = x_train[-held:], y_train[-held:]
    mean = x_fit.mean(axis=0)
    std = x_fit.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.tensor((values - mean) / std, dtype=torch.float32)

    xf, yf = tensor(x_fit), torch.tensor(y_fit, dtype=torch.float32)
    xv, yv = tensor(x_val), torch.tensor(y_val, dtype=torch.float32)
    evaluate_t = [tensor(x) for x in evaluate]
    loss_fn = nn.functional.binary_cross_entropy_with_logits

    outputs = [np.zeros(len(x)) for x in evaluate]
    for seed in range(cfg.seeds):
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        width = x_train.shape[1]
        for hidden in cfg.mlp_hidden:
            layers += [nn.Linear(width, hidden), nn.GELU(), nn.Dropout(cfg.mlp_dropout)]
            width = hidden
        model = nn.Sequential(*layers, nn.Linear(width, 1))
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        generator = torch.Generator().manual_seed(seed)

        best, best_state, waited = math.inf, None, 0
        for _ in range(cfg.mlp_epochs):
            model.train()
            order = torch.randperm(len(xf), generator=generator)
            for start in range(0, len(xf), 512):
                batch = order[start : start + 512]
                optimiser.zero_grad()
                loss_fn(model(xf[batch]).squeeze(-1), yf[batch]).backward()
                optimiser.step()
            model.eval()
            with torch.no_grad():
                loss = float(loss_fn(model(xv).squeeze(-1), yv))
            if loss < best - 1e-6:
                best, waited = loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                waited += 1
                if waited >= cfg.mlp_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for out, x in zip(outputs, evaluate_t):
                out += torch.sigmoid(model(x).squeeze(-1)).numpy()
    return [out / cfg.seeds for out in outputs]


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    evaluate: list[np.ndarray],
    cfg: RiskConfig,
) -> list[np.ndarray]:
    """P(fall) for each array in ``evaluate``, from the configured model."""
    if len(np.unique(y_train)) < 2:
        # One class only: no information, so predict the base rate everywhere.
        rate = float(y_train.mean()) if len(y_train) else 0.0
        return [np.full(len(x), rate) for x in evaluate]
    if cfg.model == "lgbm":
        return _fit_lgbm(x_train, y_train, evaluate, cfg.seeds)
    if cfg.model == "mlp":
        return _fit_mlp(x_train, y_train, evaluate, cfg)
    trees = _fit_lgbm(x_train, y_train, evaluate, cfg.seeds)
    network = _fit_mlp(x_train, y_train, evaluate, cfg)
    return [(a + b) / 2.0 for a, b in zip(trees, network)]


def risk_positions(
    risk: np.ndarray,
    exit_line: np.ndarray,
    entry_line: np.ndarray,
    min_flat_bars: int,
) -> np.ndarray:
    """1 = long, 0 = flat. Long by default; out above the exit line.

    Re-entry needs risk below a lower line *and* ``min_flat_bars`` spent
    flat, so a reading hovering at the boundary cannot churn the position and
    pay a round trip every few bars.
    """
    positions = np.ones(len(risk))
    long, flat = True, 0
    for i in range(len(risk)):
        if long and risk[i] > exit_line[i]:
            long, flat = False, 0
        elif not long:
            flat += 1
            if risk[i] < entry_line[i] and flat >= min_flat_bars:
                long = True
        positions[i] = 1.0 if long else 0.0
    return positions


def net_returns(positions: np.ndarray, bar_returns: np.ndarray, cost: float) -> np.ndarray:
    """Per-bar returns after paying ``cost`` on every entry and exit."""
    switches = np.abs(np.diff(np.concatenate([[0.0], positions])))
    return positions * bar_returns - switches * cost


def performance(
    returns: np.ndarray, positions: np.ndarray, bars_per_year: float
) -> dict[str, Any]:
    if len(returns) == 0:
        return {
            "total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
            "exposure": 0.0, "switches_per_year": 0.0,
        }
    equity = np.cumprod(1.0 + returns)
    years = len(returns) / bars_per_year
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    final = float(equity[-1])
    std = float(returns.std())
    switches = float(np.abs(np.diff(np.concatenate([[0.0], positions]))).sum())
    return {
        "total_return": round(final - 1.0, 4),
        "cagr": round(final ** (1.0 / years) - 1.0, 4) if final > 0 and years > 0 else -1.0,
        "max_drawdown": round(float((1.0 - equity / peak).max()), 4),
        "sharpe": round(float(returns.mean()) / std * math.sqrt(bars_per_year), 3) if std else 0.0,
        "exposure": round(float(positions.mean()), 4),
        "switches_per_year": round(switches / years, 1) if years > 0 else 0.0,
    }


def risk_folds(
    rows: np.ndarray,
    labelled: np.ndarray,
    exit_index: np.ndarray,
    min_train_bars: int,
    test_bars: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding (train, test) row sets with abutting test blocks.

    Purged: a training row is kept only if its label was decided before the
    test block opens (``exit_index < first test row``). Without this the last
    weeks of training labels are scored on prices inside the test block.
    """
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = int(min_train_bars)
    while cursor + test_bars <= len(rows):
        test = rows[cursor : cursor + test_bars]
        train = rows[:cursor]
        train = train[labelled[train] & (exit_index[train] < test[0])]
        folds.append((train, test))
        cursor += test_bars
    return folds


def contiguous_tail(
    bars: pd.DataFrame, max_gap_ms: int
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Bars after the last gap longer than ``max_gap_ms``.

    A pair can be delisted and relisted (Binance's BTCUSDC was absent from
    2022-09 to 2023-03). Across such a gap a rolling feature mixes two
    unrelated regimes and the first bar after it books months of price change
    as one bar's return. Starting after the gap is the only honest option.
    """
    bars = bars.reset_index(drop=True)
    if len(bars) < 2:
        return bars, None
    steps = np.diff(bars["open_time"].to_numpy(dtype=np.int64))
    large = np.flatnonzero(steps > max_gap_ms)
    if not len(large):
        return bars, None
    cut = int(large[-1]) + 1
    times = bars["open_time"].to_numpy(dtype=np.int64)
    skipped = {
        "from": _day(times[cut - 1]),
        "to": _day(times[cut]),
        "days": round(float(steps[cut - 1]) / 86_400_000, 1),
        "bars_dropped": cut,
    }
    return bars.iloc[cut:].reset_index(drop=True), skipped


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank AUC (Mann-Whitney). None when only one class is present."""
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return round((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 4)


def _day(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d")


def _stamp(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d %H:%M UTC")


@dataclass(frozen=True)
class ExternalFeatures:
    """Feature columns built from a time-stamped table outside the bar store.

    The table is handed over explicitly, not hidden in a closure, so the
    look-ahead check can truncate it: a builder that reads a record from
    after a bar opened is caught even if it never touches a later bar.
    ``build(bars, table)`` must return one row per bar on ``bars.index``.
    """

    name: str
    table: pd.DataFrame
    time_column: str
    build: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]

    def __call__(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.build(bars, self.table)


def _feature_matrix(
    bars: pd.DataFrame, cfg: FeatureConfig, extra: ExternalFeatures | None
) -> pd.DataFrame:
    frame = build_features(bars, cfg)[feature_names(cfg)]
    if extra is None:
        return frame
    more = extra(bars)
    if len(more) != len(bars) or not more.index.equals(bars.index):
        raise ValueError("extra features must have one row per bar, on the bars' index")
    return pd.concat([frame, more], axis=1)


def check_extra_prefix(
    bars: pd.DataFrame, extra: ExternalFeatures, samples: int = 10
) -> tuple[bool, str]:
    """Prefix invariance for features from outside the bar store.

    Row ``i`` is recomputed from ``bars[:i+1]`` *and* only the table records
    stamped strictly before bar ``i`` opened, and must equal row ``i`` of the
    full computation. Truncating the bars alone would not do: a builder can
    leak through the external table without ever looking at a later bar.
    """
    full = extra(bars).to_numpy(dtype=np.float64)
    if len(bars) < 3:
        return True, "too few bars to check"
    stamps = extra.table[extra.time_column].to_numpy(dtype=np.int64)
    opens = bars["open_time"].to_numpy(dtype=np.int64)
    picks = np.unique(np.linspace(len(bars) // 4, len(bars) - 1, samples).astype(int))
    for i in picks:
        known = extra.table[stamps < opens[i]]
        part = extra.build(bars.iloc[: i + 1], known).to_numpy(dtype=np.float64)[-1]
        if not np.array_equal(part, full[i], equal_nan=True):
            return False, (
                f"{extra.name} row {int(i)} changes when later bars or records are hidden"
            )
    return True, (
        f"{len(picks)} rows of {extra.name} unchanged with later bars and records hidden"
    )


def next_bar_features(
    bars: pd.DataFrame, cfg: FeatureConfig, extra: ExternalFeatures | None = None
) -> tuple[np.ndarray, int]:
    """Feature row for the bar that has not opened yet, and its open time.

    The walk-forward's newest row belongs to the newest *stored* bar, whose
    features stop one bar earlier (``feature_lag``). A reading meant to be
    acted on now must use every closed bar, i.e. the row of the next bar.
    That row is built by appending a placeholder bar; with ``feature_lag``
    of at least 1 the placeholder's prices cannot enter it. This is checked,
    not assumed: two different placeholders must give the same row, or the
    reading would depend on prices nobody has seen.
    """
    if cfg.feature_lag < 1:
        raise LeakageError("a next-bar reading needs feature_lag >= 1")
    if len(bars) < 2:
        raise ValueError("need at least two bars")
    times = bars["open_time"].to_numpy(dtype=np.int64)
    step = int(times[-1] - times[-2])
    rows = []
    for scale in (1.0, 3.0):
        last = bars.iloc[[-1]].copy()
        last["open_time"] = times[-1] + step
        for column in ("open", "high", "low", "close", "volume"):
            last[column] = float(bars[column].iloc[-1]) * scale
        extended = pd.concat([bars, last], ignore_index=True)
        rows.append(_feature_matrix(extended, cfg, extra).to_numpy(dtype=np.float64)[-1])
    if not np.array_equal(rows[0], rows[1], equal_nan=True):
        raise LeakageError("the next-bar feature row depends on the unseen bar")
    return rows[0], int(times[-1] + step)


def trend_reading(
    close: pd.Series, sma_bars: int, next_open_ms: int, bars_per_day: float
) -> dict[str, Any]:
    """The no-model trend rule for the next bar: long while the last close is
    above its ``sma_bars`` average. Uses closed bars only."""
    window = f"{sma_bars / bars_per_day:g}-day average"
    if len(close) < sma_bars:
        return {"action": "long", "for_bar": _stamp(next_open_ms), "sma": None,
                "distance": None, "text": f"Not enough history for the {window}; hold."}
    sma = float(close.iloc[-sma_bars:].mean())
    last = float(close.iloc[-1])
    distance = last / sma - 1.0
    long = last > sma
    return {
        "action": "long" if long else "flat",
        "for_bar": _stamp(next_open_ms),
        "sma": round(sma, 2),
        "close": round(last, 2),
        "distance": round(distance, 4),
        "text": (
            f"{'Long' if long else 'Out'}: the last close is {abs(distance):.1%} "
            f"{'above' if long else 'below'} its {window}."
        ),
    }


# --------------------------------------------------------------------------
# the walk-forward
# --------------------------------------------------------------------------
def run_risk_walkforward(
    bars: pd.DataFrame,
    cfg: RiskConfig,
    base_features: FeatureConfig,
    *,
    bars_per_year: float,
    progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    check_samples: int = 10,
    extra_features: ExternalFeatures | None = None,
) -> RiskReport:
    """``extra_features`` adds columns from outside the bar store (e.g.
    funding rates). They pass their own prefix-invariance check first, and
    their names are recorded in ``params`` so a logged reading cannot be
    mistaken for the default model's."""
    # Exchange maintenance (hours, up to ~1.5 days on BTCUSDT in 2018) is not a
    # regime break; a delisting (months) is.
    bars, skipped = contiguous_tail(bars, max_gap_ms=7 * 86_400_000)
    features_cfg = replace(
        base_features, **{name: tuple(getattr(cfg, name)) for name in REGIME_FIELDS}
    )
    names = feature_names(features_cfg)

    def report(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)
        if should_stop and should_stop():
            raise Stopped("stopped")

    # The look-ahead gate runs on the exact feature set this model uses.
    report(0.0, "checking the features cannot see the future")
    check = check_prefix_invariance(bars, features_cfg, samples=check_samples)
    if not check.passed:
        raise LeakageError(f"feature look-ahead check failed: {check.detail}")
    detail = check.detail
    if extra_features is not None:
        passed, extra_detail = check_extra_prefix(bars, extra_features, samples=check_samples)
        if not passed:
            raise LeakageError(f"extra feature look-ahead check failed: {extra_detail}")
        detail = f"{detail}; {extra_detail}"

    feats = _feature_matrix(bars, features_cfg, extra_features)
    names = list(feats.columns)
    labels = label_drawdown(bars, cfg.drawdown, cfg.horizon_bars)
    # Align by time, never by position.
    frame = feats.assign(open_time=bars["open_time"].to_numpy()).merge(
        labels, on="open_time", how="left", validate="one_to_one"
    )
    x = frame[names].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype="float64", na_value=np.nan)
    exit_index = frame["exit_index"].to_numpy(dtype=np.int64)

    opens = bars["open"].to_numpy(dtype=np.float64)
    close = bars["close"].astype("float64")
    n = len(bars)
    # A position decided at row t earns open[t] -> open[t+1]; the last row
    # has no next open, so it cannot be traded.
    bar_returns = np.zeros(n)
    bar_returns[:-1] = opens[1:] / opens[:-1] - 1.0
    tradable = np.isfinite(x).all(axis=1) & (np.arange(n) < n - 1)
    rows = np.flatnonzero(tradable)
    labelled = np.isfinite(y)

    folds = risk_folds(rows, labelled, exit_index, cfg.min_train_bars, cfg.test_bars)
    if len(folds) <= cfg.warmup_folds:
        raise ValueError(
            f"only {len(folds)} test blocks fit in {n} bars; need more than "
            f"{cfg.warmup_folds}. Download more history."
        )

    oos = np.concatenate([test for _, test in folds])
    risk = np.full(n, np.nan)
    cut = {q: np.full(n, np.nan) for q in cfg.exit_quantiles}
    for index, (train, test) in enumerate(folds):
        report(0.05 + 0.8 * index / len(folds), f"block {index + 1} of {len(folds)}")
        p_test, p_train = _fit_predict(x[train], y[train], [x[test], x[train]], cfg)
        risk[test] = p_test
        for q in cfg.exit_quantiles:
            cut[q][test] = np.quantile(p_train, q)

    # Every candidate's full out-of-sample path, then a per-block choice made
    # from the earlier blocks only.
    cost = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    paths = {HOLD: np.ones(len(oos))}
    for q in cfg.exit_quantiles:
        paths[q] = risk_positions(
            risk[oos], cut[q][oos], cut[q][oos] * cfg.reentry_ratio, cfg.min_flat_bars
        )
    path_returns = {q: net_returns(p, bar_returns[oos], cost) for q, p in paths.items()}

    block_slices: list[slice] = []
    start = 0
    for _, test in folds:
        block_slices.append(slice(start, start + len(test)))
        start += len(test)

    def choose(before: int) -> float:
        if before < max(cfg.warmup_folds, 1):
            return HOLD
        seen = np.r_[tuple(block_slices[:before])]
        scores = {
            q: performance(r[seen], paths[q][seen], bars_per_year)["sharpe"]
            for q, r in path_returns.items()
        }
        return max(scores, key=lambda q: (scores[q], q))

    choices = [choose(index) for index in range(len(folds))]
    positions = np.concatenate(
        [paths[q][block] for q, block in zip(choices, block_slices)]
    )
    strategy_returns = net_returns(positions, bar_returns[oos], cost)
    hold_returns = path_returns[HOLD]

    sma = close.rolling(cfg.trend_sma_bars, min_periods=cfg.trend_sma_bars).mean()
    above = (close.shift(1) > sma.shift(1)).to_numpy()
    trend_positions = np.where(sma.shift(1).notna().to_numpy(), above, True)[oos].astype(float)
    trend_returns = net_returns(trend_positions, bar_returns[oos], cost)

    strategy = performance(strategy_returns, positions, bars_per_year)
    hold = performance(hold_returns, paths[HOLD], bars_per_year)
    trend = performance(trend_returns, trend_positions, bars_per_year)

    times = bars["open_time"].to_numpy(dtype=np.int64)
    fold_rows = []
    for index, ((train, test), q, block) in enumerate(zip(folds, choices, block_slices)):
        fold_rows.append(
            {
                "fold": index,
                "start_ms": int(times[test[0]]),
                "end_ms": int(times[test[-1]]),
                "train_rows": int(len(train)),
                "rule": "hold" if q == HOLD else f"q{q}",
                "return": round(float(np.prod(1 + strategy_returns[block]) - 1), 4),
                "hold_return": round(float(np.prod(1 + hold_returns[block]) - 1), 4),
                "exposure": round(float(positions[block].mean()), 3),
                # Per-block skill: a pooled AUC can hide that it all came
                # from one or two blocks.
                "auc": _auc(
                    y[test][labelled[test]].astype(int), risk[test][labelled[test]]
                ),
                "falls": int(np.nansum(y[test])),
            }
        )

    candidates = [
        {"rule": "hold" if q == HOLD else f"q{q}",
         **performance(path_returns[q], paths[q], bars_per_year)}
        for q in sorted(paths)
    ]

    scored = oos[labelled[oos]]
    prediction = {
        "auc": _auc(y[scored].astype(int), risk[scored]),
        "base_rate": round(float(y[scored].mean()), 4) if len(scored) else None,
        "rows": int(len(scored)),
    }

    report(0.9, "fitting today's model on all history")
    x_next, next_open = next_bar_features(bars, features_cfg, extra_features)
    today = _today(x, y, rows, labelled, cfg, choose(len(folds)), x_next, next_open)
    today["trend"] = trend_reading(
        close, cfg.trend_sma_bars, next_open, bars_per_year / 365.0
    )

    step = max(1, int(round(bars_per_year / 365)))
    sample = np.unique(np.r_[np.arange(0, len(oos), step), len(oos) - 1])
    equity = {
        "time": [int(times[oos[i]] // 1000) for i in sample],
        "strategy": [round(float(v), 4) for v in np.cumprod(1 + strategy_returns)[sample]],
        "hold": [round(float(v), 4) for v in np.cumprod(1 + hold_returns)[sample]],
        "trend": [round(float(v), 4) for v in np.cumprod(1 + trend_returns)[sample]],
    }

    placebo = placebo_p_value(
        positions, bar_returns[oos], cost, strategy["sharpe"], bars_per_year
    )
    strategy["placebo_p"] = placebo
    # The trend rule gets the same luck test. Without it "the trend rule beat
    # holding" is a claim about one path through one history.
    trend["placebo_p"] = placebo_p_value(
        trend_positions, bar_returns[oos], cost, trend["sharpe"], bars_per_year
    )
    status, verdict = _verdict(strategy, hold, placebo)
    report(1.0, "done")
    return RiskReport(
        params={
            **asdict(cfg),
            **{name: list(getattr(cfg, name)) for name in REGIME_FIELDS},
            "exit_quantiles": list(cfg.exit_quantiles),
            # Only present when used, so the default model's fingerprint
            # (and its predictions-log series) is unchanged.
            **({"extra_features": names[len(feature_names(features_cfg)):]}
               if extra_features is not None else {}),
        },
        period={
            "start": _day(times[oos[0]]),
            "end": _day(times[oos[-1]]),
            "years": round(len(oos) / bars_per_year, 2),
            "folds": len(folds),
            "history_start": _day(times[0]),
            "skipped_gap": skipped,
        },
        strategy=strategy,
        hold=hold,
        trend=trend,
        candidates=candidates,
        folds=fold_rows,
        prediction=prediction,
        today=today,
        equity=equity,
        feature_check={"passed": True, "detail": detail},
        status=status,
        verdict=verdict,
        features=names,
    )


def _today(
    x: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    labelled: np.ndarray,
    cfg: RiskConfig,
    rule: float,
    x_next: np.ndarray,
    next_open: int,
) -> dict[str, Any]:
    """The reading for the bar that opens next, from a model fitted on every
    decided label.

    Every label in the history is already decided, so nothing here sees the
    future; the newest rows simply have no label yet and are not trained on.
    The row scored is the next bar's (:func:`next_bar_features`), so the
    newest closed bar is used; scoring the newest stored row would ignore it.
    """
    base = {"as_of": _day(next_open), "for_bar": _stamp(next_open),
            "for_bar_ms": int(next_open), "rule": "hold"}
    if not np.isfinite(x_next).all():
        return {**base, "risk": None, "action": "long",
                "text": "Not enough history for a reading; hold."}
    train = rows[labelled[rows]]
    p_latest, p_train = _fit_predict(x[train], y[train], [x_next[None, :], x[train]], cfg)
    risk = float(p_latest[0])
    base["risk"] = round(risk, 4)
    if rule == HOLD:
        return {
            **base,
            "action": "long",
            "text": "Stay long. No exit rule has beaten plain holding so far.",
        }
    exit_line = float(np.quantile(p_train, rule))
    entry_line = exit_line * cfg.reentry_ratio
    base.update(
        rule=f"q{rule}", exit_line=round(exit_line, 4), entry_line=round(entry_line, 4)
    )
    if risk > exit_line:
        return {**base, "action": "flat", "text": (
            f"Step aside. Risk of a {cfg.drawdown:.0%} fall is {risk:.0%}, "
            f"above the exit line of {exit_line:.0%}.")}
    if risk < entry_line:
        return {**base, "action": "long", "text": (
            f"Stay long. Risk of a {cfg.drawdown:.0%} fall is {risk:.0%}, "
            f"below the re-entry line of {entry_line:.0%}.")}
    return {**base, "action": "keep", "text": (
        f"Keep the current position. Risk is {risk:.0%}, between the re-entry "
        f"line ({entry_line:.0%}) and the exit line ({exit_line:.0%}).")}


def placebo_p_value(
    positions: np.ndarray,
    bar_returns: np.ndarray,
    cost: float,
    sharpe: float,
    bars_per_year: float,
    runs: int = 200,
) -> float:
    """Share of randomly shifted copies of the rule that did at least as well.

    Shifting the position path in time keeps its exposure and its number of
    exits, and moves them to arbitrary dates. Being out of the market some of
    the time lowers the worst drop whatever the timing; this asks whether the
    *timing* added anything.
    """
    rng = np.random.default_rng(0)
    if len(positions) < 2 or positions.min() == positions.max():
        return 1.0
    better = 0
    for _ in range(runs):
        shifted = np.roll(positions, int(rng.integers(1, len(positions))))
        result = performance(net_returns(shifted, bar_returns, cost), shifted, bars_per_year)
        better += result["sharpe"] >= sharpe
    # Whole percent: 200 runs cannot resolve finer, and the UI and the
    # verdict text then round the same number the same way.
    return round(better / runs, 2)


def _verdict(
    strategy: dict[str, Any], hold: dict[str, Any], placebo: float
) -> tuple[str, dict[str, str]]:
    returns = (
        f"{strategy['cagr'] * 100:+.1f}% a year against {hold['cagr'] * 100:+.1f}% "
        f"for holding, worst drop {strategy['max_drawdown'] * 100:.0f}% against "
        f"{hold['max_drawdown'] * 100:.0f}%, Sharpe {strategy['sharpe']:.2f} against "
        f"{hold['sharpe']:.2f}"
    )
    luck = (
        f" Random exits at the same frequency did as well {placebo:.0%} of the time."
        if strategy["exposure"] < 1.0
        else ""
    )
    safer = strategy["max_drawdown"] < hold["max_drawdown"]
    better = strategy["sharpe"] > hold["sharpe"]
    beats_luck = placebo < 0.05
    if safer and better and beats_luck:
        if strategy["cagr"] >= hold["cagr"]:
            return "candidate", {"tone": "ok", "text": f"The risk rule beat holding: {returns}.{luck}"}
        return "candidate", {"tone": "warn", "text": (
            f"The risk rule gave up some return for a smaller worst drop: {returns}.{luck}")}
    if safer and better:
        return "draft", {"tone": "warn", "text": (
            f"The risk rule looks better than holding ({returns}), but its timing "
            f"is not distinguishable from luck.{luck}")}
    return "draft", {"tone": "bad", "text": f"The risk rule did not beat holding: {returns}.{luck}"}


def risk_process_entry(
    bars: pd.DataFrame,
    cfg: RiskConfig,
    base_features: FeatureConfig,
    bars_per_year: float,
    messages,
    stop_event,
) -> None:
    """Child-process body: the neural model needs PyTorch, which the server
    deliberately does not import (hundreds of MB, and GPU drivers)."""
    import traceback

    try:
        report = run_risk_walkforward(
            bars,
            cfg,
            base_features,
            bars_per_year=bars_per_year,
            progress=lambda fraction, message: messages.put(
                {"type": "progress", "fraction": fraction, "message": message}
            ),
            should_stop=stop_event.is_set,
        )
        messages.put({"type": "result", "report": report.to_dict()})
    except Stopped:
        messages.put({"type": "stopped"})
    except Exception as exc:  # noqa: BLE001 - process boundary: report, don't die silently
        messages.put(
            {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            }
        )
