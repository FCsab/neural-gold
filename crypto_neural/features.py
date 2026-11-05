"""Leakage-safe feature construction.

Contract: the feature row at position ``t`` is a function of bars
``0 .. t - feature_lag`` only. With the default ``feature_lag = 1`` that
means a feature never sees the bar whose outcome it is paired with, which
is the strict reading of the plan ("no use of the current bar's close in a
feature used to predict that bar's label").

``crypto_neural.leakage`` verifies the contract by recomputing every feature
on a truncated history and asserting bit-for-bit agreement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .indicators import atr, ema, rolling_zscore, rsi


def feature_names(cfg: FeatureConfig) -> list[str]:
    """Names produced by :func:`build_features`, in output order."""
    names = [f"ret_{lb}" for lb in cfg.return_lookbacks]
    names += [f"ewma_vol_{span}" for span in cfg.ewma_vol_spans]
    names += [f"rv_{window}" for window in cfg.realized_vol_windows]
    names += ["rsi", "macd_line", "macd_signal", "macd_hist", "atr_pct"]
    names += ["vol_z", "range_pct", "range_ma", "close_pos"]
    names += [f"ema_dist_{span}" for span in cfg.ema_distances]
    names += [f"mom_{window}" for window in cfg.momentum_windows]
    names += [f"mom_z_{window}" for window in cfg.momentum_z_windows]
    names += [f"sma_dist_{window}" for window in cfg.sma_windows]
    names += [f"drawdown_{window}" for window in cfg.drawdown_windows]
    names += [f"rvol_{window}" for window in cfg.rvol_windows]
    if len(cfg.rvol_windows) >= 2:
        names.append("vol_ratio")
    return names


def required_history(cfg: FeatureConfig) -> int:
    """Bars of history needed before the first complete feature row."""
    needs = [
        max(cfg.return_lookbacks),
        max(cfg.ewma_vol_spans),
        max(cfg.realized_vol_windows),
        cfg.rsi_period,
        sum(cfg.macd[:2]) if len(cfg.macd) >= 3 else max(cfg.macd),
        cfg.atr_period,
        cfg.volume_z_window,
        cfg.range_window,
        max(cfg.ema_distances),
        *(int(window) for window in _regime_windows(cfg)),
    ]
    return max(needs) + cfg.feature_lag + 1


def _regime_windows(cfg: FeatureConfig) -> list[int]:
    return [
        *cfg.momentum_windows,
        *cfg.momentum_z_windows,
        *cfg.sma_windows,
        *cfg.drawdown_windows,
        *cfg.rvol_windows,
    ]


def build_features(bars: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Build the feature frame for ``bars``.

    The returned frame has the same positional index as ``bars``. Rows in
    the warm-up region are NaN by design; callers drop them.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")

    close = bars["close"].astype("float64")
    high = bars["high"].astype("float64")
    low = bars["low"].astype("float64")
    volume = bars["volume"].astype("float64")

    log_close = np.log(close.where(close > 0))
    ret_1 = log_close.diff()

    feats: dict[str, pd.Series] = {}

    for lookback in cfg.return_lookbacks:
        feats[f"ret_{lookback}"] = log_close.diff(lookback)

    for span in cfg.ewma_vol_spans:
        feats[f"ewma_vol_{span}"] = ret_1.ewm(
            span=span, adjust=False, min_periods=span
        ).std()

    for window in cfg.realized_vol_windows:
        feats[f"rv_{window}"] = ret_1.rolling(window, min_periods=window).std()

    feats["rsi"] = (rsi(close, cfg.rsi_period) - 50.0) / 50.0

    if len(cfg.macd) < 3:
        raise ValueError("features.macd must be [fast, slow, signal]")
    fast, slow, signal_period = cfg.macd[0], cfg.macd[1], cfg.macd[2]
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(
        span=signal_period, adjust=False, min_periods=signal_period
    ).mean()
    # Normalised by price so the feature is scale-free across regimes.
    feats["macd_line"] = macd_line / close
    feats["macd_signal"] = macd_signal / close
    feats["macd_hist"] = (macd_line - macd_signal) / close

    feats["atr_pct"] = atr(high, low, close, cfg.atr_period) / close

    feats["vol_z"] = rolling_zscore(volume, cfg.volume_z_window).clip(-10.0, 10.0)

    bar_range = (high - low) / close
    feats["range_pct"] = bar_range
    feats["range_ma"] = bar_range.rolling(cfg.range_window, min_periods=cfg.range_window).mean()

    span = high - low
    feats["close_pos"] = np.where(span > 0, (close - low) / span, 0.5)

    for ema_span in cfg.ema_distances:
        anchor = ema(close, ema_span)
        feats[f"ema_dist_{ema_span}"] = (close - anchor) / anchor

    # Regime features: where price sits in its longer history. Short-horizon
    # indicators above only see about a week; a large fall over the next two
    # months is better explained by trend and volatility over months.
    def rvol(window: int) -> pd.Series:
        return ret_1.rolling(window, min_periods=window).std()

    for window in map(int, cfg.momentum_windows):
        feats[f"mom_{window}"] = log_close.diff(window)
    for window in map(int, cfg.momentum_z_windows):
        # Momentum in units of its own volatility: a 5% move means more in a
        # calm month than in a wild one.
        feats[f"mom_z_{window}"] = log_close.diff(window) / (rvol(window) * np.sqrt(window))
    for window in map(int, cfg.sma_windows):
        sma = close.rolling(window, min_periods=window).mean()
        feats[f"sma_dist_{window}"] = close / sma - 1.0
    for window in map(int, cfg.drawdown_windows):
        peak = close.rolling(window, min_periods=window).max()
        feats[f"drawdown_{window}"] = close / peak - 1.0
    vol_windows = [int(window) for window in cfg.rvol_windows]
    for window in vol_windows:
        feats[f"rvol_{window}"] = rvol(window)
    if len(vol_windows) >= 2:
        feats["vol_ratio"] = feats[f"rvol_{vol_windows[0]}"] / feats[f"rvol_{vol_windows[-1]}"]

    frame = pd.DataFrame(feats, index=bars.index)

    if cfg.feature_lag < 0:
        raise ValueError("feature_lag must be >= 0")
    if cfg.feature_lag:
        frame = frame.shift(cfg.feature_lag)

    # Guard against infinities produced by degenerate windows.
    frame = frame.replace([np.inf, -np.inf], np.nan)

    expected = feature_names(cfg)
    if list(frame.columns) != expected:
        raise AssertionError("feature name/order drift between builder and registry")
    return frame
