"""Leakage-safe feature construction.

Contract: the feature row at position ``t`` is a function of bars
``0 .. t - feature_lag`` only. With the default ``feature_lag = 1`` that
means a feature never sees the bar whose outcome it is paired with, which
is the strict reading of the plan ("no use of the current bar's close in a
feature used to predict that bar's label").

``gold_neural.leakage`` verifies the contract by recomputing every feature
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
    ]
    return max(needs) + cfg.feature_lag + 1


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
