"""Shared fixtures: deterministic synthetic bars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

BAR_MS = 300_000


def synthetic_bars(
    n: int = 3000,
    seed: int = 7,
    start_ms: int = 1_700_000_000_000 - (1_700_000_000_000 % BAR_MS),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.0008, size=n)
    close = 2000.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[2000.0], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.0006, size=n))

    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=3.0, sigma=0.5, size=n)
    open_time = start_ms + np.arange(n, dtype=np.int64) * BAR_MS

    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": open_time + BAR_MS - 1,
            "quote_volume": volume * close,
            "trades": rng.integers(1, 100, size=n),
            "taker_buy_base": volume * 0.5,
            "taker_buy_quote": volume * close * 0.5,
        }
    )


def flat_bars(closes: list[float], wick: float = 1.0) -> pd.DataFrame:
    """Bars with a constant ATR: high = close + wick, low = close - wick."""
    close = np.asarray(closes, dtype=float)
    n = len(close)
    open_time = np.arange(n, dtype=np.int64) * BAR_MS + 1_700_000_000_000
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": close.copy(),
            "high": close + wick,
            "low": close - wick,
            "close": close,
            "volume": np.full(n, 10.0),
            "close_time": open_time + BAR_MS - 1,
            "quote_volume": close * 10.0,
            "trades": np.full(n, 5),
            "taker_buy_base": np.full(n, 5.0),
            "taker_buy_quote": close * 5.0,
        }
    )


@pytest.fixture()
def bars() -> pd.DataFrame:
    return synthetic_bars()
