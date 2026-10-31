"""Triple-barrier labelling.

For each bar ``t`` whose ATR is known:

    upper_t = close_t + k_up   * ATR_t
    lower_t = close_t - k_down * ATR_t
    timeout = t + max_holding_bars

The first barrier touched inside bars ``t+1 .. t+H`` decides the label:

    +1  profit target hit first
    -1  stop hit first
     0  neither hit, position exits at the close of bar ``t+H``

Two conventions matter for honesty:

* **Intrabar ambiguity.** If both barriers are inside the same bar's
  high/low range we cannot know which came first without tick data. The
  default ``stop_priority`` books that bar as a loss. Being wrong in the
  pessimistic direction costs realism; being wrong optimistically invents
  profits that never existed.
* **Gap fills.** A barrier touched at a gap does not fill at the barrier
  price. A profit target that opens above the barrier fills at the open,
  and a stop that opens below fills at the open. ``realized_return``
  accounts for that, so the label and the return are consistent.

The implementation is vectorised with ``sliding_window_view``: O(n*H)
memory, no Python-level loop over bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig
from .indicators import atr as atr_indicator

LABEL_COLUMNS = [
    "open_time",
    "label",
    "barrier_up",
    "barrier_down",
    "exit_index",
    "exit_time",
    "exit_price",
    "realized_return",
    "holding_bars",
    "valid",
]


def label_triple_barrier(
    bars: pd.DataFrame,
    cfg: LabelConfig,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    """Label every bar that has a complete forward window.

    Rows whose ATR is still in warm-up are returned with ``valid=False``
    and a null label, so callers can see them instead of silently
    receiving a fabricated ``0``.
    """
    required = {"open", "high", "low", "close", "open_time"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")

    n = len(bars)
    horizon = cfg.max_holding_bars

    if n <= horizon:
        return pd.DataFrame(
            {
                "open_time": pd.Series(dtype="int64"),
                "label": pd.array([], dtype="Int8"),
                "barrier_up": pd.Series(dtype="float64"),
                "barrier_down": pd.Series(dtype="float64"),
                "exit_index": pd.Series(dtype="int64"),
                "exit_time": pd.Series(dtype="int64"),
                "exit_price": pd.Series(dtype="float64"),
                "realized_return": pd.Series(dtype="float64"),
                "holding_bars": pd.Series(dtype="int64"),
                "valid": pd.Series(dtype="bool"),
            }
        )

    open_time = bars["open_time"].to_numpy(dtype=np.int64)
    open_ = bars["open"].to_numpy(dtype=np.float64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    close = bars["close"].to_numpy(dtype=np.float64)

    atr = atr_indicator(
        bars["high"].astype("float64"),
        bars["low"].astype("float64"),
        bars["close"].astype("float64"),
        cfg.atr_period,
    ).to_numpy(dtype=np.float64)

    m = n - horizon  # number of bars with a complete forward window

    high_w = np.lib.stride_tricks.sliding_window_view(high, horizon)
    low_w = np.lib.stride_tricks.sliding_window_view(low, horizon)
    open_w = np.lib.stride_tricks.sliding_window_view(open_, horizon)

    labels = np.zeros(m, dtype=np.int8)
    barrier_up = np.full(m, np.nan)
    barrier_down = np.full(m, np.nan)
    exit_index = np.zeros(m, dtype=np.int64)
    exit_price = np.full(m, np.nan)
    realized = np.full(m, np.nan)
    holding = np.full(m, horizon, dtype=np.int64)
    valid = np.zeros(m, dtype=bool)

    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        rows = stop - start

        c = close[start:stop]
        a = atr[start:stop]
        up = c + cfg.k_up * a
        down = c - cfg.k_down * a

        # windows for bar t live at index t+1: high[t+1 : t+1+H]
        hw = high_w[start + 1 : stop + 1]
        lw = low_w[start + 1 : stop + 1]
        ow = open_w[start + 1 : stop + 1]

        up_hit = hw >= up[:, None]
        down_hit = lw <= down[:, None]
        touched = up_hit | down_hit
        any_hit = touched.any(axis=1)
        offset = touched.argmax(axis=1)

        idx = np.arange(rows)
        up_at = up_hit[idx, offset]
        down_at = down_hit[idx, offset]
        if cfg.ambiguity == "stop_priority":
            is_up = up_at & ~down_at
        else:
            is_up = up_at

        open_at = ow[idx, offset]
        fill_up = np.maximum(up, open_at)
        fill_down = np.minimum(down, open_at)
        timeout_price = close[start + horizon : stop + horizon]

        chunk_label = np.where(any_hit & is_up, 1, np.where(any_hit, -1, 0)).astype(np.int8)
        chunk_label[~any_hit] = 0

        chunk_exit = np.where(any_hit, start + idx + 1 + offset, start + idx + horizon)
        chunk_price = np.where(
            any_hit, np.where(is_up, fill_up, fill_down), timeout_price
        )
        chunk_valid = ~np.isnan(a) & ~np.isnan(chunk_price) & (c > 0)

        labels[start:stop] = chunk_label
        barrier_up[start:stop] = up
        barrier_down[start:stop] = down
        exit_index[start:stop] = chunk_exit
        exit_price[start:stop] = chunk_price
        realized[start:stop] = np.where(c > 0, chunk_price / c - 1.0, np.nan)
        holding[start:stop] = chunk_exit - (start + idx)
        valid[start:stop] = chunk_valid

    label_series = pd.array(labels, dtype="Int8")
    label_series[~valid] = pd.NA

    return pd.DataFrame(
        {
            "open_time": open_time[:m],
            "label": label_series,
            "barrier_up": barrier_up,
            "barrier_down": barrier_down,
            "exit_index": exit_index,
            "exit_time": open_time[exit_index],
            "exit_price": exit_price,
            "realized_return": realized,
            "holding_bars": holding,
            "valid": valid,
        }
    )


def label_distribution(labels: pd.DataFrame) -> dict[str, int]:
    """Counts for -1 / 0 / +1 among valid rows."""
    valid = labels.loc[labels["valid"], "label"].dropna().astype(int)
    counts = valid.value_counts().to_dict()
    return {
        "long": int(counts.get(1, 0)),
        "flat": int(counts.get(0, 0)),
        "stop": int(counts.get(-1, 0)),
        "total": int(len(valid)),
    }
