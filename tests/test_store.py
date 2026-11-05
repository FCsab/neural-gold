from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_neural.data.store import BAR_COLUMNS, BarStore, month_key

BAR_MS = 300_000


def _frame(open_times: list[int], base: float = 2000.0) -> pd.DataFrame:
    times = np.asarray(open_times, dtype=np.int64)
    n = len(times)
    close = np.full(n, base)
    return pd.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 2.0),
            "close_time": times + BAR_MS - 1,
            "quote_volume": close * 2.0,
            "trades": np.full(n, 4),
            "taker_buy_base": np.full(n, 1.0),
            "taker_buy_quote": close,
        }
    )


def _times_spanning_two_months() -> list[int]:
    october = int(pd.Timestamp("2023-10-31 22:00", tz="UTC").timestamp() * 1000)
    november = int(pd.Timestamp("2023-11-01 00:00", tz="UTC").timestamp() * 1000)
    return [october + i * BAR_MS for i in range(6)] + [
        november + i * BAR_MS for i in range(6)
    ]


def test_write_partitions_by_month(tmp_path):
    store = BarStore(tmp_path, "BTCUSDC", "5m")
    times = _times_spanning_two_months()
    assert month_key(times[0]) == (2023, 10)
    assert month_key(times[-1]) == (2023, 11)

    store.write_bars(_frame(times))
    partitions = store.partitions()
    assert len(partitions) == 2
    assert [p["month"] for p in partitions] == [10, 11]

    coverage = store.coverage()
    assert coverage.rows == 12
    assert coverage.start_ms == times[0]
    assert coverage.end_ms == times[-1]

    loaded = store.load_bars()
    assert len(loaded) == 12
    assert list(loaded.columns) == BAR_COLUMNS
    assert loaded["open_time"].is_monotonic_increasing


def test_rewrite_is_idempotent_and_last_write_wins(tmp_path):
    store = BarStore(tmp_path, "BTCUSDC", "5m")
    times = _times_spanning_two_months()
    store.write_bars(_frame(times))
    store.write_bars(_frame(times))
    assert store.coverage().rows == 12

    corrected = _frame(times)
    corrected.loc[3, "close"] = 2100.0
    store.write_bars(corrected)
    loaded = store.load_bars()
    assert len(loaded) == 12
    assert float(loaded.loc[3, "close"]) == 2100.0


def test_duplicates_inside_one_batch_collapse(tmp_path):
    store = BarStore(tmp_path, "BTCUSDC", "5m")
    times = [int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)]
    frame = pd.concat([_frame(times), _frame(times)], ignore_index=True)
    frame.loc[1, "close"] = 2050.0
    store.write_bars(frame)
    loaded = store.load_bars()
    assert len(loaded) == 1
    assert float(loaded.loc[0, "close"]) == 2050.0


def test_range_filtering(tmp_path):
    store = BarStore(tmp_path, "BTCUSDC", "5m")
    times = _times_spanning_two_months()
    store.write_bars(_frame(times))
    subset = store.load_bars(start_ms=times[4], end_ms=times[8])
    assert list(subset["open_time"]) == times[4:9]


def test_gap_detection(tmp_path):
    store = BarStore(tmp_path, "BTCUSDC", "5m")
    times = _times_spanning_two_months()
    with_gap = times[:4] + times[6:]
    store.write_bars(_frame(with_gap))
    gaps = store.gaps(BAR_MS)
    assert len(gaps) == 1
    assert gaps[0] == (times[3], times[6])


def test_empty_store_reads_empty(tmp_path):
    store = BarStore(tmp_path, "BTCUSDC", "5m")
    assert store.load_bars().empty
    coverage = store.coverage()
    assert coverage.empty and coverage.rows == 0
    assert store.gaps(BAR_MS) == []


def test_no_temp_files_left_behind(tmp_path):
    store = BarStore(tmp_path, "BTCUSDC", "5m")
    store.write_bars(_frame(_times_spanning_two_months()))
    leftovers = list(store.partitions_dir.rglob("*.tmp"))
    assert leftovers == []
