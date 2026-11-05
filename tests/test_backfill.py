from __future__ import annotations

from crypto_neural.data.backfill import plan_missing_range, plan_missing_windows
from crypto_neural.data.store import Coverage

from .conftest import synthetic_bars

INTERVAL = 300_000
DAY = 86_400_000
NOW = 1_800_000_000_000


def year_ms(years: float) -> int:
    return int(years * 365.25 * DAY)


def test_empty_store_gets_the_whole_request():
    windows = plan_missing_windows(Coverage(None, None, 0), INTERVAL, years=6.1, now_ms=NOW)
    assert len(windows) == 1
    start, end = windows[0]
    assert end == NOW
    assert abs((NOW - start) - year_ms(6.1)) < INTERVAL


def test_short_store_extends_backwards_and_tops_up_the_tail():
    start = NOW - year_ms(2.0)
    coverage = Coverage(start, NOW - 3 * INTERVAL, 210_000)

    windows = plan_missing_windows(coverage, INTERVAL, years=6.1, now_ms=NOW)
    assert len(windows) == 2

    back_start, back_end = windows[0]
    assert back_start < start
    assert back_end == start - INTERVAL

    tail_start, tail_end = windows[1]
    assert tail_start == coverage.end_ms + INTERVAL
    assert tail_end == NOW


def test_store_that_already_covers_the_request_needs_nothing():
    coverage = Coverage(NOW - year_ms(7.0), NOW - INTERVAL, 700_000)
    assert plan_missing_windows(coverage, INTERVAL, years=6.1, now_ms=NOW) == []
    assert plan_missing_range(coverage, INTERVAL, years=6.1, now_ms=NOW) is None


def test_only_the_tail_when_history_is_long_enough():
    coverage = Coverage(NOW - year_ms(3.0), NOW - 5 * INTERVAL, 300_000)
    windows = plan_missing_windows(coverage, INTERVAL, years=2.0, now_ms=NOW)
    assert len(windows) == 1
    assert windows[0] == (coverage.end_ms + INTERVAL, NOW)


def test_only_the_back_when_the_tail_is_current():
    coverage = Coverage(NOW - year_ms(1.0), NOW - INTERVAL, 100_000)
    windows = plan_missing_windows(coverage, INTERVAL, years=3.0, now_ms=NOW)
    assert len(windows) == 1
    assert windows[0][1] == coverage.start_ms - INTERVAL


def test_windows_start_on_the_bar_grid():
    coverage = Coverage(NOW - year_ms(1.0), NOW - INTERVAL, 100_000)
    for start, _end in plan_missing_windows(coverage, INTERVAL, years=4.0, now_ms=NOW):
        assert start % INTERVAL == 0


def test_top_up_never_turns_into_a_full_backfill(tmp_path):
    """A reading's top-up fetches only the tail; an empty or months-old store
    is left for the explicit download button (and no network is touched)."""
    import asyncio
    from dataclasses import replace

    from crypto_neural.config import AppConfig, DataConfig
    from crypto_neural.data import store_from_config, top_up

    config = replace(AppConfig(), data=DataConfig(root=str(tmp_path)))
    assert asyncio.run(top_up(config)) is None  # empty store
    store_from_config(config).write_bars(synthetic_bars(100))  # bars from 2023
    assert asyncio.run(top_up(config)) is None
